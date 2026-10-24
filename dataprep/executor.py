"""
Stage 6c - executor: plan.json -> sklearn ColumnTransformer -> fit on TRAIN only -> transform all splits.

Rule 5 (grouping): columns with identical steps share one Pipeline. For Telco, the nine
3-value service columns all get [to_str, one_hot] -> one group instead of nine.

Every statistic (median, percentiles, mean/std, category lists, frequencies) is learned in
.fit(train) and then applied unchanged to val and test - that's what keeps test data out
of the preprocessing.

Writes:
  processed\train.csv, val.csv, test.csv   features + encoded target (last column)
  pipeline.pkl                             fitted transformer, feature names, target classes
  processed\summary.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   PowerTransformer, RobustScaler, StandardScaler)

from .merger import MISSING, STAGE_OF

CLASSIFICATION = ("binary_classification", "multiclass_classification")


# ---------------- custom transformers (module level so pipeline.pkl can be loaded) ----------------

def _remember(est, X):
    est.n_features_in_ = np.asarray(X).shape[1]
    if hasattr(X, "columns"):
        est.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)


def _names(est, input_features):
    if input_features is not None:
        return np.asarray(input_features, dtype=object)
    names = getattr(est, "feature_names_in_", None)
    if names is None:
        names = [f"x{i}" for i in range(est.n_features_in_)]
    return np.asarray(names, dtype=object)


class ToStr(BaseEstimator, TransformerMixin):
    """Categories as strings, NaN -> '__missing__'. Encoders fail on mixed numbers/strings/NaN."""

    def fit(self, X, y=None):
        _remember(self, X)
        return self

    def transform(self, X):
        df = pd.DataFrame(np.asarray(X, dtype=object))
        return df.where(df.notna(), MISSING).astype(str).to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):
        return _names(self, input_features)


class Clipper(BaseEstimator, TransformerMixin):
    """Caps values at the train 1st / 99th percentiles."""

    def __init__(self, low=0.01, high=0.99):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        _remember(self, X)
        a = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(a, self.low, axis=0)
        self.hi_ = np.nanquantile(a, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return _names(self, input_features)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Category -> its share of train rows. Unseen categories -> 0."""

    def fit(self, X, y=None):
        _remember(self, X)
        df = ToStr().fit_transform(X)
        df = pd.DataFrame(df)
        self.maps_ = [df[i].value_counts(normalize=True).to_dict() for i in df.columns]
        return self

    def transform(self, X):
        df = pd.DataFrame(ToStr().fit_transform(X))
        return np.column_stack([df[i].map(m).fillna(0.0).to_numpy(dtype=float)
                                for i, m in enumerate(self.maps_)])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{n}_freq" for n in _names(self, input_features)], dtype=object)


class DateParts(BaseEstimator, TransformerMixin):
    """Date -> year, month, day of week. Unparseable dates -> the train median of that part."""

    PARTS = ("year", "month", "dayofweek")

    def _parts(self, X):
        df = pd.DataFrame(np.asarray(X, dtype=object))
        cols = []
        for i in df.columns:
            d = pd.to_datetime(df[i], errors="coerce", format="mixed")
            cols += [d.dt.year, d.dt.month, d.dt.dayofweek]
        return np.column_stack([c.to_numpy(dtype=float) for c in cols])

    def fit(self, X, y=None):
        _remember(self, X)
        parts = self._parts(X)
        with np.errstate(all="ignore"):
            med = np.nanmedian(parts, axis=0) if len(parts) else np.zeros(parts.shape[1])
        self.medians_ = np.where(np.isnan(med), 0.0, med)
        return self

    def transform(self, X):
        parts = self._parts(X)
        return np.where(np.isnan(parts), self.medians_, parts)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{n}_{p}" for n in _names(self, input_features) for p in self.PARTS], dtype=object)


def safe_log1p(x):
    """log1p, with values below -1 (only possible in val/test) lifted just above -1."""
    return np.log1p(np.clip(np.asarray(x, dtype=float), -0.999999, None))


# ---------------- plan -> sklearn ----------------

def build_step(op: str, params: dict, numeric: bool, n_cols: int):
    if op == "impute_median":
        return SimpleImputer(strategy="median")
    if op == "impute_mean":
        return SimpleImputer(strategy="mean")
    if op == "impute_mode":
        return SimpleImputer(strategy="most_frequent")
    if op == "impute_constant":
        return SimpleImputer(strategy="constant", fill_value=params.get("value", 0 if numeric else MISSING))
    if op == "clip_p1_p99":
        return Clipper()
    if op == "log1p":
        return FunctionTransformer(safe_log1p, feature_names_out="one-to-one")
    if op == "yeo_johnson":
        return PowerTransformer(method="yeo-johnson", standardize=False)
    if op == "one_hot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if op == "binary":
        return OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    if op == "ordinal":
        order = [str(v) for v in params["order"]]
        return OrdinalEncoder(categories=[order] * n_cols, handle_unknown="use_encoded_value", unknown_value=-1)
    if op == "frequency_encode":
        return FrequencyEncoder()
    if op == "parse_datetime":
        return DateParts()
    if op == "standard_scale":
        return StandardScaler()
    if op == "robust_scale":
        return RobustScaler()
    if op == "minmax_scale":
        return MinMaxScaler()
    raise ValueError(f"unknown operation {op}")


def group_columns(plan: dict, numeric_cols: set[str]) -> list[tuple[list[str], list[dict], bool]]:
    """Rule 5: columns with identical step sequences share one pipeline."""
    groups: dict[tuple, list[str]] = {}
    steps_of: dict[tuple, list[dict]] = {}
    for col, steps in plan["columns"].items():
        numeric = col in numeric_cols
        key = (numeric, tuple((s["op"], json.dumps(s["params"], sort_keys=True)) for s in steps))
        groups.setdefault(key, []).append(col)
        steps_of[key] = steps
    return [(cols, steps_of[key], key[0]) for key, cols in groups.items()]


def build_transformer(plan: dict, numeric_cols: set[str]) -> ColumnTransformer:
    transformers = []
    for i, (cols, steps, numeric) in enumerate(group_columns(plan, numeric_cols)):
        if not steps:
            transformers.append((f"g{i}", "passthrough", cols))
            continue
        pipe_steps, has_str = [], False
        for j, s in enumerate(steps):
            if STAGE_OF[s["op"]] == "encode" and not has_str:
                pipe_steps.append(("to_str", ToStr()))
                has_str = True
            pipe_steps.append((f"s{j}_{s['op']}", build_step(s["op"], s["params"], numeric, len(cols))))
        transformers.append((f"g{i}", Pipeline(pipe_steps), cols))
    if plan.get("flags"):
        transformers.append(("missing_flags", MissingIndicator(features="all"), list(plan["flags"])))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def feature_names(ct: ColumnTransformer) -> list[str]:
    try:
        names = ct.get_feature_names_out()
    except ValueError:                                    # two groups produced the same name
        ct.set_params(verbose_feature_names_out=True)
        names = ct.get_feature_names_out()
    out = []
    for n in names:
        n = str(n)
        if n.startswith("missingindicator_"):
            n = n[len("missingindicator_"):] + "_missing"
        out.append(n)
    return out


# ---------------- run ----------------

def encode_target(train: pd.Series, others: dict[str, pd.Series], task_type: str):
    if task_type in CLASSIFICATION:
        classes = sorted(train.astype(str).unique())
        mapping = {c: i for i, c in enumerate(classes)}
        enc = lambda s: s.astype(str).map(mapping).fillna(-1).astype(int)     # -1: class unseen in train
        return enc(train), {k: enc(v) for k, v in others.items()}, mapping
    num = lambda s: pd.to_numeric(s, errors="coerce")
    return num(train), {k: num(v) for k, v in others.items()}, None


def execute(plan: dict, run_dir: Path, task_type: str) -> dict:
    data = run_dir / "data"
    splits = {name: pd.read_csv(data / f"{name}.csv", low_memory=False) for name in ("train", "val", "test")}
    target = plan["target"]
    X = {name: df.drop(columns=[target]) for name, df in splits.items()}
    numeric_cols = {c for c in X["train"].columns if pd.api.types.is_numeric_dtype(X["train"][c])
                    or pd.api.types.is_bool_dtype(X["train"][c])}

    ct = build_transformer(plan, numeric_cols)
    ct.fit(X["train"])
    names = feature_names(ct)

    y_train, y_other, mapping = encode_target(splits["train"][target],
                                              {k: splits[k][target] for k in ("val", "test")}, task_type)
    ys = {"train": y_train, **y_other}

    out_dir = run_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes, nan_after, unseen = {}, {}, {}
    for name in ("train", "val", "test"):
        arr = ct.transform(X[name])
        df = pd.DataFrame(arr, columns=names)
        df[target] = ys[name].to_numpy()
        df.to_csv(out_dir / f"{name}.csv", index=False)
        shapes[name] = [int(df.shape[0]), len(names)]
        nan_after[name] = int(df[names].isna().sum().sum())
        if mapping is not None:
            unseen[name] = int((df[target] == -1).sum())

    joblib.dump({"transformer": ct, "feature_names": names, "target": target, "target_mapping": mapping},
                run_dir / "pipeline.pkl")

    summary = {
        "n_input_columns": len(plan["columns"]) + len(plan["dropped"]),
        "n_features": len(names),
        "feature_names": names,
        "shapes": shapes,
        "nan_after": nan_after,
        "target": target,
        "target_mapping": mapping,
        "unseen_target_rows": unseen,
        "n_groups": len(group_columns(plan, numeric_cols)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
