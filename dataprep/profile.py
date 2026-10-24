"""
Stage 5d - profile the TRAIN split only. This profile is all the preprocessing agents see.

Per column:
  kind          numeric / categorical / boolean / datetime / text
  null_pct      share of missing values
  n_unique, unique_ratio   (unique_ratio ~ 1.0 -> identifier-like)
  numeric:      mean, std, min, p1, p25, median, p75, p99, max, skew,
                outlier_pct = share outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
  categorical:  top values with their shares
  target_assoc  strength of the link with the target, 0-1:
                  classification + numeric column     -> correlation ratio (eta)
                  classification + categorical column -> Cramer's V
                  regression + numeric column         -> |Pearson r|
                  regression + categorical column     -> eta of the target across categories
                (near 1.0 on a non-ID column is a leakage warning)

Large tables are profiled on a random sample of profile_max_rows rows.
"""
import math

import numpy as np
import pandas as pd

from .split import detect_datetime


def _r(x, nd=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd)


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "boolean" if s.dropna().nunique() == 2 and set(s.dropna().unique()) <= {0, 1} else "numeric"
    if detect_datetime(s):
        return "datetime"
    nonnull = s.dropna().astype(str)
    if len(nonnull) and nonnull.str.len().mean() > 50 and nonnull.nunique() / len(nonnull) > 0.5:
        return "text"
    return "boolean" if nonnull.nunique() == 2 else "categorical"


def eta(values: pd.Series, groups: pd.Series) -> float | None:
    """Correlation ratio: share of the variance in `values` explained by `groups` (sqrt)."""
    d = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "g": groups}).dropna()
    if len(d) < 3 or d["g"].nunique() < 2:
        return None
    total = ((d["v"] - d["v"].mean()) ** 2).sum()
    if total == 0:
        return 0.0
    stats = d.groupby("g")["v"].agg(["count", "mean"])
    between = (stats["count"] * (stats["mean"] - d["v"].mean()) ** 2).sum()
    return math.sqrt(between / total)


def cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 3:
        return None
    ct = pd.crosstab(d["a"], d["b"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    obs = ct.to_numpy(dtype=float)
    n = obs.sum()
    expected = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / n
    chi2 = float(((obs - expected) ** 2 / expected).sum())
    return math.sqrt(chi2 / (n * (min(ct.shape) - 1)))


def _assoc(s: pd.Series, kind: str, y: pd.Series, classification: bool) -> float | None:
    if kind in ("datetime", "text"):
        return None
    if kind != "numeric" and s.nunique() > 5000:
        return None                                  # too many categories to cross-tabulate
    if classification:
        return eta(s, y) if kind == "numeric" else cramers_v(s.astype(str).where(s.notna()), y)
    if kind == "numeric":
        d = pd.DataFrame({"x": s, "y": pd.to_numeric(y, errors="coerce")}).dropna()
        return abs(float(d["x"].corr(d["y"]))) if len(d) > 2 else None
    return eta(y, s.astype(str).where(s.notna()))


def _numeric_stats(s: pd.Series) -> dict:
    x = s.dropna().astype(float)
    if x.empty:
        return {}
    q = x.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    iqr = q[0.75] - q[0.25]
    lo, hi = q[0.25] - 1.5 * iqr, q[0.75] + 1.5 * iqr
    return {
        "mean": _r(x.mean()), "std": _r(x.std()), "min": _r(x.min()),
        "p1": _r(q[0.01]), "p25": _r(q[0.25]), "median": _r(q[0.5]), "p75": _r(q[0.75]),
        "p99": _r(q[0.99]), "max": _r(x.max()), "skew": _r(x.skew()),
        "outlier_pct": _r(((x < lo) | (x > hi)).mean()),
    }


def profile_frame(train: pd.DataFrame, target: str, task_type: str, max_rows: int = 200_000,
                  seed: int = 42) -> dict:
    sampled = len(train) > max_rows
    df = train.sample(max_rows, random_state=seed) if sampled else train
    y = df[target]
    classification = task_type in ("binary_classification", "multiclass_classification")

    if classification:
        shares = y.value_counts(normalize=True)
        target_info = {"name": target, "kind": "categorical", "n_classes": int(y.nunique()),
                       "class_shares": {str(k): _r(v) for k, v in shares.items()}}
    else:
        target_info = {"name": target, "kind": "numeric", **_numeric_stats(pd.to_numeric(y, errors="coerce"))}

    columns = []
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        kind = column_kind(s)
        n_unique = int(s.nunique(dropna=True))
        nonnull = int(s.notna().sum())
        info = {
            "name": col,
            "kind": kind,
            "dtype": str(s.dtype),
            "null_pct": _r(s.isna().mean()),
            "n_unique": n_unique,
            "unique_ratio": _r(n_unique / nonnull) if nonnull else None,
        }
        if kind == "numeric":
            info.update(_numeric_stats(s))
        elif kind in ("categorical", "boolean"):
            top = s.value_counts(normalize=True, dropna=True).head(5)
            info["top_values"] = {str(k): _r(v) for k, v in top.items()}
        elif kind == "text":
            info["avg_length"] = _r(s.dropna().astype(str).str.len().mean(), 1)
        info["target_assoc"] = _r(_assoc(s, kind, y, classification))
        columns.append(info)

    return {
        "n_rows": int(len(train)),
        "n_rows_profiled": int(len(df)),
        "sampled": sampled,
        "n_columns": len(columns),
        "target": target_info,
        "columns": columns,
    }
