"""
Stage 5c - train / val / test split.

Two cuts: first test off the whole table, then val off the rest, so the shares come out right:
    test = 15% of 7,043 -> 1,057 rows (sklearn rounds the test part up)
    val  = 0.15 / 0.85 = 17.6% of the remaining 5,986 -> 1,057 rows
    train = 4,929 rows

Classification: stratified, so each split keeps the same class shares (~26.5% churn in all three).
Time-series forecasting: sorted by the date column, oldest -> train, newest -> test (no shuffling,
otherwise the model would see the future).
Everything else: random, with a fixed seed.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from .spec import ProjectSpec

_DATEISH = r"[-/:]"


def detect_datetime(s: pd.Series, min_ratio: float = 0.95) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(200)
    sample = sample[sample.str.contains(_DATEISH, regex=True)]
    if len(sample) < 10:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= min_ratio


def split_frame(df: pd.DataFrame, target: str, spec: ProjectSpec, cfg: dict, log: list[str]):
    fr = cfg.get("split", {"train": 0.70, "val": 0.15, "test": 0.15})
    seed = cfg.get("seed", 42)
    total = fr["train"] + fr["val"] + fr["test"]
    test_share = fr["test"] / total
    val_share = fr["val"] / (fr["train"] + fr["val"])

    if spec.task_type == "time_series_forecasting":
        date_cols = [c for c in df.columns if c != target and detect_datetime(df[c])]
        if date_cols:
            col = date_cols[0]
            order = pd.to_datetime(df[col], errors="coerce", format="mixed").sort_values(kind="stable").index
            df = df.loc[order].reset_index(drop=True)
            n = len(df)
            n_test = int(round(n * test_share))
            n_val = int(round((n - n_test) * val_share))
            train = df.iloc[: n - n_test - n_val]
            val = df.iloc[n - n_test - n_val: n - n_test]
            test = df.iloc[n - n_test:]
            log.append(f"split: chronological by {col} (no shuffling)")
            return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))
        log.append("split: forecasting task but no date column found - falling back to random")

    stratify = None
    if spec.task_type in ("binary_classification", "multiclass_classification"):
        counts = df[target].value_counts()
        if counts.min() >= 3:
            stratify = df[target]
        else:
            log.append(f"split: not stratified - smallest class has only {counts.min()} rows")

    train_val, test = train_test_split(df, test_size=test_share, random_state=seed, stratify=stratify)
    strat2 = train_val[target] if stratify is not None else None
    train, val = train_test_split(train_val, test_size=val_share, random_state=seed, stratify=strat2)
    log.append(f"split: {'stratified' if stratify is not None else 'random'}, seed {seed}")
    return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))
