"""
Stage 5b - load, clean and find the target. Everything here runs BEFORE the split,
and none of it learns statistics from the data, so nothing here can leak.

1. Load every downloaded file. Files with identical columns are concatenated; if the files
   disagree, the group of matching files with the most rows is kept.
2. Strip column names. In text columns, missing-value tokens ("", "?", "null", ...) -> NaN.
3. Text columns that are >= numeric_coerce_min numbers become numeric.
     TotalCharges: 11 of 7,043 values are " " -> those 11 become NaN, the other 7,032 parse
     (100% of non-blank values) -> numeric column.
4. Drop exact duplicate rows (must happen before the split, or copies land in train AND test).
5. Find the target column by name; ask the LLM only if no column name matches.
6. Drop rows where the target is missing.
"""
import re

import pandas as pd
from pydantic import BaseModel

from .config import get_llm
from .llm_json import invoke_json
from .spec import ProjectSpec


def load_table(path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".tsv":
        sep = "\t"
    else:
        with open(path, "rb") as f:
            head = f.readline().decode("utf-8", errors="ignore")
        sep = ";" if head.count(";") > head.count(",") else ","
    try:
        return pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, encoding="latin-1", low_memory=False)


def combine(paths, log: list[str]) -> pd.DataFrame:
    frames = {p.name: load_table(p) for p in paths}
    groups: dict[tuple, list[str]] = {}
    for name, df in frames.items():
        key = tuple(str(c).strip() for c in df.columns)
        groups.setdefault(key, []).append(name)
    best = max(groups.values(), key=lambda names: sum(len(frames[n]) for n in names))
    for names in groups.values():
        if names is not best:
            log.append(f"ignored {', '.join(names)} (different columns from {best[0]})")
    if len(best) > 1:
        log.append(f"concatenated {len(best)} files with the same columns: {', '.join(best)}")
    return pd.concat([frames[n] for n in best], ignore_index=True)


def clean_frame(df: pd.DataFrame, cfg: dict, log: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    tokens = {str(t).strip().lower() for t in cfg.get("missing_tokens", ["", "?", "nan", "null", "n/a"])}
    min_ratio = cfg.get("numeric_coerce_min", 0.95)

    for col in df.columns:
        s = df[col]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        # plain object column, NaN kept as NaN, everything else stripped text
        text = s.where(s.isna(), s.astype(str).str.strip())
        is_token = (text.str.lower().isin(tokens) & s.notna()).to_numpy(dtype=bool)
        n_tok = int(is_token.sum())
        if n_tok:
            s = s.mask(is_token)
            text = text.mask(is_token)
            log.append(f"{col}: {n_tok} missing-value tokens -> NaN")

        nonnull = text.dropna()
        if len(nonnull) == 0:
            df[col] = s
            continue
        parsed = pd.to_numeric(nonnull, errors="coerce")
        ratio = float(parsed.notna().mean())
        if ratio >= min_ratio:
            numeric = pd.to_numeric(text, errors="coerce").astype("float64")
            n_bad = int(numeric.isna().sum() - text.isna().sum())
            df[col] = numeric
            extra = f", {n_bad} unparseable -> NaN" if n_bad else ""
            log.append(f"{col}: text -> numeric ({ratio:.1%} of non-blank values parsed{extra})")
        else:
            df[col] = s

    before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    if len(df) < before:
        log.append(f"dropped {before - len(df):,} exact duplicate rows")
    return df


class TargetPick(BaseModel):
    target: str | None
    reason: str = ""


TARGET_SYSTEM = """You pick the target column for a machine-learning project.
Choose the ONE column that holds what the project predicts, from the given column list.
Reply with ONLY JSON: {"target": "<exact column name>" or null, "reason": "<one sentence>"}"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_target(df: pd.DataFrame, spec: ProjectSpec, llm_role: str, log: list[str]) -> str:
    cols = list(df.columns)
    if spec.target_hint:
        t = _norm(spec.target_hint)
        exact = [c for c in cols if _norm(c) == t]
        if exact:
            log.append(f"target: {exact[0]} (column name matches '{spec.target_hint}')")
            return exact[0]
        partial = [c for c in cols if len(t) >= 4 and t in _norm(c)]
        if len(partial) == 1:
            log.append(f"target: {partial[0]} (column name contains '{spec.target_hint}')")
            return partial[0]

    sample = ", ".join(f"{c} (e.g. {df[c].dropna().astype(str).head(3).tolist()})" for c in cols[:80])
    pick = invoke_json(get_llm(llm_role), TARGET_SYSTEM,
                       f"Project: {spec.summary}\nTask: {spec.task_type}\nColumns: {sample}", TargetPick)
    if not pick.target or pick.target not in cols:
        raise ValueError(f"Could not identify the target column. Columns: {cols}")
    log.append(f"target: {pick.target} (chosen by the LLM: {pick.reason})")
    return pick.target


def drop_missing_target(df: pd.DataFrame, target: str, log: list[str]) -> pd.DataFrame:
    missing = int(df[target].isna().sum())
    if missing:
        log.append(f"dropped {missing:,} rows with no {target} value")
        df = df[df[target].notna()].reset_index(drop=True)
    return df
