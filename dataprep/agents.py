"""
Stage 6a - the four preprocessing specialists.

Each specialist is ONE LLM call. It reads the train profile (statistics only, never raw rows)
and proposes operations for the columns it is responsible for. The four run in parallel
(LangGraph Send in the graph; a thread pool in scripts/try_preprocess.py).

To save tokens and mistakes, each specialist only sees the columns and fields it needs:
  missing       columns with null_pct > 0
  distribution  numeric columns (not 0/1 booleans)
  encoding      non-numeric columns
  leakage       every column
If a specialist has no columns to look at, no LLM call is made.

Each specialist may only use its own operations (ALLOWED); the merger rejects anything else.
"""
import json

from pydantic import BaseModel, Field

from .config import get_llm, load_config
from .llm_json import invoke_json
from .spec import ProjectSpec

AGENTS = ["missing", "distribution", "encoding", "leakage"]

ALLOWED = {
    "missing": {"impute_median", "impute_mean", "impute_mode", "impute_constant",
                "add_missing_flag", "drop_column"},
    "distribution": {"clip_p1_p99", "log1p", "yeo_johnson",
                     "standard_scale", "robust_scale", "minmax_scale"},
    "encoding": {"one_hot", "binary", "ordinal", "frequency_encode", "parse_datetime", "drop_column"},
    "leakage": {"drop_column"},
}

FIELDS = {
    "missing": ["name", "kind", "dtype", "null_pct", "n_unique", "mean", "median", "skew", "outlier_pct"],
    "distribution": ["name", "null_pct", "mean", "std", "min", "p1", "median", "p99", "max",
                     "skew", "outlier_pct"],
    "encoding": ["name", "kind", "dtype", "null_pct", "n_unique", "unique_ratio", "top_values", "avg_length"],
    "leakage": ["name", "kind", "dtype", "null_pct", "n_unique", "unique_ratio", "target_assoc", "top_values"],
}

_NUMERIC_PREFIXES = ("int", "uint", "float", "Int", "UInt", "Float")


def is_numeric(col: dict) -> bool:
    """Numeric storage type (0/1 and True/False columns count as numeric: they need no encoding)."""
    dtype = str(col.get("dtype", ""))
    return dtype.startswith(_NUMERIC_PREFIXES) or dtype in ("bool", "boolean")


def columns_for(agent: str, profile: dict) -> list[dict]:
    cols = profile["columns"]
    if agent == "missing":
        cols = [c for c in cols if (c.get("null_pct") or 0) > 0]
    elif agent == "distribution":
        cols = [c for c in cols if c["kind"] == "numeric" and is_numeric(c)]
    elif agent == "encoding":
        cols = [c for c in cols if not is_numeric(c)]
    return [{k: c[k] for k in FIELDS[agent] if c.get(k) is not None} for c in cols]


# ---------------- prompts ----------------

COMMON = """You are the {role} specialist in a data-preprocessing team for a machine-learning project.
Project: {summary}
Task: {task_type}. Target column: {target} - never propose anything for it.
Training rows: {n_rows}.

You see a statistical profile of the TRAINING split, not the raw data. Fractions are 0-1
(null_pct 0.18 = 18% missing; outlier_pct = share outside Q1-1.5*IQR .. Q3+1.5*IQR).

Use ONLY these operations: {ops}

Rules:
{rules}

Reply with ONLY a JSON object:
{{"proposals": [{{"column": "<exact column name>", "op": "<operation>", "params": {{}}, "reason": "<short, cite the numbers>"}}]}}
One entry per operation (a column can have several). Leave out columns that need nothing.
An empty list is a valid answer.

Columns:
{columns}"""

ROLES = {
    "missing": "missing-values",
    "distribution": "distribution and scaling",
    "encoding": "categorical-encoding",
    "leakage": "leakage and data-quality",
}

RULES = {
    "missing": """- Only columns with missing values are listed.
- null_pct > {drop_null} -> drop_column (too little data to fill reliably).
- Numeric column: impute_median if |skew| > 1 or outlier_pct > 0.01 (the mean is pulled by the tail),
  otherwise impute_mean.
- Categorical / boolean text column: impute_mode if null_pct < 0.05; otherwise impute_constant with
  params {{"value": "__missing__"}} so "missing" becomes its own category.
- Also add_missing_flag when null_pct >= 0.05: whether a value was missing can itself predict the target.""",

    "distribution": """- Only numeric columns are listed.
- clip_p1_p99 when outlier_pct > 0.01 AND |skew| > 1 (caps extreme values at the 1st / 99th percentile).
- skew > 1: log1p if the smallest value (p1 when you also clip, else min) is greater than -1,
  otherwise yeo_johnson. skew < -1: yeo_johnson. |skew| <= 1: no transform.
- Scale every listed column exactly once: robust_scale if outlier_pct > 0.05 and you are NOT clipping
  it, otherwise standard_scale.""",

    "encoding": """- Only non-numeric columns are listed.
- 2 distinct values -> binary.
- 3 to {max_oh} distinct values -> one_hot.
- More than {max_oh} distinct values -> frequency_encode.
- If the values have a natural order (Low/Medium/High, education levels, sizes, ratings), use ordinal
  with params {{"order": ["<lowest>", ..., "<highest>"]}} using the exact value strings.
- kind "datetime" -> parse_datetime (becomes year, month, day-of-week numbers).
- kind "text" (long free text) -> drop_column; this version does not process free text.""",

    "leakage": """- Drop identifiers: categorical or text columns with unique_ratio >= 0.95, and columns whose name
  says id / key / uuid / number and that are nearly all unique. They let a model memorise rows.
  A numeric measurement with many distinct values (e.g. an amount) is NOT an identifier.
- Drop target leaks: columns only known AFTER the outcome (e.g. churn reason, refund, cancellation date,
  final status, score derived from the target), and any non-identifier column with target_assoc >= 0.9.
- target_assoc between 0.6 and 0.9: drop only if the name or values suggest post-outcome information;
  otherwise keep it - strong genuine predictors are what the model needs.
- Drop constant columns (n_unique <= 1).
- Never drop a column just because it is weakly related to the target.""",
}


class Proposal(BaseModel):
    column: str
    op: str
    params: dict = Field(default_factory=dict)
    reason: str = ""


class AgentOutput(BaseModel):
    proposals: list[Proposal] = Field(default_factory=list)


def build_prompt(agent: str, profile: dict, spec: ProjectSpec, cols: list[dict]) -> str:
    pcfg = load_config()["preprocess"]
    rules = RULES[agent].format(drop_null=pcfg.get("drop_if_null_above", 0.6),
                                max_oh=pcfg.get("one_hot_max_categories", 15))
    return COMMON.format(
        role=ROLES[agent], summary=spec.summary, task_type=spec.task_type,
        target=profile["target"]["name"], n_rows=profile["n_rows"],
        ops=", ".join(sorted(ALLOWED[agent])), rules=rules,
        columns=json.dumps(cols, ensure_ascii=False),
    )


def run_agent(agent: str, profile: dict, spec: ProjectSpec) -> list[dict]:
    """Returns this specialist's proposals as dicts with an 'agent' field. [] if nothing to look at."""
    cols = columns_for(agent, profile)
    if not cols:
        return []
    llm = get_llm(load_config()["agents"].get("model", "fast"))
    system = build_prompt(agent, profile, spec, cols)
    out = invoke_json(llm, system, "Return your proposals now.", AgentOutput)
    return [{"agent": agent, **p.model_dump()} for p in out.proposals]
