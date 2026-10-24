"""
Stage 1 - requirement parser: plain-language idea -> ProjectSpec.
"""
from .config import get_llm, load_config
from .llm_json import invoke_json
from .spec import ProjectSpec

SYSTEM = """You turn a short machine-learning project idea into a structured spec.
The spec is used to search Hugging Face, Kaggle and OpenML for a public dataset.

Reply with ONLY a JSON object with exactly these keys:

- "summary": one sentence restating the project.
- "task_type": one of "binary_classification", "multiclass_classification", "regression",
  "time_series_forecasting", "clustering", "anomaly_detection".
- "modality": the kind of input data: "tabular", "text", "image", "audio" or "video".
  Rows-and-columns data (including time-indexed tables) is "tabular".
- "target_hint": what is predicted, as a likely column name in snake_case (e.g. "churn",
  "house_price"). null for clustering.
- "domain_keywords": 3 to 6 short search terms describing the DOMAIN and DATA, not the ML
  method. Never use words like "classification", "prediction", "model", "machine learning".
  Include common synonyms (e.g. "churn", "customer attrition").
- "feature_hints": input variables the user mentioned or that the task obviously needs,
  at most 8. Empty list if unclear.
- "min_rows": smallest dataset size that would be usable. 1000 unless the user says otherwise.
- "constraints": ONLY constraints the user actually stated (license, country/region,
  language, time period). Empty list if none. Never invent constraints.

Example
Idea: "predict whether a telecom customer will leave, preferably Indian data"
{"summary": "Predict whether a telecom customer will churn.",
 "task_type": "binary_classification", "modality": "tabular", "target_hint": "churn",
 "domain_keywords": ["telecom", "customer churn", "customer attrition", "subscriber"],
 "feature_hints": ["tenure", "monthly charges", "contract type"],
 "min_rows": 1000, "constraints": ["Indian data preferred"]}"""


def parse_requirements(idea: str) -> ProjectSpec:
    idea = idea.strip()
    if not idea:
        raise ValueError("Project idea is empty")
    cfg = load_config()
    llm = get_llm(cfg["parser"]["model"])
    return invoke_json(llm, SYSTEM, f"Idea: {idea}", ProjectSpec)


def scope_problem(spec: ProjectSpec) -> str | None:
    """Returns a message if this version can't handle the spec, else None."""
    allowed = load_config()["scope"]["modalities"]
    if spec.modality not in allowed:
        return (f"This looks like a {spec.modality} project; this version only supports "
                f"{', '.join(allowed)} data.")
    return None
