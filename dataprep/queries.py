"""
Stage 2 - query expansion: ProjectSpec -> a few short keyword queries.

Hub and Kaggle search match WORDS in titles, not meaning, so "predict whether a
customer will leave" finds little while "telecom churn" finds a lot.

On a retry (the best dataset was a weak match), the queries already tried are passed in
as `avoid`, and the model is asked for different, broader ones.
"""
from pydantic import BaseModel, Field

from .config import get_llm, load_config
from .llm_json import invoke_json
from .spec import ProjectSpec

# words that appear in many titles and add nothing to a search
_BANNED = {"dataset", "datasets", "prediction", "predict", "predicting", "classification",
           "classifier", "model", "models", "machine", "learning", "ml", "ai"}

SYSTEM = """You write search queries for dataset search engines (Hugging Face Hub and Kaggle).
These engines match KEYWORDS in dataset titles, not meaning.

Write {n} queries. Rules:
- 2 to 4 words each, using words a dataset creator would put in a title.
- The first query is the most direct one (e.g. "telecom churn").
- The others use synonyms, abbreviations or related names ("telco customer", "customer attrition").
- Never use ML words: prediction, classification, model, dataset, machine learning.
- No quotes, no boolean operators.

Reply with ONLY a JSON object: {{"queries": ["...", "..."]}}"""

RETRY_NOTE = """

These queries were already tried and found only weak matches:
{tried}
Write {n} DIFFERENT queries: broader terms, other synonyms, the wider industry, or the
kind of records the data would contain. Do not repeat or lightly reword the ones above."""


class SearchQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=10)


def _clean(q: str) -> str:
    words = [w for w in q.lower().replace('"', " ").split() if w not in _BANNED]
    return " ".join(words)


def expand_queries(spec: ProjectSpec, avoid: list[str] | None = None) -> list[str]:
    cfg = load_config()["queries"]
    n = cfg.get("n", 4)
    llm = get_llm(cfg["model"])
    avoid = avoid or []

    user = f"Project spec:\n{spec.model_dump_json(indent=2)}"
    if avoid:
        user += RETRY_NOTE.format(tried="\n".join(f"- {q}" for q in avoid), n=n)
    result = invoke_json(llm, SYSTEM.format(n=n), user, SearchQueries)

    out = []
    seen = {_clean(q) for q in avoid}
    candidates = result.queries + spec.domain_keywords    # keywords as fallback if the LLM gives too few
    for q in candidates:
        q = _clean(q)
        if q and q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) == n:
            break
    return out