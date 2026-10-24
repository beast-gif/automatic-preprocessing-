"""
Stage 4 - ranking: pick the dataset.

Only enriched candidates are scored (the top K from search: Kaggle with metadata,
HF with schema).

1. Deterministic score (weights in config.yaml -> rank.weights):
     score = 0.45*coverage + 0.10*modality + 0.15*size + 0.10*license + 0.20*popularity
       coverage   = share of the project's terms (spec keywords, feature hints, target, search
                    queries) that appear ANYWHERE in the title, description or column names.
                    Presence only, so a long description is never diluted, and it's an absolute
                    0-1 number (not scaled to the best in the pool).
                    x0.6 if the dataset has no description AND no schema.
       modality   = HF: 1 schema or tabular evidence, 0.5 data but not tabular, 0 no data files
                    Kaggle: 0.8 (file types are only checked in step 5)
       size       = min(1, log10(rows) / 5); rows = HF exact count, else HF size category,
                    else Kaggle size_bytes / kaggle_bytes_per_row
       license    = 1 open, 0.7 non-commercial, 0.4 restricted / unknown / missing
       popularity = log10(1 + downloads) / log10(1 + max(max downloads in the same source, floor))
     x0.5 if the title contains a bad word (corrupted, fake, ...)

2. LLM re-rank of the top N: the smart model reads descriptions, column lists and download
   counts and returns fit (0-10) and whether the target column exists. If the schema has a
   column matching the spec's target, has_target is set to "yes" whatever the LLM said.
     final = (1 - mix)*score + mix*fit/10,  x0.5 if has_target == "no"

The pick is always one the LLM has judged. weak=True if its final is under weak_threshold.
"""
import math
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .candidate import Candidate
from .config import get_llm, load_config
from .llm_json import invoke_json
from .spec import ProjectSpec

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "and", "for", "with", "from", "data", "dataset", "datasets", "set"}
_OPEN = ("cc0", "public domain", "pddl", "cc-by", "cc by", "mit", "apache", "bsd", "odbl",
         "odc", "gpl", "cdla", "openrail", "unlicense")
_TABULAR_FORMATS = {"csv", "parquet", "tsv", "excel", "sqlite"}


# ---------------- deterministic parts ----------------

def license_score(lic: Optional[str]) -> float:
    if not lic:
        return 0.4
    l = lic.lower()
    tokens = l.replace("-", " ").replace("_", " ").split()
    if "nc" in tokens or "noncommercial" in l or "non-commercial" in l:   # check before cc-by
        return 0.7
    if any(k in l for k in _OPEN):
        return 1.0
    return 0.4                                         # copyright-authors, other, unknown


def est_rows(c: Candidate, bytes_per_row: float) -> Optional[int]:
    if c.rows_exact:
        return c.rows_exact
    if c.rows_est:
        return c.rows_est
    if c.size_bytes:
        return max(1, int(c.size_bytes / bytes_per_row))
    return None


def size_score(rows: Optional[int]) -> float:
    if not rows or rows < 1:
        return 0.0
    return min(1.0, math.log10(rows) / 5)


def modality_score(c: Candidate, spec: ProjectSpec) -> float:
    if c.source == "kaggle":
        return 0.8
    mods = {m.lower() for m in c.modalities}
    fmts = {f.lower() for f in c.formats}
    if spec.modality == "tabular" and c.columns:
        return 1.0                                     # it has a table schema
    if spec.modality in mods or (spec.modality == "tabular" and fmts & _TABULAR_FORMATS):
        return 1.0
    if mods or fmts:
        return 0.5
    return 0.0                                         # no format/modality tags: likely no data files


def _humanize(col: str) -> str:
    """MonthlyCharges -> Monthly Charges, customer_id -> customer id (so words can match)."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", col)
    return re.sub(r"[_\-.]+", " ", s)


def terms(text: str) -> set[str]:
    """Lowercase word set; plural s stripped so 'charges' matches 'charge'."""
    out = set()
    for t in _TOKEN.findall(text.lower()):
        if len(t) < 3 or t in _STOP:
            continue
        out.add(t[:-1] if t.endswith("s") and len(t) > 4 else t)
    return out


def want_terms(spec: ProjectSpec, queries: list[str] | None = None) -> set[str]:
    parts = spec.domain_keywords + spec.feature_hints + (queries or [])
    if spec.target_hint:
        parts.append(spec.target_hint.replace("_", " "))
    return terms(" ".join(parts))


def cand_text(c: Candidate) -> str:
    cols = " ".join(_humanize(col["name"]) for col in c.columns)
    return f"{c.title} {re.sub(r'[/_-]', ' ', c.id)} {c.description} {cols}"


def coverage(want: set[str], c: Candidate) -> float:
    if not want:
        return 0.0
    return len(want & terms(cand_text(c))) / len(want)


def has_info(c: Candidate) -> bool:
    return bool(c.description.strip()) or bool(c.columns)


def target_in_columns(c: Candidate, spec: ProjectSpec) -> Optional[bool]:
    """True if a column matches the target ("churn" ~ "Churn", "churn_flag"); None if no schema."""
    if not c.columns or not spec.target_hint:
        return None
    t = re.sub(r"[^a-z0-9]", "", spec.target_hint.lower())
    for col in c.columns:
        n = re.sub(r"[^a-z0-9]", "", col["name"].lower())
        if n == t or (len(t) >= 4 and t in n):
            return True
    return False          # the target may still exist under another name (e.g. "Exited")


# ---------------- results ----------------

class Scored(BaseModel):
    candidate: Candidate
    rows_est: Optional[int] = None
    parts: dict[str, float]
    penalized: bool = False
    score: float
    fit: Optional[float] = None
    has_target: Optional[str] = None
    reason: str = ""
    final: float = 0.0


class RankResult(BaseModel):
    pick: Optional[Scored]
    ranked: list[Scored]
    weak: bool


def score_candidates(cands: list[Candidate], spec: ProjectSpec, cfg: dict,
                     queries: list[str] | None = None) -> list[Scored]:
    pool = [c for c in cands if c.enriched]
    if not pool:
        return []
    w = cfg["weights"]
    bpr = cfg.get("kaggle_bytes_per_row", 30)
    bad_words = {b.lower() for b in cfg.get("bad_title_words", [])}
    floor = cfg.get("popularity_floor", 10000)
    no_info = cfg.get("no_info_similarity_factor", 0.6)
    want = want_terms(spec, queries)

    max_dl = {}
    for c in pool:
        max_dl[c.source] = max(max_dl.get(c.source, 0), c.downloads)

    out = []
    for c in pool:
        cov = coverage(want, c)
        if not has_info(c):
            cov *= no_info
        rows = est_rows(c, bpr)
        top = max(max_dl[c.source], floor)
        pop = math.log10(1 + c.downloads) / math.log10(1 + top)
        parts = {
            "coverage": round(cov, 3),
            "modality": modality_score(c, spec),
            "size": round(size_score(rows), 3),
            "license": license_score(c.license),
            "popularity": round(pop, 3),
        }
        s = sum(w[k] * v for k, v in parts.items())
        bad = bool(bad_words & set(_TOKEN.findall(f"{c.title} {c.id}".lower())))
        if bad:
            s *= 0.5
        s = round(s, 4)
        out.append(Scored(candidate=c, rows_est=rows, parts=parts, penalized=bad, score=s, final=s))
    out.sort(key=lambda x: x.score, reverse=True)
    return out


# ---------------- LLM re-rank ----------------

RERANK_SYSTEM = """You judge which public dataset best fits a machine-learning project.
Use only the information given about each dataset.

For every dataset return:
- "fit": 0 to 10. 10 = exactly the data the project needs, with the target column and realistic
  records. 5 = related, but the target or key features are unclear. 0 = wrong kind of data.
  Lower the score for synthetic, toy, corrupted or already feature-engineered data when the
  description says so, and for small re-uploads of a dataset that appears elsewhere in the list.
  Download counts show how widely a dataset is used; a widely used original is usually the
  safer choice than a little-used copy.
- "has_target": "yes" if the description or columns show the project's target,
  "no" if it clearly has none, "unknown" if the information doesn't say.
- "reason": one short sentence.

Reply with ONLY a JSON object, one entry per dataset, using the given ids:
{"judgements": [{"id": "owner/name", "fit": 8, "has_target": "yes", "reason": "..."}]}"""


class Judgement(BaseModel):
    id: str
    fit: float = Field(ge=0, le=10)
    has_target: Literal["yes", "no", "unknown"] = "unknown"
    reason: str = ""


class Rerank(BaseModel):
    judgements: list[Judgement]


def _dataset_block(s: Scored) -> str:
    c = s.candidate
    if c.rows_exact:
        rows = f"{c.rows_exact:,} (exact)"
    else:
        rows = f"~{s.rows_est:,} (estimate)" if s.rows_est else "unknown"
    dl_kind = "all-time" if c.source == "kaggle" else "recent"
    desc = c.description.strip()[:1200] or "(no description)"
    block = (f"id: {c.id}\nsource: {c.source}\ntitle: {c.title}\nlicense: {c.license or 'none listed'}\n"
             f"downloads: {c.downloads:,} ({dl_kind})\nrows: {rows}\ndescription: {desc}")
    if c.columns:
        cols = ", ".join(f"{col['name']} ({col['dtype']})" for col in c.columns[:40])
        more = f" ... and {len(c.columns) - 40} more" if len(c.columns) > 40 else ""
        block += f"\ncolumns ({len(c.columns)}): {cols}{more}"
    return block


def rerank(top: list[Scored], spec: ProjectSpec, cfg: dict) -> None:
    """Fills fit / has_target / reason / final on the given items (in place)."""
    llm = get_llm(cfg.get("rerank_model", "smart"))
    user = (f"Project spec:\n{spec.model_dump_json(indent=2)}\n\nDatasets:\n\n"
            + "\n\n---\n\n".join(_dataset_block(s) for s in top))
    result = invoke_json(llm, RERANK_SYSTEM, user, Rerank)
    by_id = {j.id.strip(): j for j in result.judgements}
    mix = cfg.get("final_mix", 0.5)
    for s in top:
        j = by_id.get(s.candidate.id)
        if not j:
            s.reason = "(not judged by the LLM)"
            continue
        s.fit, s.has_target, s.reason = j.fit, j.has_target, j.reason
        if target_in_columns(s.candidate, spec):        # the schema is harder evidence than the LLM
            s.has_target = "yes"
        final = (1 - mix) * s.score + mix * j.fit / 10
        if s.has_target == "no":
            final *= 0.5
        s.final = round(final, 4)


def rank_candidates(cands: list[Candidate], spec: ProjectSpec,
                    queries: list[str] | None = None) -> RankResult:
    cfg = load_config()["rank"]
    scored = score_candidates(cands, spec, cfg, queries)
    if not scored:
        return RankResult(pick=None, ranked=[], weak=True)

    n = cfg.get("rerank_top_n", 8)
    top, rest = scored[:n], scored[n:]
    rerank(top, spec, cfg)
    top.sort(key=lambda s: s.final, reverse=True)

    pick = top[0]
    return RankResult(pick=pick, ranked=top + rest, weak=pick.final < cfg.get("weak_threshold", 0.6))