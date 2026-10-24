"""
Stage 3 - multi-source search.

1. Fan out: every query x every source, all calls in parallel on persistent sessions.
2. Parse each result (no LLM) and merge duplicates by source:id, recording which
   queries found each dataset (query_hits).
3. Pre-rank with free signals only:
       pre_score = 0.5*title_overlap + 0.3*popularity + 0.2*(query_hits / n_queries)
   title_overlap = share of the spec's keyword + query tokens found in the title/id
   popularity    = log10(1 + downloads) / log10(1 + max downloads FROM THE SAME SOURCE)
4. Enrich only the top K, all calls in parallel:
     Kaggle -> dataset_metadata       (description, license, usability)
     HF     -> hub_repo_details with dataset_structure  (exact rows, schema, parquet URLs)
   Only enriched candidates are scored in stage 4.
"""
import asyncio
import math
import re

from .candidate import Candidate
from .config import load_config
from .mcp_client import describe, open_tools
from .parse_results import (parse_hf_search, parse_hf_structure, parse_kaggle_metadata,
                            parse_kaggle_search, result_text)
from .spec import ProjectSpec

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return {t for t in _TOKEN.findall(s.lower()) if len(t) >= 3}


def pre_rank(cands: list[Candidate], spec: ProjectSpec, queries: list[str]) -> None:
    n_queries = len(queries)
    want = set()
    # query words included so synonyms the LLM found ("telco") count, not only "telecom"
    for k in spec.domain_keywords + queries + ([spec.target_hint] if spec.target_hint else []):
        want |= _tokens(k.replace("_", " "))

    max_dl = {}
    for c in cands:
        max_dl[c.source] = max(max_dl.get(c.source, 0), c.downloads)

    for c in cands:
        have = _tokens(f"{c.title} {c.id}")
        overlap = len(want & have) / len(want) if want else 0.0
        top = max_dl.get(c.source, 0)
        pop = math.log10(1 + c.downloads) / math.log10(1 + top) if top > 0 else 0.0
        hits = len(c.query_hits) / n_queries if n_queries else 0.0
        c.pre_score = round(0.5 * overlap + 0.3 * pop + 0.2 * hits, 4)


async def _search_calls(tools: dict, queries: list[str], limit: int):
    hf = tools.get("huggingface", {}).get("hub_repo_search")
    kg = tools.get("kaggle", {}).get("datasets_list")
    jobs = []
    for q in queries:
        if hf:
            jobs.append(("huggingface", q, hf.ainvoke({"query": q, "repo_types": ["dataset"], "limit": limit})))
        if kg:
            jobs.append(("kaggle", q, kg.ainvoke({"search": q})))
    results = await asyncio.gather(*(j[2] for j in jobs), return_exceptions=True)
    return [(source, q, res) for (source, q, _), res in zip(jobs, results)]


async def _enrich_kaggle(tools: dict, cands: list[Candidate], warnings: list[str]) -> None:
    meta_tool = tools.get("kaggle", {}).get("dataset_metadata")
    if not meta_tool or not cands:
        return
    calls = []
    for c in cands:
        owner, slug = c.id.split("/", 1)
        calls.append(meta_tool.ainvoke({"owner": owner, "dataset_slug": slug}))
    results = await asyncio.gather(*calls, return_exceptions=True)
    for c, res in zip(cands, results):
        if isinstance(res, BaseException):
            warnings.append(f"kaggle metadata {c.id}: {describe(res)}")
            continue
        info = parse_kaggle_metadata(result_text(res))
        if not info:
            warnings.append(f"kaggle metadata {c.id}: could not parse")
            continue
        parts = [info["subtitle"], info["description"]]
        c.description = "\n\n".join(p for p in parts if p)[:2000]
        c.license = info["license"]
        c.usability = info["usability"]
        c.enriched = True


async def _enrich_hf(tools: dict, cands: list[Candidate], warnings: list[str]) -> None:
    tool = tools.get("huggingface", {}).get("hub_repo_details")
    if not tool or not cands:
        return
    calls = [tool.ainvoke({"repo_ids": [c.id], "repo_type": "dataset", "operations": ["dataset_structure"]})
             for c in cands]
    results = await asyncio.gather(*calls, return_exceptions=True)
    for c, res in zip(cands, results):
        if isinstance(res, BaseException):
            warnings.append(f"hf structure {c.id}: {describe(res)}")
            continue
        info = parse_hf_structure(result_text(res))
        if not info:
            warnings.append(f"hf structure {c.id}: no schema (probably no data files) - skipped")
            continue
        c.rows_exact = info["rows"]
        c.columns = info["columns"]
        c.download_urls = info["parquet_urls"]
        c.enriched = True


async def search_datasets(spec: ProjectSpec, queries: list[str]) -> tuple[list[Candidate], list[str]]:
    """Returns (candidates sorted by pre_score, warnings)."""
    cfg = load_config()["search"]
    limit = cfg.get("per_query_limit", 10)
    top_k = cfg.get("enrich_top_k", 16)
    warnings: list[str] = []

    async with open_tools() as (tools, errors):
        for server, err in errors.items():
            warnings.append(f"{server} unavailable: {err}")
        if not any(tools.get(s) for s in ("huggingface", "kaggle")):
            raise RuntimeError("No search source available. " + " | ".join(warnings))

        merged: dict[str, Candidate] = {}
        for source, q, res in await _search_calls(tools, queries, limit):
            if isinstance(res, BaseException):
                warnings.append(f"{source} search '{q}': {describe(res)}")
                continue
            text = result_text(res)
            found = parse_hf_search(text) if source == "huggingface" else parse_kaggle_search(text)
            if not found and text.strip() and "no " not in text.lower()[:80]:
                warnings.append(f"{source} search '{q}': nothing parsed from {text[:120]!r}")
            for c in found[:limit]:
                if c.key in merged:
                    if q not in merged[c.key].query_hits:
                        merged[c.key].query_hits.append(q)
                else:
                    c.query_hits = [q]
                    merged[c.key] = c

        cands = list(merged.values())
        pre_rank(cands, spec, queries)
        cands.sort(key=lambda c: c.pre_score, reverse=True)

        top = cands[:top_k]
        await asyncio.gather(
            _enrich_kaggle(tools, [c for c in top if c.source == "kaggle"], warnings),
            _enrich_hf(tools, [c for c in top if c.source == "huggingface"], warnings),
        )

    return cands, warnings