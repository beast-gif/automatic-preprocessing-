"""
Test stages 1-3: idea -> spec -> queries -> candidates.

    python -m scripts.try_search "predict whether a telecom customer will churn"

Saves runs\\<timestamp>\\spec.json, queries.json, candidates.json
"""
import asyncio
import json
import sys
import time

from dataprep.config import ROOT, load_config
from dataprep.parser import parse_requirements, scope_problem
from dataprep.queries import expand_queries
from dataprep.search import search_datasets


def _size(c) -> str:
    if c.rows_exact:
        return f"{c.rows_exact:,} rows"
    if c.rows_est:
        return f"~{c.rows_est:,} rows"
    if c.size_bytes is None:
        return "?"
    n = float(c.size_bytes)
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def main():
    sys.stdout.reconfigure(errors="replace")      # emoji in titles on older Windows consoles
    idea = " ".join(sys.argv[1:]).strip() or input("Project idea: ")

    t0 = time.time()
    spec = parse_requirements(idea)
    print(f"Spec: {spec.task_type} / {spec.modality} / target={spec.target_hint}")
    print(f"      keywords: {', '.join(spec.domain_keywords)}")
    problem = scope_problem(spec)
    if problem:
        print(f"\nSTOP: {problem}")
        return

    queries = expand_queries(spec)
    print(f"Queries: {queries}")

    cands, warnings = asyncio.run(search_datasets(spec, queries))
    elapsed = time.time() - t0

    for w in warnings:
        print(f"  ! {w}")

    top_k = load_config()["search"].get("enrich_top_k", 8)
    n_enriched = sum(c.enriched for c in cands)
    print(f"\n{len(cands)} unique datasets in {elapsed:.0f}s "
          f"({n_enriched} of the top {top_k} enriched)\n")
    print(f"{'#':>2}  {'src':3} {'score':>5} {'hits':>4} {'downloads':>9}  {'size':>14}  {'license':18} id")
    for i, c in enumerate(cands[:15], 1):
        src = "HF" if c.source == "huggingface" else "KG"
        lic = (c.license or "-")[:18]
        print(f"{i:>2}  {src:3} {c.pre_score:5.2f} {len(c.query_hits):>4} {c.downloads:>9,}  "
              f"{_size(c):>14}  {lic:18} {c.id}")

    run_dir = ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "queries.json").write_text(json.dumps(queries, indent=2), encoding="utf-8")
    (run_dir / "candidates.json").write_text(
        json.dumps([c.model_dump() for c in cands], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()