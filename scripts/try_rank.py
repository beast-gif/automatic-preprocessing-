"""
Test stage 4 on a saved search run (no new searching, so it's cheap to repeat).

    python -m scripts.try_rank                                  (newest folder in runs\\)
    python -m scripts.try_rank --run runs\\20261023-173016

Saves ranking.json in the same run folder.
"""
import argparse
import json
import sys
from pathlib import Path

from dataprep.candidate import Candidate
from dataprep.config import ROOT, load_config
from dataprep.rank import rank_candidates
from dataprep.spec import ProjectSpec


def newest_run() -> Path:
    runs = sorted(p for p in (ROOT / "runs").glob("*") if (p / "candidates.json").exists())
    if not runs:
        raise SystemExit("No runs found. Run scripts.try_search first.")
    return runs[-1]


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run folder (default: newest in runs\\)")
    args = ap.parse_args()

    run = Path(args.run) if args.run else newest_run()
    if not run.is_absolute():
        run = ROOT / run
    spec = ProjectSpec.model_validate_json((run / "spec.json").read_text(encoding="utf-8"))
    cands = [Candidate.model_validate(d)
             for d in json.loads((run / "candidates.json").read_text(encoding="utf-8"))]
    print(f"Run: {run.name}  |  {len(cands)} candidates, "
          f"{sum(c.enriched for c in cands)} with metadata  |  project: {spec.summary}\n")

    queries_file = run / "queries.json"
    queries = json.loads(queries_file.read_text(encoding="utf-8")) if queries_file.exists() else []
    result = rank_candidates(cands, spec, queries)
    if not result.ranked:
        raise SystemExit("Nothing to rank (no candidate has metadata).")

    n = load_config()["rank"].get("rerank_top_n", 8)
    print(f"{'#':>2} {'src':3} {'cov':>4} {'mod':>4} {'size':>4} {'lic':>4} {'pop':>4} "
          f"{'score':>5} {'fit':>4} {'tgt':>7} {'final':>5}  id")
    for i, s in enumerate(result.ranked[:14], 1):
        p, c = s.parts, s.candidate
        src = "HF" if c.source == "huggingface" else "KG"
        fit = f"{s.fit:.0f}" if s.fit is not None else "-"
        flag = " (bad title x0.5)" if s.penalized else ""
        print(f"{i:>2} {src:3} {p['coverage']:4.2f} {p['modality']:4.2f} {p['size']:4.2f} "
              f"{p['license']:4.2f} {p['popularity']:4.2f} {s.score:5.2f} {fit:>4} "
              f"{s.has_target or '-':>7} {s.final:5.2f}  {c.id}{flag}")
        if i == n:
            print("   " + "-" * 70 + "  (above: judged by the LLM)")

    print("\nLLM reasons:")
    for s in result.ranked[:n]:
        print(f"  {s.candidate.id}: {s.reason}")

    p = result.pick
    print(f"\nPICK: {p.candidate.id}  (final {p.final:.2f})  {p.candidate.url}")
    if result.weak:
        print("WEAK: best final score is below rank.weak_threshold - the graph will retry with new queries.")

    (run / "ranking.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nSaved {run / 'ranking.json'}")


if __name__ == "__main__":
    main()