"""
Test stage 5 on a ranked run: download the pick, clean, split, profile.

    python -m scripts.try_prepare                                     (newest run with ranking.json)
    python -m scripts.try_prepare --run runs\\20261023-184308
    python -m scripts.try_prepare --candidate MLLab-TS/telco_customer_churn   (another ranked dataset)
"""
import argparse
import asyncio
import sys
from pathlib import Path

from dataprep.config import ROOT
from dataprep.prepare import prepare
from dataprep.rank import RankResult
from dataprep.spec import ProjectSpec


def newest_ranked_run() -> Path:
    runs = sorted(p for p in (ROOT / "runs").glob("*") if (p / "ranking.json").exists())
    if not runs:
        raise SystemExit("No ranked runs found. Run scripts.try_search and scripts.try_rank first.")
    return runs[-1]


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.2f}"


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--candidate", help="owner/name of another dataset in ranking.json")
    args = ap.parse_args()

    run = Path(args.run) if args.run else newest_ranked_run()
    if not run.is_absolute():
        run = ROOT / run
    spec = ProjectSpec.model_validate_json((run / "spec.json").read_text(encoding="utf-8"))
    ranking = RankResult.model_validate_json((run / "ranking.json").read_text(encoding="utf-8"))

    if args.candidate:
        match = [s for s in ranking.ranked if s.candidate.id == args.candidate]
        if not match:
            raise SystemExit(f"{args.candidate} is not in {run.name}\\ranking.json")
        cand = match[0].candidate
    else:
        cand = ranking.pick.candidate
    print(f"Run: {run.name}  |  dataset: {cand.id} ({cand.source})\n")

    summary, profile = asyncio.run(prepare(cand, spec, run))

    for line in summary["log"]:
        print(f"  - {line}")

    r = summary["rows"]
    print(f"\nRows: loaded {r['loaded']:,} -> after cleaning {r['after_cleaning']:,} -> "
          f"train {r['train']:,} / val {r['val']:,} / test {r['test']:,}")
    if "class_shares" in summary:
        for name, shares in summary["class_shares"].items():
            print(f"  {name:5}: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in shares.items()))

    print(f"\nProfile of train ({profile['n_rows_profiled']:,} rows, target = {profile['target']['name']}):\n")
    print(f"{'column':24} {'kind':11} {'null':>6} {'uniq':>6} {'uniq%':>6} {'skew':>6} {'outl':>6} {'assoc':>6}")
    for c in profile["columns"]:
        print(f"{c['name'][:24]:24} {c['kind']:11} {_fmt(c['null_pct'], True):>6} {c['n_unique']:>6} "
              f"{_fmt(c['unique_ratio'], True):>6} {_fmt(c.get('skew')):>6} "
              f"{_fmt(c.get('outlier_pct'), True):>6} {_fmt(c['target_assoc']):>6}")

    print(f"\nSaved: {run / 'data'}\\(train|val|test).csv, profile.json, prepare_log.json")


if __name__ == "__main__":
    main()
