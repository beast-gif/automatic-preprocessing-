"""
Test step 6 on a prepared run: 4 specialists (parallel) -> merger -> executor -> report.

    python -m scripts.try_preprocess                         (newest run with profile.json)
    python -m scripts.try_preprocess --run runs\\20261024-094129
    python -m scripts.try_preprocess --reuse                 (skip the LLM: re-merge saved proposals.json)
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dataprep.agents import AGENTS, run_agent
from dataprep.config import ROOT, load_config
from dataprep.executor import execute
from dataprep.merger import merge
from dataprep.report import write_report
from dataprep.spec import ProjectSpec


def newest_prepared_run() -> Path:
    runs = sorted(p for p in (ROOT / "runs").glob("*") if (p / "profile.json").exists())
    if not runs:
        raise SystemExit("No prepared runs found. Run scripts.try_prepare or dataprep.run first.")
    return runs[-1]


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--reuse", action="store_true", help="use the saved proposals.json instead of calling the LLM")
    args = ap.parse_args()

    run = Path(args.run) if args.run else newest_prepared_run()
    if not run.is_absolute():
        run = ROOT / run
    spec = ProjectSpec.model_validate_json((run / "spec.json").read_text(encoding="utf-8"))
    profile = json.loads((run / "profile.json").read_text(encoding="utf-8"))
    print(f"Run: {run.name}  |  {profile['n_columns']} columns, target {profile['target']['name']}\n")

    if args.reuse:
        proposals = json.loads((run / "proposals.json").read_text(encoding="utf-8"))
        print(f"Reusing {len(proposals)} saved proposals\n")
    else:
        t0 = time.time()
        with ThreadPoolExecutor(len(AGENTS)) as pool:
            results = list(pool.map(lambda a: run_agent(a, profile, spec), AGENTS))
        proposals = [p for r in results for p in r]
        counts = ", ".join(f"{a} {len(r)}" for a, r in zip(AGENTS, results))
        print(f"Specialists ({time.time() - t0:.1f}s, in parallel): {counts}\n")
        (run / "proposals.json").write_text(json.dumps(proposals, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'agent':12} {'column':22} {'op':17} reason")
    for p in proposals:
        print(f"{p['agent']:12} {p['column'][:22]:22} {p['op']:17} {p.get('reason', '')[:70]}")

    plan = merge(proposals, profile, load_config()["preprocess"])
    (run / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDropped: {plan['dropped'] or 'none'}")
    print("\nPlan:")
    for col, steps in plan["columns"].items():
        ops = " -> ".join(s["op"] for s in steps) or "(kept as is)"
        auto = "  [safety net]" if any(s["agent"] == "merger" for s in steps) else ""
        print(f"  {col[:24]:24} {ops}{auto}")
    if plan["flags"]:
        print(f"  + missing flags: {', '.join(plan['flags'])}")
    if plan["log"]:
        print("\nMerger log:")
        for e in plan["log"]:
            print(f"  {e['action']:9} {e['column']}.{e['op']} ({e['agent']}) - {e['rule']}"
                  + (f": {e['detail']}" if e.get("detail") else ""))

    summary = execute(plan, run, spec.task_type)
    print(f"\nExecutor: {summary['n_input_columns']} columns -> {summary['n_features']} features "
          f"in {summary['n_groups']} pipeline groups")
    for name, shape in summary["shapes"].items():
        print(f"  {name:5} {shape[0]:>6,} x {shape[1]}   missing values left: {summary['nan_after'][name]}")
    if summary["target_mapping"]:
        print(f"  target: {summary['target_mapping']}")

    path = write_report(run)
    print(f"\nSaved: processed\\, pipeline.pkl, plan.json, proposals.json\nReport: {path}")


if __name__ == "__main__":
    main()
