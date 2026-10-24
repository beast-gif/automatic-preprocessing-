"""
Run the whole pipeline as one LangGraph graph.

    python -m dataprep.run "predict whether a telecom customer will churn"

Prints each node as it finishes, then a summary. Everything is saved in runs\\<timestamp>\\.
"""
import argparse
import asyncio
import sys
import time

from .config import ROOT
from .graph import build_graph


def describe(node: str, ch: dict) -> str:
    if node == "parse":
        spec = ch["spec"]
        line = f"{spec.task_type} / {spec.modality} / target={spec.target_hint}"
        return line + (f"\n           STOP: {ch['stop_reason']}" if ch.get("stop_reason") else "")
    if node == "expand":
        return f"attempt {ch['attempts']}: {ch['new_queries']}"
    if node == "search":
        cands = ch.get("candidates", [])
        return f"{len(cands)} datasets in the pool ({sum(c.enriched for c in cands)} with details)"
    if node == "rank":
        r = ch["ranking"]
        if not r.pick:
            return "nothing to rank"
        tag = "WEAK" if r.weak else "ok"
        return f"pick {r.pick.candidate.id} (final {r.pick.final:.2f}, {tag})"
    if node == "prepare":
        rows = ch["summary"]["rows"]
        return (f"{ch['dataset']}: target {ch['summary']['target']}, "
                f"train {rows['train']:,} / val {rows['val']:,} / test {rows['test']:,}")
    if node == "specialist":
        r = ch["agent_runs"][0]
        return f"{r['agent']:12} {r['n']:>3} proposals ({r['seconds']}s)"
    if node == "merger":
        p = ch["plan"]
        count = lambda a: sum(1 for e in p["log"] if e["action"] == a)
        dropped = ", ".join(p["dropped"]) or "none"
        return (f"{len(p['columns'])} columns kept, dropped: {dropped} | "
                f"rejected {count('rejected')}, rewritten {count('rewritten')}, added {count('added')}")
    if node == "executor":
        s = ch["processed"]
        return (f"{s['n_input_columns']} columns -> {s['n_features']} features "
                f"({s['n_groups']} pipeline groups)")
    if node == "report":
        return ch["report"]
    return ""


async def run(idea: str) -> dict:
    run_dir = ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {run_dir}\n")

    graph = build_graph()
    state = {"idea": idea, "run_dir": str(run_dir), "warnings": []}
    final = dict(state)
    t_start = t_prev = time.time()

    async for update in graph.astream(state, stream_mode="updates"):
        for node, changes in update.items():
            changes = changes or {}
            for k, v in changes.items():
                if k == "warnings":
                    final["warnings"] = final["warnings"] + v
                else:
                    final[k] = v
            now = time.time()
            print(f"[{node:10}] {now - t_prev:5.1f}s  {describe(node, changes)}")
            for w in changes.get("warnings", []):
                print(f"           ! {w}")
            t_prev = now

    print(f"\nDone in {time.time() - t_start:.0f}s")
    return final


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("idea", nargs="*")
    args = ap.parse_args()
    idea = " ".join(args.idea).strip() or input("Project idea: ")

    final = asyncio.run(run(idea))

    if final.get("stop_reason"):
        print(f"Stopped: {final['stop_reason']}")
        return
    if not final.get("summary"):
        print("No usable dataset found.")
        return

    run_dir = final["run_dir"]
    print(f"Dataset:  {final['dataset']}")
    if final.get("processed"):
        print(f"Model-ready data: {run_dir}\\processed\\train.csv, val.csv, test.csv  (target = last column)")
        print(f"Fitted pipeline:  {run_dir}\\pipeline.pkl")
        print(f"Report:           {final.get('report', run_dir + chr(92) + 'report.md')}")
    else:
        print(f"Files:    {run_dir}\\data\\train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()