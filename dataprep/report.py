"""
Stage 6d - report.md: what was picked, why, and every preprocessing decision.
Built from the run folder's JSON files only (no LLM), so it always matches what was done.
"""
import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _md(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def write_report(run_dir: Path, warnings: list[str] | None = None) -> Path:
    spec = _load(run_dir / "spec.json") or {}
    ranking = _load(run_dir / "ranking.json") or {}
    prep = _load(run_dir / "prepare_log.json") or {}
    plan = _load(run_dir / "plan.json") or {}
    summary = _load(run_dir / "processed" / "summary.json") or {}

    L = ["# Dataset preparation report", ""]
    L += [f"**Project:** {spec.get('summary', '-')}  ",
          f"**Task:** {spec.get('task_type', '-')}  |  **Target:** `{prep.get('target', '-')}`", ""]

    # dataset
    chosen = next((s for s in ranking.get("ranked", []) if s["candidate"]["id"] == prep.get("dataset")), None)
    L += ["## Dataset", ""]
    if chosen:
        c = chosen["candidate"]
        fit = f"{chosen['fit']:.0f}/10" if chosen.get("fit") is not None else "-"
        L += [f"- **{c['id']}** ({c['source']}) - {c['url']}",
              f"- Ranking: final {chosen['final']:.2f} (score {chosen['score']:.2f}, LLM fit {fit})",
              f"- Why: {chosen.get('reason') or '-'}",
              f"- License: {c.get('license') or 'none listed'}"]
    rows = prep.get("rows", {})
    if rows:
        L.append(f"- Rows: {rows.get('loaded', 0):,} loaded -> {rows.get('after_cleaning', 0):,} after cleaning -> "
                 f"train {rows.get('train', 0):,} / val {rows.get('val', 0):,} / test {rows.get('test', 0):,}")
    if prep.get("class_shares"):
        shares = prep["class_shares"]["train"]
        L.append("- Class shares (train): " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in shares.items()))
    if prep.get("log"):
        L += ["", "Cleaning steps:", ""] + [f"- {_md(x)}" for x in prep["log"]]

    # dropped
    L += ["", "## Dropped columns", ""]
    if plan.get("dropped"):
        L += ["| column | why |", "|---|---|"]
        L += [f"| `{c}` | {_md(why)} |" for c, why in plan["dropped"].items()]
    else:
        L.append("None.")

    # per column
    L += ["", "## Preprocessing per column", "", "| column | steps | proposed by | why |", "|---|---|---|---|"]
    for col, steps in plan.get("columns", {}).items():
        if not steps:
            L.append(f"| `{col}` | kept as is | - | already numeric, no gaps |")
            continue
        ops = " -> ".join(s["op"] + (f" {s['params']}" if s["params"] else "") for s in steps)
        who = ", ".join(dict.fromkeys(s["agent"] for s in steps))
        why = "; ".join(_md(s["reason"]) for s in steps if s.get("reason"))
        L.append(f"| `{col}` | {_md(ops)} | {who} | {why} |")
    if plan.get("flags"):
        L += ["", "Missing-value flags (new 0/1 columns): " + ", ".join(f"`{c}_missing`" for c in plan["flags"])]

    # merger decisions
    log = plan.get("log", [])
    L += ["", "## Merger decisions", ""]
    if log:
        L += ["| action | column | op | from | rule | detail |", "|---|---|---|---|---|---|"]
        L += [f"| {e['action']} | `{e['column']}` | {e['op']} | {e['agent']} | {e['rule']} | {_md(e.get('detail', ''))} |"
              for e in log]
    else:
        L.append("No conflicts - every proposal was accepted as is.")

    # output
    L += ["", "## Output", ""]
    if summary:
        sh = summary["shapes"]
        L += [f"- {summary['n_input_columns']} input columns -> **{summary['n_features']} features** "
              f"({summary['n_groups']} pipeline groups)",
              f"- train {sh['train'][0]:,} x {sh['train'][1]}, val {sh['val'][0]:,}, test {sh['test'][0]:,}"]
        if summary.get("target_mapping"):
            L.append("- Target encoding: " + ", ".join(f"{k} -> {v}" for k, v in summary["target_mapping"].items()))
        if any(summary.get("nan_after", {}).values()):
            L.append(f"- WARNING: missing values remain after preprocessing: {summary['nan_after']}")
    L += ["- Files: `processed/train.csv`, `processed/val.csv`, `processed/test.csv` (target is the last column), "
          "`pipeline.pkl` (fitted transformer - apply it to new data with `.transform`)"]

    if warnings:
        L += ["", "## Warnings", ""] + [f"- {_md(w)}" for w in warnings]

    path = run_dir / "report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
