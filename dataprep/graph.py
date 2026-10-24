"""
The whole pipeline as one LangGraph graph.

START -> parse --[outside v1 scope]--> END
           |
           v
         expand -> search -> rank --[weak match, retries left]--> expand   (new, different queries)
                               |
                               +--[nothing usable found at all]----> END
                               +--[good match, or out of retries]--> prepare
                                                                       |
                         Send x4 (parallel) ---------------------------+
                         |            |              |            |
                      missing   distribution     encoding      leakage      ("specialist" node, 4 runs)
                         |            |              |            |
                         +------------+--------------+------------+
                                              |
                                  merger -> executor -> report -> END

- Each node returns only the state keys it changes.
- The 4 specialists append to `proposals` (reducer operator.add), so their outputs add up;
  the merger runs once, after all 4 have finished.
- search ADDS to the candidate pool, so a retry re-ranks old and new candidates together.
- rank measures coverage against attempt 1's queries only, so retries don't move the yardstick.
- prepare tries the pick first; if it can't be downloaded or has no usable target, it falls
  back to the next datasets the LLM judged (graph.prepare_fallbacks).
- Every node also writes its output into the run folder, so the try_* scripts work on graph runs.
"""
import json
import operator
import time
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents import AGENTS, run_agent
from .candidate import Candidate
from .config import load_config
from .download import DownloadError
from .executor import execute
from .merger import merge
from .parser import parse_requirements, scope_problem
from .report import write_report
from .prepare import prepare
from .queries import expand_queries
from .rank import RankResult, rank_candidates
from .search import search_datasets
from .spec import ProjectSpec


class State(TypedDict, total=False):
    idea: str
    run_dir: str
    spec: ProjectSpec
    stop_reason: Optional[str]
    queries: list[str]               # every query tried so far
    core_queries: list[str]          # attempt 1's queries - the yardstick for ranking coverage
    new_queries: list[str]           # this attempt's queries
    attempts: int                    # search attempts made
    candidates: list[Candidate]      # pool across all attempts
    ranking: RankResult
    dataset: str                     # id of the dataset actually prepared
    summary: dict
    profile: dict
    proposals: Annotated[list[dict], operator.add]   # the 4 specialists append here
    agent_runs: Annotated[list[dict], operator.add]  # {agent, n, seconds} per specialist
    plan: dict
    processed: dict
    report: str
    warnings: Annotated[list[str], operator.add]    # every node can append


def _save(run_dir: str, name: str, data) -> None:
    path = Path(run_dir) / name
    if hasattr(data, "model_dump_json"):
        text = data.model_dump_json(indent=2)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    path.write_text(text, encoding="utf-8")


# ---------------- nodes ----------------

def parse_node(state: State) -> dict:
    spec = parse_requirements(state["idea"])
    _save(state["run_dir"], "spec.json", spec)
    return {"spec": spec, "stop_reason": scope_problem(spec),
            "queries": [], "attempts": 0, "candidates": []}


def expand_node(state: State) -> dict:
    tried = state.get("queries", [])
    new = expand_queries(state["spec"], avoid=tried)
    out = {"new_queries": new, "queries": tried + new, "attempts": state.get("attempts", 0) + 1}
    if not state.get("core_queries"):
        out["core_queries"] = new
    if not new:
        out["warnings"] = ["query expansion produced no new queries"]
    _save(state["run_dir"], "queries.json", out["queries"])
    return out


async def search_node(state: State) -> dict:
    if not state.get("new_queries"):
        return {}
    found, warnings = await search_datasets(state["spec"], state["new_queries"])

    pool = {c.key: c for c in state.get("candidates", [])}
    for c in found:
        old = pool.get(c.key)
        if old is None:
            pool[c.key] = c
            continue
        hits = list(dict.fromkeys(old.query_hits + c.query_hits))
        keep = old if old.enriched or not c.enriched else c       # keep the copy with details
        pool[c.key] = keep.model_copy(update={"query_hits": hits})

    cands = list(pool.values())
    _save(state["run_dir"], "candidates.json", [c.model_dump() for c in cands])
    return {"candidates": cands, "warnings": warnings}


def rank_node(state: State) -> dict:
    # Coverage is measured against the spec + attempt 1's queries only. Retry queries are
    # deliberately broad; counting them would grow the term list and lower every dataset's
    # coverage on each retry (blastchar fell 0.85 -> 0.83 -> 0.76 that way).
    core = state.get("core_queries") or state["queries"]
    ranking = rank_candidates(state["candidates"], state["spec"], core)
    _save(state["run_dir"], "ranking.json", ranking)
    return {"ranking": ranking}


async def prepare_node(state: State) -> dict:
    cfg = load_config()["graph"]
    ranking = state["ranking"]
    warnings = []
    if ranking.weak:
        warnings.append(f"no strong match after {state['attempts']} search attempt(s); "
                        f"using the best one found (final {ranking.pick.final:.2f})")

    judged = [s for s in ranking.ranked if s.fit is not None]       # pick is judged[0]
    tries = judged[: 1 + cfg.get("prepare_fallbacks", 2)]
    for s in tries:
        try:
            summary, profile = await prepare(s.candidate, state["spec"], Path(state["run_dir"]))
            return {"dataset": s.candidate.id, "summary": summary, "profile": profile,
                    "warnings": warnings}
        except (DownloadError, ValueError) as e:
            warnings.append(f"could not use {s.candidate.id}: {e}")

    raise RuntimeError("None of the top datasets could be prepared:\n  " + "\n  ".join(warnings))


def specialist_node(payload: dict) -> dict:
    """Runs once per Send. payload = {"agent", "profile", "spec"} (not the full state)."""
    agent, t0 = payload["agent"], time.time()
    try:
        props = run_agent(agent, payload["profile"], payload["spec"])
        warnings = []
    except Exception as e:                       # one failed specialist must not stop the run;
        props = []                               # the merger's safety net covers its columns
        warnings = [f"{agent} specialist failed ({type(e).__name__}: {e}); safety net used instead"]
    run = {"agent": agent, "n": len(props), "seconds": round(time.time() - t0, 1)}
    return {"proposals": props, "agent_runs": [run], "warnings": warnings}


def merger_node(state: State) -> dict:
    _save(state["run_dir"], "proposals.json", state.get("proposals", []))
    plan = merge(state.get("proposals", []), state["profile"], load_config()["preprocess"])
    _save(state["run_dir"], "plan.json", plan)
    return {"plan": plan}


def executor_node(state: State) -> dict:
    summary = execute(state["plan"], Path(state["run_dir"]), state["spec"].task_type)
    warnings = []
    if any(summary["nan_after"].values()):
        warnings.append(f"missing values remain after preprocessing: {summary['nan_after']}")
    if any(summary.get("unseen_target_rows", {}).values()):
        warnings.append(f"target classes not seen in train (encoded -1): {summary['unseen_target_rows']}")
    return {"processed": summary, "warnings": warnings}


def report_node(state: State) -> dict:
    path = write_report(Path(state["run_dir"]), state.get("warnings", []))
    return {"report": str(path)}


# ---------------- edges ----------------

def after_parse(state: State) -> str:
    return END if state.get("stop_reason") else "expand"


def after_rank(state: State) -> str:
    ranking = state["ranking"]
    if ranking.pick and not ranking.weak:
        return "prepare"
    max_retries = load_config()["graph"].get("max_retries", 2)
    if state["attempts"] <= max_retries:              # attempts 1..max_retries may retry
        return "expand"
    return "prepare" if ranking.pick else END


def dispatch(state: State) -> list[Send]:
    """Fan out: one Send per specialist, all four run in the same step (in parallel)."""
    payload = {"profile": state["profile"], "spec": state["spec"]}
    return [Send("specialist", {"agent": a, **payload}) for a in AGENTS]


def build_graph():
    g = StateGraph(State)
    g.add_node("parse", parse_node)
    g.add_node("expand", expand_node)
    g.add_node("search", search_node)
    g.add_node("rank", rank_node)
    g.add_node("prepare", prepare_node)
    g.add_node("specialist", specialist_node)
    g.add_node("merger", merger_node)
    g.add_node("executor", executor_node)
    g.add_node("report", report_node)

    g.add_edge(START, "parse")
    g.add_conditional_edges("parse", after_parse, {"expand": "expand", END: END})
    g.add_edge("expand", "search")
    g.add_edge("search", "rank")
    g.add_conditional_edges("rank", after_rank, {"prepare": "prepare", "expand": "expand", END: END})
    g.add_conditional_edges("prepare", dispatch, ["specialist"])     # fan-out
    g.add_edge("specialist", "merger")                                 # fan-in: merger waits for all 4
    g.add_edge("merger", "executor")
    g.add_edge("executor", "report")
    g.add_edge("report", END)
    return g.compile()