"""
Stage 6b - merger: four lists of proposals -> one valid, ordered plan. Plain code, no LLM,
so the same proposals always give the same plan.

Rule 0  sanity      unknown op, op outside that specialist's whitelist, unknown column, target -> rejected
Rule 1  drop wins   a dropped column loses every other proposal
Rule 2  ownership   one op per column per stage; the stage's owner wins a conflict
                    (add_missing_flag is additive: it creates a new column, so it never conflicts)
Rule 3  validity    checked against the profile; fixable problems are REWRITTEN, others rejected
                      log1p with min <= -1            -> yeo_johnson
                      impute_median/mean on text      -> impute_mode
                      one_hot above one_hot_hard_limit categories -> frequency_encode
                      ordinal without an order        -> one_hot / frequency_encode
                      numeric-only op on a text column, encode op on a numeric column -> rejected
                      clip on fewer than clip_min_rows rows, flag on a column with no missing values -> rejected
Rule 4  order       impute -> clip -> transform -> encode -> scale
Rule 5  grouping    done in the executor (columns with identical steps share one sklearn Pipeline)
Rule 6  safety net  every kept column must end up numeric with no gaps:
                      numeric with missing values and no impute -> impute_median
                      text with no encoding -> binary / one_hot / frequency_encode / parse_datetime,
                      free text -> dropped
Every decision is written to plan["log"].
"""
from .agents import ALLOWED, is_numeric

STAGE_ORDER = ["impute", "clip", "transform", "encode", "scale"]
STAGE_OF = {
    "drop_column": "drop",
    "impute_median": "impute", "impute_mean": "impute", "impute_mode": "impute", "impute_constant": "impute",
    "add_missing_flag": "flag",
    "clip_p1_p99": "clip",
    "log1p": "transform", "yeo_johnson": "transform",
    "one_hot": "encode", "binary": "encode", "ordinal": "encode",
    "frequency_encode": "encode", "parse_datetime": "encode",
    "standard_scale": "scale", "robust_scale": "scale", "minmax_scale": "scale",
}
OWNER = {"impute": "missing", "clip": "distribution", "transform": "distribution",
         "encode": "encoding", "scale": "distribution"}
NUMERIC_ONLY = {"impute_median", "impute_mean", "clip_p1_p99", "log1p", "yeo_johnson",
                "standard_scale", "robust_scale", "minmax_scale"}
MISSING = "__missing__"


def _step(p: dict) -> dict:
    return {"op": p["op"], "params": p.get("params") or {}, "agent": p["agent"], "reason": p.get("reason", "")}


def _rewrite(p: dict, op: str, params: dict | None = None) -> dict:
    return {**p, "op": op, "params": params if params is not None else {}}


def merge(proposals: list[dict], profile: dict, cfg: dict) -> dict:
    cols = {c["name"]: c for c in profile["columns"]}
    target = profile["target"]["name"]
    n_rows = profile["n_rows"]
    max_oh = cfg.get("one_hot_max_categories", 15)
    hard_oh = cfg.get("one_hot_hard_limit", 50)
    clip_min = cfg.get("clip_min_rows", 100)
    log: list[dict] = []

    def note(action, p, rule, detail=""):
        log.append({"action": action, "column": p.get("column"), "op": p.get("op"),
                    "agent": p.get("agent"), "rule": rule, "detail": detail})

    # Rule 0 - sanity
    valid = []
    for p in proposals:
        if p.get("op") not in STAGE_OF:
            note("rejected", p, "unknown operation")
        elif p["op"] not in ALLOWED.get(p.get("agent"), set()):
            note("rejected", p, "not this specialist's operation")
        elif p.get("column") == target:
            note("rejected", p, "target column")
        elif p.get("column") not in cols:
            note("rejected", p, "no such column")
        else:
            valid.append(p)

    # Rule 1 - drop wins
    dropped: dict[str, str] = {}
    for p in valid:
        if p["op"] == "drop_column" and p["column"] not in dropped:
            dropped[p["column"]] = f"{p['agent']}: {p.get('reason', '')}"
    live = []
    for p in valid:
        if p["op"] == "drop_column":
            continue
        if p["column"] in dropped:
            note("discarded", p, "column is dropped")
        else:
            live.append(p)

    # Rule 2 - stage ownership
    plan: dict[str, dict[str, dict]] = {}
    flags: dict[str, dict] = {}
    for p in live:
        stage = STAGE_OF[p["op"]]
        if stage == "flag":
            flags.setdefault(p["column"], p)
            continue
        slot = plan.setdefault(p["column"], {})
        cur = slot.get(stage)
        if cur is None:
            slot[stage] = p
        elif p["agent"] == OWNER[stage] and cur["agent"] != OWNER[stage]:
            note("rejected", cur, "stage ownership", f"{stage} belongs to the {OWNER[stage]} specialist")
            slot[stage] = p
        else:
            why = "one operation per stage" if p["agent"] == cur["agent"] else \
                f"{stage} belongs to the {OWNER[stage]} specialist"
            note("rejected", p, "stage ownership", why)

    # Rule 3 - validity against the profile
    for col, slot in plan.items():
        cp = cols[col]
        num = is_numeric(cp)
        n_unique = cp.get("n_unique") or 0
        for stage in STAGE_ORDER:                       # in order, so log1p can see whether clip survived
            p = slot.get(stage)
            if p is None:
                continue
            op = p["op"]
            new = p
            if op in NUMERIC_ONLY and not num:
                if op in ("impute_median", "impute_mean"):
                    new = _rewrite(p, "impute_mode")
                    note("rewritten", p, "validity", "column is not numeric -> impute_mode")
                else:
                    new = None
                    note("rejected", p, "validity", "numeric-only operation on a non-numeric column")
            elif stage == "encode" and num:
                new = None
                note("rejected", p, "validity", "column is already numeric")
            elif op == "parse_datetime" and cp["kind"] != "datetime":
                new = _rewrite(p, "one_hot" if n_unique <= max_oh else "frequency_encode")
                note("rewritten", p, "validity", f"not a date column -> {new['op']}")
            elif op == "log1p":
                low = cp.get("p1") if "clip" in slot else cp.get("min")
                if low is None or low <= -1:
                    new = _rewrite(p, "yeo_johnson")
                    note("rewritten", p, "validity", f"smallest value {low} <= -1, log1p undefined -> yeo_johnson")
            elif op == "clip_p1_p99" and n_rows < clip_min:
                new = None
                note("rejected", p, "validity", f"only {n_rows} rows; percentiles would be noise")
            elif op == "one_hot" and n_unique > hard_oh:
                new = _rewrite(p, "frequency_encode")
                note("rewritten", p, "validity", f"{n_unique} categories -> frequency_encode")
            elif op == "ordinal" and not isinstance((p.get("params") or {}).get("order"), list):
                new = _rewrite(p, "one_hot" if n_unique <= max_oh else "frequency_encode")
                note("rewritten", p, "validity", f"no category order given -> {new['op']}")
            elif op == "impute_constant":
                value = (p.get("params") or {}).get("value")
                if num and not isinstance(value, (int, float)):
                    new = _rewrite(p, "impute_constant", {"value": 0})
                elif not num and value is None:
                    new = _rewrite(p, "impute_constant", {"value": MISSING})
            if new is None:
                del slot[stage]
            else:
                slot[stage] = new

    for col, p in list(flags.items()):
        if (cols[col].get("null_pct") or 0) == 0:
            note("rejected", p, "validity", "no missing values in train - the flag would be constant")
            del flags[col]

    # Rule 6 - safety net
    def auto(col, op, why, params=None):
        p = {"column": col, "op": op, "params": params or {}, "agent": "merger", "reason": why}
        note("added", p, "safety net", why)
        return p

    for col, cp in cols.items():
        if col == target or col in dropped:
            continue
        slot = plan.setdefault(col, {})
        if is_numeric(cp):
            if (cp.get("null_pct") or 0) > 0 and "impute" not in slot:
                slot["impute"] = auto(col, "impute_median", "numeric column with missing values, no imputation proposed")
        elif "encode" not in slot:
            n_unique = cp.get("n_unique") or 0
            if cp["kind"] == "text":
                dropped[col] = "merger: free text is not supported in this version"
                note("added", {"column": col, "op": "drop_column", "agent": "merger"}, "safety net",
                     "free-text column with no encoding")
                plan.pop(col)
                continue
            op = ("parse_datetime" if cp["kind"] == "datetime" else
                  "binary" if n_unique <= 2 else
                  "one_hot" if n_unique <= max_oh else "frequency_encode")
            slot["encode"] = auto(col, op, f"text column ({n_unique} values) with no encoding proposed")

    # Rule 4 - canonical order
    columns = {col: [_step(slot[s]) for s in STAGE_ORDER if s in slot]
               for col, slot in plan.items() if col not in dropped}

    return {
        "target": target,
        "dropped": dropped,
        "columns": columns,
        "flags": {col: _step(p) for col, p in flags.items() if col not in dropped},
        "log": log,
    }
