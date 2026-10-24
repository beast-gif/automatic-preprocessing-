"""
Stage 5 - download -> clean -> split -> profile, for the dataset picked in stage 4.

Writes into the run folder:
  raw\                  downloaded files (reused if you run again)
  data\train.csv, val.csv, test.csv
  profile.json          profile of the TRAIN split (input for the preprocessing agents)
  prepare_log.json      every decision made here
"""
import json
from pathlib import Path

from .candidate import Candidate
from .clean import clean_frame, combine, drop_missing_target, find_target
from .config import load_config
from .download import fetch_dataset
from .profile import profile_frame
from .spec import ProjectSpec
from .split import split_frame


async def prepare(cand: Candidate, spec: ProjectSpec, run_dir: Path) -> tuple[dict, dict]:
    """Returns (summary, profile)."""
    cfg = load_config()["prepare"]
    log: list[str] = []

    paths = await fetch_dataset(cand, run_dir / "raw", cfg)
    for p in paths:
        log.append(f"downloaded {p.name} ({p.stat().st_size:,} bytes)")

    df = combine(paths, log)
    n_loaded = len(df)
    log.append(f"loaded {n_loaded:,} rows x {df.shape[1]} columns")

    df = clean_frame(df, cfg, log)
    target = find_target(df, spec, cfg.get("target_model", "fast"), log)
    df = drop_missing_target(df, target, log)

    train, val, test = split_frame(df, target, spec, cfg, log)
    profile = profile_frame(train, target, spec.task_type,
                            cfg.get("profile_max_rows", 200_000), cfg.get("seed", 42))

    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parts = {"train": train, "val": val, "test": test}
    for name, part in parts.items():
        part.to_csv(data_dir / f"{name}.csv", index=False)

    summary = {
        "dataset": cand.id,
        "source": cand.source,
        "target": target,
        "rows": {"loaded": n_loaded, "after_cleaning": len(df),
                 **{name: len(part) for name, part in parts.items()}},
        "log": log,
    }
    if spec.task_type in ("binary_classification", "multiclass_classification"):
        summary["class_shares"] = {
            name: {str(k): round(float(v), 4) for k, v in part[target].value_counts(normalize=True).items()}
            for name, part in parts.items()
        }

    (run_dir / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "prepare_log.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, profile
