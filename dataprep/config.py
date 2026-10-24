"""
Loads config.yaml and .env, and builds the Velona chat models.

Everything else imports from here, so paths and keys are resolved in one place.
"""
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent   # the dataset-agent/ folder
load_dotenv(ROOT / ".env")


@lru_cache
def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_llm(role: str = "fast"):
    """role is a key from llm.models in config.yaml ("fast" or "smart")."""
    from langchain_openai import ChatOpenAI

    cfg = load_config()["llm"]
    key = os.environ.get(cfg["api_key_env"])
    if not key:
        raise RuntimeError(
            f"{cfg['api_key_env']} is not set. Copy .env.example to .env and put your Velona key in it."
        )
    if role not in cfg["models"]:
        raise KeyError(f"llm.models has no '{role}' entry (have: {list(cfg['models'])})")

    return ChatOpenAI(
        model=cfg["models"][role],
        base_url=cfg["base_url"],
        api_key=key,
        temperature=cfg.get("temperature", 0),
        max_retries=cfg.get("max_retries", 3),
        timeout=cfg.get("timeout_s", 90),
    )
