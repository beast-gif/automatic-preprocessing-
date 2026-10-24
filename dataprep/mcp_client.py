"""
MCP connections built from the mcp.servers section of config.yaml.

Two ways in:
- load_tools(): opens a short session per call; used by scripts/try_mcp.py to inspect servers.
- open_tools(): one persistent session per server, read-only tools only; used by the pipeline.
"""
import asyncio
import os
from contextlib import AsyncExitStack, asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .config import load_config

# The only tools the pipeline ever receives. Kaggle's server also exposes
# dataset_delete, competition_submit, kernel_push, model_delete etc. - never loaded.
READ_ONLY_TOOLS = {
    "huggingface": {"hub_repo_search", "hub_repo_details"},
    "kaggle": {"datasets_list", "dataset_files", "dataset_metadata", "dataset_download",
               "dataset_download_file"},
}


def server_connections(cfg: dict | None = None) -> dict:
    servers = (cfg or load_config())["mcp"]["servers"]
    conns = {}
    for name, s in servers.items():
        if not s.get("enabled", True):
            continue
        transport = s["transport"]
        if transport == "stdio":
            conns[name] = {"transport": "stdio", "command": s["command"], "args": s.get("args", [])}
        elif transport in ("streamable_http", "http"):
            conn = {"transport": "streamable_http", "url": s["url"]}
            token = os.environ.get(s["token_env"]) if s.get("token_env") else None
            if token and not token.startswith("hf_your"):      # skip the .env.example placeholder
                conn["headers"] = {"Authorization": f"Bearer {token}"}
            conns[name] = conn
        else:
            raise ValueError(f"mcp.servers.{name}: unknown transport '{transport}'")
    return conns


def make_client(cfg: dict | None = None) -> tuple[MultiServerMCPClient, list[str]]:
    conns = server_connections(cfg)
    return MultiServerMCPClient(conns), list(conns)


def describe(e: BaseException) -> str:
    """Unwraps ExceptionGroups (raised by the MCP transport) down to the real errors."""
    if isinstance(e, BaseExceptionGroup):
        return "; ".join(describe(sub) for sub in e.exceptions)
    return f"{type(e).__name__}: {e}"


def _timeout(timeout_s: float | None) -> float:
    if timeout_s is not None:
        return timeout_s
    return load_config()["mcp"].get("load_timeout_s", 120)


async def load_tools(client: MultiServerMCPClient, server_names: list[str],
                     timeout_s: float | None = None) -> tuple[dict, dict]:
    """Returns (tools_by_server, errors_by_server). Every tool the server has, for inspection."""
    timeout_s = _timeout(timeout_s)
    tools, errors = {}, {}
    for name in server_names:
        try:
            tools[name] = await asyncio.wait_for(client.get_tools(server_name=name), timeout_s)
        except asyncio.TimeoutError:
            errors[name] = f"no response after {timeout_s:.0f}s"
        except Exception as e:  # includes ExceptionGroup from the transport layer
            errors[name] = describe(e)
    return tools, errors


@asynccontextmanager
async def open_tools(timeout_s: float | None = None, servers: list[str] | None = None):
    """
    Opens one session per enabled server and keeps it open for the whole block.

        async with open_tools() as (tools, errors):
            search = tools["kaggle"]["datasets_list"]

    servers: only open these (default: every enabled server).
    tools[server][tool_name] -> LangChain tool (read-only whitelist only).
    errors[server] -> why that server couldn't be opened.
    """
    timeout_s = _timeout(timeout_s)
    client, names = make_client()
    if servers is not None:
        names = [n for n in names if n in servers]
    tools, errors = {}, {}
    async with AsyncExitStack() as stack:
        for name in names:
            try:
                session = await asyncio.wait_for(stack.enter_async_context(client.session(name)), timeout_s)
                loaded = await asyncio.wait_for(load_mcp_tools(session), timeout_s)
                allowed = READ_ONLY_TOOLS.get(name, set())
                tools[name] = {t.name: t for t in loaded if t.name in allowed}
            except asyncio.TimeoutError:
                errors[name] = f"no response after {timeout_s:.0f}s"
            except Exception as e:
                errors[name] = describe(e)
        yield tools, errors