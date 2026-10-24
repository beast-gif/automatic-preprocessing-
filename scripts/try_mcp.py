"""
Test the MCP connections.

List every tool each enabled server exposes:
    python -m scripts.try_mcp

Call one tool (each --arg is key=value; numbers/true/false are parsed automatically):
    python -m scripts.try_mcp --server huggingface --tool <tool_name> --arg query="telecom churn" --arg limit=5
"""
import argparse
import asyncio
import json

from dataprep.mcp_client import make_client, load_tools


def parse_kv(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--arg must be key=value, got: {p}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)   # 5 -> 5, true -> True, ["a"] -> list
        except json.JSONDecodeError:
            if v.startswith("[") and v.endswith("]"):
                # [dataset,model] -> ["dataset", "model"]  (PowerShell strips inner quotes)
                out[k] = [s.strip().strip("'\"") for s in v[1:-1].split(",") if s.strip()]
            else:
                out[k] = v           # plain string
    return out


def arg_names(tool) -> list[str]:
    try:
        return list(tool.args)
    except Exception:
        return []


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server")
    ap.add_argument("--tool")
    ap.add_argument("--arg", action="append", default=[])
    ap.add_argument("--max-chars", type=int, default=3000)
    args = ap.parse_args()

    client, names = make_client()
    if args.server:
        if args.server not in names:
            raise SystemExit(f"'{args.server}' is not an enabled server (have: {names})")
        names = [args.server]

    tools, errors = await load_tools(client, names)

    for name, err in errors.items():
        print(f"[{name}] FAILED to load: {err}\n")

    if not args.tool:
        for name, ts in tools.items():
            print(f"[{name}] {len(ts)} tools")
            for t in ts:
                first_line = (t.description or "").strip().splitlines()[0:1]
                print(f"  - {t.name}({', '.join(arg_names(t))})")
                if first_line:
                    print(f"      {first_line[0][:110]}")
            print()
        return

    match = [t for ts in tools.values() for t in ts if t.name == args.tool]
    if not match:
        raise SystemExit(f"No tool named '{args.tool}'. Run without --tool to list them.")
    result = await match[0].ainvoke(parse_kv(args.arg))
    text = result if isinstance(result, str) else str(result)
    print(text[: args.max_chars])
    if len(text) > args.max_chars:
        print(f"\n... ({len(text) - args.max_chars} more characters cut)")


if __name__ == "__main__":
    asyncio.run(main())