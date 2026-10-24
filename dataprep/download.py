"""
Stage 5a - get the picked dataset's files onto disk.

Kaggle: dataset_files (MCP) -> choose files -> dataset_download_file (MCP) returns a
        pre-signed storage URL -> plain HTTPS download.
HF:     parquet export URLs from the schema call in stage 3 -> plain HTTPS download.

The file bytes never pass through MCP or the LLM - only the URLs do.
Files already on disk are reused, so re-running is free.
"""
import asyncio
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from .candidate import Candidate
from .mcp_client import describe, open_tools
from .parse_results import parse_download_url, parse_kaggle_files, result_text

TABULAR_EXT = (".csv", ".tsv", ".parquet")
_SKIP_NAMES = re.compile(r"sample[_-]?submission|submission", re.I)


class DownloadError(RuntimeError):
    pass


def choose_files(files: list[tuple[str, int]], max_files: int) -> list[tuple[str, int]]:
    """Tabular files only, no submission templates, largest first, at most max_files."""
    tab = [(n, s) for n, s in files if n.lower().endswith(TABULAR_EXT) and not _SKIP_NAMES.search(n)]
    if not tab:
        names = [n for n, _ in files][:10]
        raise DownloadError(f"No CSV/TSV/parquet file in this dataset. Files: {names}")
    tab.sort(key=lambda f: f[1], reverse=True)
    return tab[:max_files]


async def _kaggle_files(dataset_id: str, max_files: int) -> list[tuple[str, str, int | None]]:
    owner, slug = dataset_id.split("/", 1)
    async with open_tools(servers=["kaggle"]) as (tools, errors):
        if "kaggle" in errors:
            raise DownloadError(f"Kaggle MCP unavailable: {errors['kaggle']}")
        kg = tools.get("kaggle", {})
        if "dataset_files" not in kg or "dataset_download_file" not in kg:
            raise DownloadError("Kaggle MCP server is missing dataset_files / dataset_download_file")

        listing = result_text(await kg["dataset_files"].ainvoke({"owner": owner, "dataset_slug": slug}))
        files = parse_kaggle_files(listing)
        if not files:
            raise DownloadError(f"Could not read the file list: {listing[:200]!r}")
        chosen = choose_files(files, max_files)

        results = await asyncio.gather(
            *(kg["dataset_download_file"].ainvoke({"owner": owner, "dataset_slug": slug, "file_name": n})
              for n, _ in chosen),
            return_exceptions=True,
        )
    out = []
    for (name, size), res in zip(chosen, results):
        if isinstance(res, BaseException):
            raise DownloadError(f"No download URL for {name}: {describe(res)}")
        url = parse_download_url(result_text(res))
        if not url:
            raise DownloadError(f"No download URL for {name}: {result_text(res)[:200]!r}")
        out.append((name, url, size))
    return out


def _hf_files(c: Candidate, max_files: int) -> list[tuple[str, str, int | None]]:
    if not c.download_urls:
        raise DownloadError(f"{c.id} has no parquet export URLs (stage 3 found no schema)")
    out = []
    for url in c.download_urls[:max_files]:
        parts = unquote(urlparse(url).path).split("/")
        out.append(("__".join(parts[-3:]), url, None))     # default__train__0000.parquet
    return out


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name.replace("/", "__"))


def _headers_for(url: str) -> dict:
    token = os.environ.get("HF_TOKEN", "")
    host = urlparse(url).hostname or ""
    if token and not token.startswith("hf_your") and host.endswith("huggingface.co"):
        return {"Authorization": f"Bearer {token}"}      # httpx drops it on cross-host redirects
    return {}


def download_file(url: str, dest: Path, max_bytes: int) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size                       # already downloaded
    tmp = dest.with_name(dest.name + ".part")
    n = 0
    with httpx.stream("GET", url, headers=_headers_for(url), follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        length = r.headers.get("content-length")
        if length and int(length) > max_bytes:
            raise DownloadError(f"{dest.name} is {int(length) / 1e6:.0f} MB, over the {max_bytes / 1e6:.0f} MB cap")
        f = open(tmp, "wb")
        try:
            for chunk in r.iter_bytes(1 << 20):
                n += len(chunk)
                if n > max_bytes:
                    raise DownloadError(f"{dest.name} passed the {max_bytes / 1e6:.0f} MB cap")
                f.write(chunk)
        except BaseException:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
        f.close()
    tmp.replace(dest)
    return n


async def fetch_dataset(c: Candidate, raw_dir: Path, cfg: dict) -> list[Path]:
    """Downloads the candidate's tabular files into raw_dir and returns their paths."""
    max_bytes = int(cfg.get("max_download_mb", 1024) * 1024 * 1024)
    max_files = cfg.get("max_files", 3)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if c.source == "kaggle":
        files = await _kaggle_files(c.id, max_files)
    else:
        files = _hf_files(c, max_files)

    known = sum(s for _, _, s in files if s)
    if known > max_bytes:
        raise DownloadError(f"Files total {known / 1e6:.0f} MB, over the {max_bytes / 1e6:.0f} MB cap")

    paths = [raw_dir / _safe_name(name) for name, _, _ in files]
    await asyncio.gather(*(asyncio.to_thread(download_file, url, p, max_bytes)
                           for (_, url, _), p in zip(files, paths)))
    return paths
