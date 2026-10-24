"""
Turns raw MCP tool output into Candidate objects. No LLM: the formats are regular.

Formats (as returned by the servers, Oct 2026):

Kaggle datasets_list - markdown, one entry per dataset:
    - **Telecom Churn Dataset** (`mnassrib/telecom-churn-datasets`)
      Size: 101892 bytes | Downloads: 41151

Hugging Face hub_repo_search - markdown blocks separated by ---:
    ### hipnologo/telecom_churn_fake
    (optional description)
    **Author:** hipnologo | **Downloads:** 9 | **Likes:** 0 | ...
    **Tags:** size_categories:1K<n<10K, format:parquet, license:mit, ...
    **Link:** [https://hf.co/datasets/...](...)

Kaggle dataset_metadata - a Python dict printed as text ({'info': {...}}).

Hugging Face hub_repo_details with operations=["dataset_structure"] - markdown:
    Total rows: 7.0K.
    Parquet URLs:
    - `default/train/0000.parquet`: https://huggingface.co/datasets/.../0000.parquet
    ### Schema Preview
    | # | Column | Type |
    | 1 | customerID | {"dtype":"string","\\_type":"Value"} |
"""
import ast
import json
import math
import re

from .candidate import Candidate


def result_text(result) -> str:
    """MCP tool results arrive as a list of content blocks [{'type': 'text', 'text': ...}]."""
    if isinstance(result, str):
        return result
    if isinstance(result, tuple):          # (content, artifact) in some adapter versions
        result = result[0]
    if isinstance(result, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in result)
    return str(result)


def _int(s: str | None) -> int | None:
    if not s:
        return None
    try:
        return int(s.replace(",", ""))
    except ValueError:
        return None


# ---------------- Kaggle search ----------------

_KG_HEAD = re.compile(r"\*\*\s*(?P<title>.+?)\s*\*\*\s*\(`(?P<ref>[^`]+)`\)")
_KG_SIZE = re.compile(r"Size:\s*(\d[\d,]*)\s*bytes", re.I)
_KG_DL = re.compile(r"Downloads:\s*(\d[\d,]*)", re.I)


def parse_kaggle_search(text: str) -> list[Candidate]:
    out = []
    # each entry starts with "- **" at the start of a line
    for entry in re.split(r"(?m)^-\s+(?=\*\*)", text):
        head = _KG_HEAD.search(entry)
        if not head or "/" not in head["ref"]:
            continue
        ref = head["ref"].strip()
        size, dl = _KG_SIZE.search(entry), _KG_DL.search(entry)
        out.append(Candidate(
            source="kaggle",
            id=ref,
            title=head["title"].strip(),
            url=f"https://www.kaggle.com/datasets/{ref}",
            size_bytes=_int(size.group(1)) if size else None,
            downloads=_int(dl.group(1)) or 0 if dl else 0,
        ))
    return out


# ---------------- Kaggle metadata ----------------

def parse_kaggle_metadata(text: str) -> dict | None:
    text = text.strip()
    data = None
    for loader in (ast.literal_eval, json.loads):     # literal_eval evaluates literals only, never code
        try:
            data = loader(text)
            break
        except (ValueError, SyntaxError, TypeError):
            continue
    if not isinstance(data, dict):
        return None
    info = data.get("info", data)
    if not isinstance(info, dict):
        return None
    licenses = [l.get("name") for l in (info.get("licenses") or []) if isinstance(l, dict) and l.get("name")]
    return {
        "title": info.get("title") or "",
        "subtitle": info.get("subtitle") or "",
        "description": info.get("description") or "",
        "license": licenses[0] if licenses else None,
        "usability": info.get("usabilityRating"),
        "downloads": info.get("totalDownloads"),
    }


# ---------------- Hugging Face search ----------------

_HF_ID = re.compile(r"(?m)^###\s+(\S+)")
_HF_DL = re.compile(r"\*\*Downloads:\*\*\s*(\d[\d,]*)")
_HF_LIKES = re.compile(r"\*\*Likes:\*\*\s*(\d[\d,]*)")
_HF_TAGS = re.compile(r"\*\*Tags:\*\*\s*(.+)")
_HF_LINK = re.compile(r"\*\*Link:\*\*\s*\[([^\]]+)\]")
_HF_BOILERPLATE = re.compile(r'Dataset Card for "[^"]*"|More Information needed', re.I)

_UNIT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_SIZE_RANGE = re.compile(r"(\d+)([KMBT])<n<(\d+)([KMBT])")
_SIZE_BELOW = re.compile(r"n<(\d+)([KMBT])")
_SIZE_ABOVE = re.compile(r"n>(\d+)([KMBT])")


def hf_rows(size_cat: str) -> tuple[int | None, int | None]:
    """
    size_categories tag -> (rows_low, rows_est).
    rows_est is the geometric midpoint: 1K<n<10K -> sqrt(1,000 x 10,000) = 3,162.
    """
    m = _SIZE_RANGE.search(size_cat)
    if m:
        low = int(m[1]) * _UNIT[m[2]]
        high = int(m[3]) * _UNIT[m[4]]
    elif (m := _SIZE_BELOW.search(size_cat)):
        high = int(m[1]) * _UNIT[m[2]]
        low = high / 10                       # "n<1K": assume at least 100 rows
    elif (m := _SIZE_ABOVE.search(size_cat)):
        low = int(m[1]) * _UNIT[m[2]]
        high = low * 10
    else:
        return None, None
    return int(low), int(math.sqrt(low * high))


def _hf_description(block: str) -> str:
    head = _HF_ID.search(block)
    author = block.find("**Author:**")
    if not head or author == -1 or author < head.end():
        return ""
    raw = _HF_BOILERPLATE.sub(" ", block[head.end():author])
    text = " ".join(raw.split())             # collapse tabs/newlines
    return text if len(text) >= 20 else ""


def parse_hf_search(text: str) -> list[Candidate]:
    out = []
    for block in re.split(r"(?m)^---\s*$", text):
        head = _HF_ID.search(block)
        if not head or "/" not in head.group(1):
            continue
        repo_id = head.group(1).strip()
        tags_m = _HF_TAGS.search(block)
        tags = [t.strip() for t in tags_m.group(1).split(",")] if tags_m else []

        def tag_values(prefix):
            return [t.split(":", 1)[1] for t in tags if t.startswith(prefix + ":")]

        sizes = tag_values("size_categories")
        rows_low, rows_est = hf_rows(sizes[0]) if sizes else (None, None)
        licenses = tag_values("license")
        link = _HF_LINK.search(block)
        dl, likes = _HF_DL.search(block), _HF_LIKES.search(block)

        out.append(Candidate(
            source="huggingface",
            id=repo_id,
            title=repo_id.split("/", 1)[1].replace("_", " ").replace("-", " "),
            url=link.group(1) if link else f"https://huggingface.co/datasets/{repo_id}",
            description=_hf_description(block),
            downloads=_int(dl.group(1)) or 0 if dl else 0,
            likes=_int(likes.group(1)) if likes else None,
            rows_low=rows_low,
            rows_est=rows_est,
            license=licenses[0] if licenses else None,
            formats=tag_values("format"),
            modalities=tag_values("modality"),
        ))
    return out


# ---------------- Kaggle files / download URL ----------------

_KG_FILE = re.compile(r"(?m)^\s*-\s+(.+?)\s+\((\d[\d,]*)\s*bytes\)\s*$")
_KG_URL = re.compile(r"Download URL:\s*(\S+)")
_ANY_URL = re.compile(r"https?://\S+")


def parse_kaggle_files(text: str) -> list[tuple[str, int]]:
    """'- WA_Fn-UseC_-Telco-Customer-Churn.csv (977501 bytes)' -> [(name, bytes)]"""
    return [(name.strip(), _int(size) or 0) for name, size in _KG_FILE.findall(text)]


def parse_download_url(text: str) -> str | None:
    m = _KG_URL.search(text) or _ANY_URL.search(text)
    if not m:
        return None
    url = m.group(1) if m.lastindex else m.group(0)
    return url.rstrip(").,'\"")


# ---------------- Hugging Face dataset_structure ----------------

_HF_TOTAL_ROWS = re.compile(r"Total rows:\s*([\d.,]+)\s*([KMBT])?", re.I)
_HF_SCHEMA_ROW = re.compile(r"(?m)^\|\s*\d+\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$")
_HF_DTYPE = re.compile(r'"dtype"\s*:\s*"([^"]+)"')
_HF_PARQUET = re.compile(r"https://huggingface\.co/\S+?\.parquet")
_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|])")      # markdown escapes like \_ -> _


def _count(num: str, unit: str | None) -> int:
    n = float(num.replace(",", ""))
    return int(n * _UNIT[unit.upper()]) if unit else int(n)


def parse_hf_structure(text: str) -> dict | None:
    """-> {"rows": 7000, "columns": [{"name": "Churn", "dtype": "string"}, ...], "parquet_urls": [...]}"""
    if "Schema" not in text and "Total rows" not in text:
        return None                           # usually: the repo has no data files
    m = _HF_TOTAL_ROWS.search(text)
    rows = _count(m[1], m[2]) if m else None

    columns = []
    schema_part = text.split("Schema Preview", 1)[1] if "Schema Preview" in text else ""
    for name, typ in _HF_SCHEMA_ROW.findall(schema_part):
        d = _HF_DTYPE.search(_MD_ESCAPE.sub(r"\1", typ))
        columns.append({
            "name": _MD_ESCAPE.sub(r"\1", name).strip(),
            "dtype": d.group(1) if d else _MD_ESCAPE.sub(r"\1", typ).strip()[:40],
        })

    urls = list(dict.fromkeys(_HF_PARQUET.findall(text)))    # dedupe, keep order
    return {"rows": rows, "columns": columns, "parquet_urls": urls}