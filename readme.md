# Dataset Agent

Describe a machine-learning project in one sentence, and it finds a suitable public dataset,
downloads it, and turns it into model-ready train / val / test data, explaining every decision
it made along the way.

```
python -m dataprep.run "predict whether a telecom customer will churn"
```

```
[parse     ]   7.7s  binary_classification / tabular / target=churn
[expand    ]   1.7s  attempt 1: ['telecom churn', 'telco customer churn', 'customer attrition telecom', 'subscriber churn']
[search    ]  19.8s  50 datasets in the pool (16 with details)
[rank      ]   7.1s  pick blastchar/telco-customer-churn (final 0.86, ok)
[prepare   ]  19.8s  blastchar/telco-customer-churn: target Churn, train 4,929 / val 1,057 / test 1,057
[specialist]   3.8s  missing        1 proposals (3.8s)
[specialist]   0.1s  leakage        1 proposals (3.8s)
[specialist]   0.2s  distribution   3 proposals (4.0s)
[specialist]   4.3s  encoding      16 proposals (8.3s)
[merger    ]   0.0s  19 columns kept, dropped: customerID | rejected 0, rewritten 0, added 0
[executor  ]   2.1s  20 columns -> 38 features (6 pipeline groups)
[report    ]   0.2s  runs\20261024-110747\report.md

Done in 67s
```

Built with **LangGraph**, **MCP** (Hugging Face + Kaggle servers) and **scikit-learn**.
Version 1 handles **tabular** data (classification, regression, forecasting, clustering, anomaly detection).

---

## How it works

```
START -> parse --[not tabular]--> END
           |
         expand -> search -> rank --[weak match, retries left]--> expand  (new, broader queries)
                               |
                               +--[good match / out of retries]--> prepare
                                                                     |
                       Send x4, in parallel  ------------------------+
                       |             |              |             |
                    missing    distribution     encoding       leakage
                       |             |              |             |
                       +-------------+--------------+-------------+
                                            |
                                merger -> executor -> report -> END
```

| # | Node | What it does | LLM? |
|---|---|---|---|
| 1 | **parse** | Idea -> structured spec: task type, modality, target, domain keywords, feature hints | Haiku |
| 2 | **expand** | Spec -> 4 short title-style search queries (synonyms included, e.g. *telecom* / *telco*) | Haiku |
| 3 | **search** | 4 queries x 2 sources over MCP, in parallel -> merged pool -> cheap pre-rank -> details for the top 16 | - |
| 4 | **rank** | Score every enriched candidate, then one LLM call judges the top 8 | Sonnet |
| 5 | **prepare** | Download -> clean -> dedupe -> find target -> stratified split -> profile the train split | - |
| 6 | **specialists** | Four specialists read the profile and propose operations, in parallel | Haiku x4 |
| 7 | **merger** | Fixed rules turn four proposal lists into one valid, ordered plan | - |
| 8 | **executor** | Plan -> sklearn `ColumnTransformer`, fitted on **train only**, applied to all splits | - |
| 9 | **report** | `report.md`: which dataset and why, every step and every merger decision | - |

### Ranking (stage 4)

```
score = 0.45*coverage + 0.10*modality + 0.15*size + 0.10*license + 0.20*popularity
final = 0.5*score + 0.5*(LLM fit / 10)        (halved if the target column is clearly missing)
```

- **coverage**: share of project terms (spec + first-attempt queries) found in the title, description or column names
- **size**: `min(1, log10(rows) / 5)`
- **license**: 1.0 open, 0.7 non-commercial, 0.4 restricted / unknown
- **popularity**: `log10(1 + downloads) / log10(1 + max(source max, 10,000))`, per source

If the best `final` is under `weak_threshold` (0.6), the graph searches again with different, broader
queries (up to `max_retries` times). The pool grows across attempts, but the ranking yardstick stays fixed.

### The four specialists (stage 6)

Each is a single LLM call that sees only the statistics it needs, never raw rows, and may only use
its own operations:

| Specialist | Sees | Operations |
|---|---|---|
| missing | columns with nulls | `impute_median/mean/mode/constant`, `add_missing_flag`, `drop_column` |
| distribution | numeric columns | `clip_p1_p99`, `log1p`, `yeo_johnson`, `standard/robust/minmax_scale` |
| encoding | text columns | `one_hot`, `binary`, `ordinal`, `frequency_encode`, `parse_datetime`, `drop_column` |
| leakage | all columns | `drop_column` (IDs, post-outcome columns, constants) |

### Merger rules (no LLM, deterministic)

0. **Sanity**: unknown op, op outside the specialist's whitelist, unknown column, or the target -> rejected
1. **Drop wins**: a dropped column loses every other proposal
2. **Stage ownership**: one op per column per stage; the stage's owner wins conflicts
3. **Validity**: checked against the profile; fixable problems are rewritten
   (e.g. `log1p` on a column with min <= -1 -> `yeo_johnson`)
4. **Canonical order**: impute -> clip -> transform -> encode -> scale
5. **Grouping**: columns with identical steps share one pipeline (done in the executor)
6. **Safety net**: every kept column ends numeric with no gaps, even if a specialist missed it or failed

---

## Setup (Windows / PowerShell)

Requires Python 3.11+.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install uv                 # runs the Kaggle MCP server
copy .env.example .env
```

**Keys**

| What | Where |
|---|---|
| Velona API key (OpenAI-compatible LLM endpoint) | `.env` -> `VELONA_API_KEY` |
| Hugging Face token (optional, raises rate limits) | `.env` -> `HF_TOKEN` |
| Kaggle API token | Kaggle -> Settings -> API -> *Create New Token*, save as `%USERPROFILE%\.kaggle\kaggle.json` |

**Check the connections**

```powershell
python -m scripts.try_parser "predict whether a telecom customer will churn"
python -m scripts.try_mcp
```

`try_mcp` should list the tools of both servers.

---

## Usage

```powershell
python -m dataprep.run "your project idea"
```

Every run gets its own folder:

```
runs\<timestamp>\
├── spec.json            parsed idea
├── queries.json         every search query tried
├── candidates.json      every dataset found
├── ranking.json         scores, LLM fit and reasons
├── raw\                 downloaded files
├── data\                train.csv, val.csv, test.csv   (cleaned, split, not yet transformed)
├── profile.json         statistics of the train split
├── prepare_log.json     cleaning and split decisions
├── proposals.json       the four specialists' raw proposals
├── plan.json            the merged plan + merger log
├── processed\           train.csv, val.csv, test.csv   (model-ready; target = last column)
├── pipeline.pkl         fitted transformer
└── report.md
```

**Using the output**

```python
import joblib, pandas as pd

train = pd.read_csv(r"runs\<timestamp>\processed\train.csv")
X, y = train.iloc[:, :-1], train.iloc[:, -1]

# apply the same preprocessing to new raw rows (same columns as data\train.csv, minus the target)
bundle = joblib.load(r"runs\<timestamp>\pipeline.pkl")
X_new = bundle["transformer"].transform(new_rows)
```

Load `pipeline.pkl` from inside this project (with `dataprep` importable): it contains custom
transformers defined in `dataprep/executor.py`.

### Testing one stage at a time

Each stage can be run on its own against a saved run, which is cheaper than re-running everything:

| Script | Runs | Needs |
|---|---|---|
| `python -m scripts.try_parser "idea"` | stage 1 | - |
| `python -m scripts.try_mcp [--server --tool --arg k=v]` | list / call any MCP tool | - |
| `python -m scripts.try_search "idea"` | stages 1-3, creates a run | - |
| `python -m scripts.try_rank [--run <dir>]` | stage 4 | `candidates.json` |
| `python -m scripts.try_prepare [--run <dir>] [--candidate owner/name]` | stage 5 | `ranking.json` |
| `python -m scripts.try_preprocess [--run <dir>] [--reuse]` | stage 6 | `profile.json` |

`--reuse` re-runs the merger and executor on the saved `proposals.json` with no LLM calls.
Without `--run`, each script uses the newest suitable run.

---

## Configuration

Everything tunable lives in `config.yaml`:

| Section | Main settings |
|---|---|
| `llm` | Velona base URL, `fast` / `smart` model names, retries, timeout |
| `queries` | number of queries per attempt |
| `search` | results per query, how many candidates get details (`enrich_top_k`) |
| `rank` | score weights, `rerank_top_n`, `final_mix`, `weak_threshold`, bad-title words |
| `graph` | `max_retries`, `prepare_fallbacks` |
| `prepare` | download cap (1 GB), split ratios, seed, missing-value tokens, numeric-coercion threshold |
| `agents` / `preprocess` | specialist model, one-hot limits, null-drop threshold |
| `mcp` | server definitions and timeout |

---

## Design decisions

- **MCP for search and metadata only.** File bytes never pass through MCP or the LLM. MCP returns
  a signed URL, and the file is downloaded over plain HTTPS.
- **Read-only tools.** The Kaggle MCP server also exposes `dataset_delete`, `competition_submit`,
  `kernel_push` and others. Only a fixed read-only whitelist is ever loaded.
- **No LLM for parsing.** MCP outputs are parsed with regex and `ast.literal_eval`. LLM replies
  are validated with Pydantic and retried once with the error.
- **No leakage by construction.** Duplicate rows are removed *before* the split. Everything that
  learns from data (medians, percentiles, scalers, category lists) is fitted on the train split only.
- **Code decides, the LLM judges.** Search is a fixed recipe (predictable cost); the LLM is used for
  judgement: understanding the idea, judging datasets, proposing preprocessing.

---

## Limitations

- Tabular data only (v1). Text, image and audio projects stop after parsing.
- Hugging Face and Kaggle only; datasets over 1 GB are refused.
- Free-text columns are dropped, not processed.
- Licenses are scored, not enforced. Check the dataset's license before any commercial use.
- Kaggle row counts are estimated from file size until the file is downloaded.

---

## Project structure

```
dataset-agent\
├── config.yaml
├── .env.example
├── requirements.txt
├── dataprep\
│   ├── config.py          config + Velona models
│   ├── llm_json.py        JSON from the LLM, validated with Pydantic
│   ├── mcp_client.py      MCP connections, read-only whitelist
│   ├── spec.py            ProjectSpec
│   ├── candidate.py       Candidate
│   ├── parse_results.py   parsers for MCP outputs
│   ├── parser.py          1  idea -> spec
│   ├── queries.py         2  spec -> queries
│   ├── search.py          3  queries -> candidates
│   ├── rank.py            4  candidates -> pick
│   ├── download.py        5a download
│   ├── clean.py           5b clean
│   ├── split.py           5c split
│   ├── profile.py         5d profile
│   ├── prepare.py         5  download -> clean -> split -> profile
│   ├── agents.py          6a the four specialists
│   ├── merger.py          6b merge rules
│   ├── executor.py        6c sklearn pipeline
│   ├── report.py          6d report.md
│   ├── graph.py           LangGraph graph
│   └── run.py             entry point
└── scripts\               try_*.py, one stage at a time
```