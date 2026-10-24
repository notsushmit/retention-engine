# Full setup and the data pipeline

The detail behind the README's "Full setup" section: dependency groups, the Kaggle download, and the first three pipeline stages (data, churn labels, features). Every command runs from the repository root.

## Requirements

[uv](https://docs.astral.sh/uv/) installs Python 3.11 itself, so no system Python 3.11 is needed.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once per machine
make setup                                        # Python 3.11, the dev, training, api and dashboard groups, pre-commit hook
make test                                         # ruff + pytest
```

On macOS, LightGBM and XGBoost need the OpenMP runtime: `brew install libomp` (on Linux they use the system libgomp). The scoring stack needs Docker (Docker Desktop on macOS).

Dependency groups, installed with `uv sync --group <name>` (from `pyproject.toml`):

| Group | Contents | Used for |
| --- | --- | --- |
| core (base install) | DuckDB, pandas, pyarrow, NumPy, PyYAML, Kaggle CLI, py7zr | data, labels, features |
| `training` | scikit-learn, LightGBM, XGBoost, lifelines, scikit-survival, MLflow, ruptures | models, trend detection, the registry |
| `models` | `training` plus libraries listed for the explanation, treatment-effect and optimizer phases; several were not used in the end (the README's tech stack lists only what the code imports) | optional |
| `api` | FastAPI, Uvicorn, the redis client (it talks to Valkey), LightGBM, scikit-learn, MLflow, psutil | the scoring service and its image |
| `dashboard` | Streamlit, httpx | the dashboard |
| `dev` | pytest, ruff, pre-commit, httpx | tests and linting |

### Kaggle API setup

1. Easiest: run `uv run kaggle auth login` and approve in the browser. The CLI caches the credentials and you have no token to manage. Then skip to step 3.
2. Or use a token. Sign in at kaggle.com, open **Settings → API** (<https://www.kaggle.com/settings/api>) and generate a token. You get either an API token string or a legacy `kaggle.json`. Store it in one of these ways:
   - **API token, file:** `mkdir -p ~/.kaggle && pbpaste > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token`
   - **Legacy key, file:** `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   - **Environment:** `cp .env.example .env`, then uncomment `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` and `KAGGLE_KEY`. `.env` is git-ignored and the Makefile loads it.
3. Accept the competition rules while signed in: <https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/rules>. Downloads fail with 403 until you do.
4. Check the setup: `uv run kaggle competitions files -c kkbox-churn-prediction-challenge`.

### Download the KKBox data

```bash
make data                       # download, extract, then a 20% customer sample as Parquet
make data SAMPLE_FRACTION=0.5   # a bigger sample later; the 20% customers are all included
make data SKIP_V1_LOGS=1        # skip the large v1 listening log (user_logs.csv)
```

What `make data` does (the file list is in [configs/kkbox.yaml](configs/kkbox.yaml)):

1. **Download** each competition file into `data/raw/archives/.incoming/`. It moves to `data/raw/archives/` only once its size matches what Kaggle reports. An archive already there at the right size is skipped, and one at the wrong size stops the run (archives are never modified or deleted automatically). If a download stops growing for 120 s, it's killed and retried, up to 3 times, resuming from the partial file. Settings are in the config under `download`.
2. **Extract** into `data/raw/csv/`. Files that already exist are skipped, so re-running is safe and never overwrites raw data.
3. **Sample**: keep a customer when the first 8 hex digits of `md5(msno)`, mod 100, are below the sample percent. The same customers are chosen in every table, and smaller samples are subsets of larger ones. Output goes to `data/interim/kkbox/sample_<pct>/<table>.parquet`, with a `manifest.json` that gives rows, customers and date ranges for each table. Column types here are DuckDB's guesses. Phase 1 replaces them with explicit types and validation.

Files:

| File | Set | Contents |
| --- | --- | --- |
| `train_v2.csv` | v2 | churn labels for March 2017 expiries |
| `transactions_v2.csv` | v2 | payment transactions, refreshed |
| `user_logs_v2.csv` | v2 | daily listening logs, refreshed |
| `members_v3.csv` | v2 | member attributes |
| `train.csv` | v1 | churn labels for February 2017 expiries |
| `transactions.csv` | v1 | payment transaction history |
| `user_logs.csv` | v1 | daily listening log history (the large one) |
| `WSDMChurnLabeller.scala` | — | the official labelling script |

Measured on 2026-10-09, on the full files:

| File | Archive | Rows | Customers | Dates |
| --- | --- | --- | --- | --- |
| `transactions.csv` | 708 MB | 21,547,746 | 2,363,626 | transactions 2015-01-01 to 2017-02-28 |
| `transactions_v2.csv` | 49 MB | 1,431,009 | 1,197,050 | transactions 2015-01-01 to 2017-03-31, 75% in March 2017 |
| `user_logs.csv` | 7.14 GB (30.5 GB as CSV, never unpacked) | 392,106,543 | 5,234,111 | 2015-01-01 to 2017-02-28 |
| `user_logs_v2.csv` | 686 MB | 18,396,362 | 1,103,894 | 2017-03-01 to 2017-03-31 |
| `train.csv` | 34 MB | 992,931 | 992,931 | labels, churn rate 6.39% |
| `train_v2.csv` | 33 MB | 970,960 | 970,960 | labels, churn rate 8.99% |

The two listening logs join up with no gap and no overlap: every day from 2015-01-01 to 2017-03-31 is present. Disk use with the 20% sample is about 14 GB: 8.3 GB of archives, 3.6 GB of extracted CSVs (everything except `user_logs.csv`) and 1.9 GB of Parquet.

`user_logs.csv` is never unpacked. `make data` streams it out of the archive and writes only the sampled customers. At 20% this takes about 7.5 minutes (measured 445 s) and writes a 1.6 GB Parquet file.

All data stays under `data/`, which git ignores. The KKBox files are competition data and are not redistributed by this project.

### Churn labels

```bash
make labels                        # all five stages for the sample made by `make data`
uv run python -m retention.labels quality   # or one stage: clean, timeline, quality, labels, compare
```

Churn means no new subscription within 30 days (inclusive) after a membership expiry. The full definition, with the reasons for each choice, is in [docs/churn_definition.md](docs/churn_definition.md); every number that defines a label is in the `labels` block of [configs/kkbox.yaml](configs/kkbox.yaml).

| Stage | Writes to `data/processed/kkbox/sample_<pct>/` | Does |
| --- | --- | --- |
| clean | `transactions`, `members`, `train`, `train_v2` (.parquet) | explicit types, strict date parsing, v1 + v2 union, row flags; structural errors stop the run |
| timeline | `timeline`, `spells` | each customer's effective expiry after every row, and continuous memberships |
| quality | `month_quality`, `snapshot_reliability` | monthly check for months with missing transactions; reliable / uncertain / unreliable |
| labels | `labels` | one row per customer with coverage at each snapshot: time to churn, censoring, and churn flags at the 7, 30 and 60-day windows (the models use day 59, a demonstration) |
| compare | `label_comparison` | agreement with the competition labels, one reason code per disagreement |

Every stage also writes its checks and counts to `validation.json`. On the 20% sample the five stages take about 5 minutes (305 s measured) and write about 410 MB.

### Features

```bash
make features                      # features.parquet for every labelled snapshot (needs make labels)
uv run python -m retention.features docs   # regenerate the feature dictionary from the registry
```

One row per customer and snapshot, keyed to the labels table; every feature for snapshot T reads only rows dated before T. The feature dictionary, with the window and null rule of every column, is in [docs/features.md](docs/features.md).
