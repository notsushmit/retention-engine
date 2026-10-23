# Retention decision engine

Predicts which subscribers will churn and when, explains why, estimates which retention action works for each customer, and allocates a retention budget. The primary data is the KKBox churn prediction challenge (WSDM Cup 2018).

A customer is anyone with at least one paid transaction or a full discount on a priced plan. Free trials and free promotional periods alone do not count.

Status: Phase 1 (data and churn labels).

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.11 itself, so you don't need a system Python 3.11.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once per machine
make setup                                        # Python 3.11, dev and training dependencies, pre-commit hook
make test                                         # ruff + pytest
```

On macOS, LightGBM and XGBoost need the OpenMP runtime: `brew install libomp` (on Linux they use the system libgomp).

Dependency groups, installed with `uv sync --group <name>`:

| Group | Contents | Needed from |
| --- | --- | --- |
| core (base install) | DuckDB, pandas, pyarrow, Kaggle CLI, py7zr | Phase 0 |
| `training` | scikit-learn, LightGBM, XGBoost, lifelines, scikit-survival, MLflow | Phase 3 |
| `models` | `training` plus SHAP, UMAP, EconML, DiCE, ruptures, OR-Tools | Phases 4 to 7 |
| `serving` | FastAPI, Redis, Streamlit, Evidently, Prefect | Phase 8 |
| `dev` | pytest, ruff, pre-commit | Phase 0 |

## Kaggle API setup

1. Easiest: run `uv run kaggle auth login` and approve in the browser. The CLI caches the credentials and you have no token to manage. Then skip to step 3.
2. Or use a token. Sign in at kaggle.com, open **Settings → API** (<https://www.kaggle.com/settings/api>) and generate a token. You get either an API token string or a legacy `kaggle.json`. Store it in one of these ways:
   - **API token, file:** `mkdir -p ~/.kaggle && pbpaste > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token`
   - **Legacy key, file:** `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   - **Environment:** `cp .env.example .env`, then uncomment `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` and `KAGGLE_KEY`. `.env` is git-ignored and the Makefile loads it.
3. Accept the competition rules while signed in: <https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/rules>. Downloads fail with 403 until you do.
4. Check it works: `uv run kaggle competitions files -c kkbox-churn-prediction-challenge`.

## Download the KKBox data

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

`user_logs.csv` is never unpacked. `make data` streams it out of the archive and keeps only sampled customers. At 20% this takes about 7.5 minutes (measured 445 s) and writes a 1.6 GB Parquet file.

All data stays under `data/`, which git ignores. The KKBox files are competition data and are not redistributed by this project.

## Churn labels

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
| labels | `labels` | one row per customer with coverage at each snapshot: time to churn, censoring, churn within 7 / 30 / 60 days |
| compare | `label_comparison` | agreement with the competition labels, one reason code per disagreement |

Every stage also writes its checks and counts to `validation.json`. On the 20% sample the five stages take about 5 minutes (305 s measured) and write about 410 MB.

## Features

```bash
make features                      # features.parquet for every labelled snapshot (needs make labels)
uv run python -m retention.features docs   # regenerate the feature dictionary from the registry
```

One row per customer and snapshot, keyed to the labels table; every feature for snapshot T reads only rows dated before T. The feature dictionary, with the window and null rule of every column, is in [docs/features.md](docs/features.md).

## Budget optimizer

```bash
uv run python -m retention.optimize recount    # counts only: prices, customers without a usable price
uv run python -m retention.optimize value      # dt01 refit with an exact reproduction check, value inputs
uv run python -m retention.optimize plan       # every scenario and budget level, strategies, stability
uv run python -m retention.optimize observed   # one read of 2016 outcomes per snapshot
uv run python -m retention.optimize report     # reports/strategy_backtest.md from the logged runs
```

Chooses at most one action per customer (a reminder, an auto-renew prompt or a discount) to maximize the expected net value within a budget. Every effect, cost and margin is a labelled assumption in `configs/actions.yaml`; no KKBox effect of any action has been estimated. KKBox backtest figures are model estimates under assumed effects, not observed outcomes. The comparison with simpler targeting rules (highest risk, highest assumed uplift x value, random, reminder only, no action) is arithmetic on assumed effects, and the optimizer maximizes that quantity by design. The observed churn of each strategy's targeted customers says nothing about whether any action works. Each step runs once, at a clean commit; the results are in [reports/strategy_backtest.md](reports/strategy_backtest.md) once the steps have run.

## Scoring service

> The service replays 2016 KKBox data; there is no live source.
>
> Customers expiring more than 30 days ahead (22% of active customers on 2016-11-01) get no score.
>
> Plans exist for two dates only (2016-10-01 and 2016-11-01) and were built from the 30-day probability before the ordering fix.
>
> KKBox backtest figures are model estimates under assumed effects, not observed outcomes.

A FastAPI service answers `GET /health`, `POST /score`, `GET /at-risk` and `GET /alerts` from a Valkey store loaded for one snapshot date, 2016-11-01. The models load from the MLflow registry alias `retention-scoring@serving`, and every model file is checked by SHA-256 at startup and on every call. Risk is given at 7, 30 and 59 days, one day short of the phase plan's 60-day horizon; every 59-day probability is marked as a demonstration. Endpoints and fields: [docs/scoring_api.md](docs/scoring_api.md); every comparison with batch scoring: [reports/serving_parity.md](reports/serving_parity.md).

```bash
make up    # Valkey on 127.0.0.1:6379 and the API on 127.0.0.1:8000, then /health
uv run python -m retention.api load-snapshot --snapshot 2016-11-01
make down
```

Valkey writes nothing to disk, so the store is empty after any restart and `/health` answers 503 until the snapshot is loaded again from the host; a load after the first one needs `--reload-reason`.

Parity over HTTP: the registered mark compared 1,000 seeded answers from the container with the host's answers as exact strings, and it was not met: 996 of 1,000 matched. The other 4 differed only in float values, at most 4 ulps apart (the smallest steps a float can take), with no text, action name or flag different; the likely source, untested, is the container's Linux libraries against the Mac's. Amendment A10, written after that result was seen, judges floats within a relative 1e-12 and every other field exactly; under it all 1,000 pass, and the 996 of 1,000 stays on record as not met as written. Step 6's score parity is a separate result: on all 169,673 scored rows of 2016-11-01 the replayed store gave the same scores as batch within the registered tolerance (final probabilities within a relative 1e-12; the largest gap was 0.0). The HTTP sample was 1,000 of the 169,673 scored customers (0.6%), drawn with one seed, scored customers only, at one store date; outside_population and not_active answers are covered by neither the parity mark nor the step 8 HTTP checks, and were checked in-process only (the tests and the step 5 smoke run, not over HTTP).

Latency, p95 on this Mac, one run of 1,000 calls: p95 69.429 ms, with p99 97.809 ms and a maximum of 108.142 ms, measured before the 1.5 GiB limit and before the dashboard endpoints, over loopback with one client and one connection, against a threshold of 100 ms (my unvalidated choice). The p50 was 59.242 ms. Every timed call includes the service's check of its model and pinned files, about 15 ms in an earlier in-process measurement.

Memory: the API container is limited to 1.5 GiB, my choice. After one filtered `GET /at-risk` call it used 672.4 MiB, measured with swap above 0 (97.38 MB) and no concurrent clients; the filter index together with 8 concurrent clients was not measured, so the headroom is not established.

Unverified: `ps` showed Docker's VM process at 9.83 GB resident, above the VM's 7.75 GiB, while free memory read 74% and swap 0, and this is not explained; the API image is listed at 1.47GB by `docker images` and at 322,050,115 bytes by `docker image inspect`, and the two sizes are not reconciled. The figures 672.4 MiB, 97.38 MB and 9.83 GB are console readings (docker stats, sysctl, ps) with no report written to disk.

## Make targets

| Target | Does |
| --- | --- |
| `make setup` | installs Python 3.11, dev dependencies and the pre-commit hook |
| `make data` | downloads, extracts and samples the KKBox data |
| `make labels` | typed tables, timelines, data-quality check, churn labels, comparison with the competition labels |
| `make test` | ruff check, ruff format check, pytest |
| `make features` | leakage-safe feature table for every labelled snapshot (docs/features.md) |
| `make train` | Phase 3 (not yet implemented) |
| `make up` | starts Valkey and the scoring API (docker compose), then prints `/health` |
| `make down` | stops them |
| `make dashboard` | serves the dashboard on 127.0.0.1:8501, reading only the API |

## Dashboard

> Replay of 2016 data, not real-time.

A Streamlit app with five pages that reads only the scoring API: `GET /health`, `/at-risk`, `/alerts`, `/summary`, `/plans` and `/model`, and `POST /score`. It has no store client and no data reader, and it reads no file under `data/`; a test fails if one appears. Pages, API calls, tooltips and every change since the plan: [docs/dashboard.md](docs/dashboard.md).

```bash
make up         # the scoring API on 127.0.0.1:8000 (load the snapshot as above)
make dashboard  # the dashboard on 127.0.0.1:8501 only; RETENTION_API_URL must be a loopback address
```

| Page | What it answers |
| --- | --- |
| Overview | how many scored customers are expected to lapse in days 1 to 7, 8 to 30 and 31 to 59 (a demonstration), the reason groups the model lists first among the 1,000 customers most at risk, the knowable trend alerts, and the top 20 at risk |
| At-risk list | who is most at risk at 7, 30 or 59 days, with the auto_renew and in_plan filters and a CSV of the displayed rows |
| Customer profile | one customer's three probabilities, two reasons, expected paid days and the labelled action block |
| Action planner | the plans precomputed in Phase 7 for the store's date, per budget level, with the strategy comparison; nothing is re-planned |
| Model health | the model version, its eight file hashes, the config hash and the calibration table before and after the ordering fix; drift is not monitored in this phase |

![Overview, capture mode](docs/img/dashboard/overview.png)

![Action planner](docs/img/dashboard/planner.png)

![Model health](docs/img/dashboard/model_health.png)

The Overview tile "Expiring more than 30 days ahead" shows 22%: 51,692 of the 230,286 customers with coverage on 2016-11-01 expire more than 30 days ahead and get no score. It is not the whole population left out: with the trial-only (8,890) and promo-only (31) customers, the customers outside the population are 60,613 of the same 230,286, about 26%.

Look: one cream background, near-black text, one orange accent used only as a fill for the leading at-risk item, and greys. The eight colours were sampled from a reference slide: background #F4EDE3, ink #090B0E, accent #FE5A00, rust #942A11 (orange text), card grey #1B222C, bar grey #40454B, body grey #413A31, label grey #514A40. The orange's contrast on the background is 2.71:1, so it is never used for text. The fonts are self-hosted, with no font service: Space Grotesk (headings and body), Instrument Serif italic (the one big number on each page) and JetBrains Mono (small labels), each under the SIL Open Font License 1.1, whose text sits beside the font files in `dashboard/static/fonts/`.

Captures: the screenshots above come from three pages only. The Overview is captured with `?capture=1`, which leaves out its top-20 table and the cancel events; the Action planner and Model health show no customer ID. The At-risk list and the Customer profile show customer IDs and are never captured; `scripts/capture_pages.py` refuses them. Capture mode is a convenience for screenshots, not access control: anyone who removes `?capture=1` sees the IDs again. The dashboard must never be hosted with customer rows.
