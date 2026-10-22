.PHONY: setup data labels features train test lint up down dashboard

# Load .env if present (Kaggle credentials, SAMPLE_FRACTION).
-include .env
export

PY := uv run python -u
SAMPLE_FRACTION ?=
SKIP_V1_LOGS ?=
DATA_ARGS := $(if $(SKIP_V1_LOGS),--skip-group v1_logs,)
SAMPLE_ARGS := $(if $(SAMPLE_FRACTION),--fraction $(SAMPLE_FRACTION),)

setup:
	uv python install 3.11
	uv sync --group dev --group training --group api --group dashboard
	uv run pre-commit install

data:
	mkdir -p data/raw data/interim data/processed
	$(PY) -m retention.data download $(DATA_ARGS)
	$(PY) -m retention.data sample $(DATA_ARGS) $(SAMPLE_ARGS)

# Typed tables, timelines, data-quality check, labels and the comparison with
# the competition labels, for the sample made by `make data`.
labels:
	$(PY) -m retention.labels all $(SAMPLE_ARGS)

# Feature table for every labelled snapshot of the sample (needs `make labels`).
features:
	$(PY) -m retention.features build $(SAMPLE_ARGS)

train:
	@echo "make train is not implemented until Phase 3" && exit 1

# The scoring stack (configs/serving.yaml, step7): Valkey and the API on 127.0.0.1.
# /health answers 503 until a store is loaded (load-snapshot, from the host).
API_URL ?= http://127.0.0.1:8000

up:
	docker compose up -d --wait
	@curl -s -w ' (HTTP %{http_code})\n' $(API_URL)/health

down:
	docker compose down

# The dashboard (configs/dashboard.yaml serve): reads only the API at RETENTION_API_URL
# (default http://127.0.0.1:8000), served on 127.0.0.1:8501 only.
dashboard:
	uv run streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501

lint:
	uv run ruff check .
	uv run ruff format --check .

test: lint
	uv run pytest
