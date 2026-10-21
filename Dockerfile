# The scoring API image (configs/serving.yaml, step7 and step7_images).
# Build stage: uv installs the locked "api" group into /app/.venv (no editable install).
# Runtime stage: the venv, the configs, libgomp1 for LightGBM, a non-root user.
FROM ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a AS uv

FROM python:3.11.17-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf AS build
COPY --from=uv /uv /bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group api --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --group api --no-editable

FROM python:3.11.17-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf
# libgomp1: GCC's OpenMP runtime for LightGBM; GPL-3.0-or-later with the GCC Runtime Library Exception 3.1.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY configs ./configs
ENV PATH=/app/.venv/bin:$PATH \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1
USER app
EXPOSE 8000
CMD ["uvicorn", "--factory", "retention.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
