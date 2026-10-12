"""MLflow runs for the model stages: names, tags, data hashes, resources and git facts.

Every run is tagged with its kind (label_check, fold, calibration_fold,
check_month, freeze, test, sensitivity), the tracking backend, the git commit
and whether the tree had uncommitted changes, the SHA-256 of
configs/evaluation.yaml, the binding pre-registration commit (the last commit
that changed that file), the SHA-256 of the data files used, package versions,
and, at the end, runtime and peak memory (tags under `timing.`).

Paths in tags and parameters are repository-relative; MLflow's own
mlflow.source.name (an absolute script path by default) is overwritten, and a
run whose tags or parameters contain "/Users/" or the home directory raises.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from retention.models.resources import peak_rss_gb

CONFIG = Path("configs/evaluation.yaml")
PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "lifelines",
    "mlflow",
)
SOURCE = "src/retention/models/__main__.py"


class AbsolutePathError(ValueError):
    """A tag or parameter would publish an absolute local path."""


# --- git and hashes ------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout.strip()


def head_commit(repo: Path) -> str:
    return git(repo, "rev-parse", "--short=7", "HEAD")


def tree_dirty(repo: Path) -> bool:
    """True if a tracked file differs from HEAD (untracked files are ignored)."""
    return bool(git(repo, "status", "--porcelain", "--untracked-files=no"))


def config_commits(repo: Path, config: Path = CONFIG) -> list[str]:
    """Every commit that changed the evaluation config, oldest first (short hashes)."""
    out = git(repo, "log", "--format=%h", "--abbrev=7", "--reverse", "--", str(config))
    return out.split() if out else []


@lru_cache(maxsize=32)
def _sha256(path: str, size: int, mtime: float) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    st = path.stat()
    return _sha256(str(path), st.st_size, st.st_mtime)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- absolute paths ---------------------------------------------------------------------


def check_no_absolute_paths(values: Mapping[str, Any]) -> None:
    home = str(Path.home())
    for k, v in values.items():
        s = str(v)
        if "/Users/" in s or (home not in ("", "/") and home in s):
            raise AbsolutePathError(f"{k} holds an absolute local path")


def relative(path: Path, repo: Path) -> str:
    """A repository-relative path string for tags and parameters."""
    p = Path(path)
    return str(p.resolve().relative_to(repo.resolve())) if p.is_absolute() else str(p)


# --- setup and runs ----------------------------------------------------------------------


@dataclass(frozen=True)
class Tracking:
    repo: Path
    uri: str
    backend: str
    experiment_id: str


def setup(repo: Path, uri: str | None = None) -> Tracking:
    """Open the configured SQLite store; if it fails, the folder store (tagged as such).

    MLflow 3 keeps the folder store in maintenance mode and refuses it unless
    MLFLOW_ALLOW_FILE_STORE is true, so the fallback sets that variable.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    cfg = yaml.safe_load((repo / CONFIG).read_text())["tracking"]
    sqlite = uri or "sqlite:///" + str(repo / cfg["uri"].removeprefix("sqlite:///"))
    folder = (repo / cfg["fallback_uri"]).resolve().as_uri()
    for backend, target in (("sqlite", sqlite), ("folder", folder)):
        if backend == "folder":
            # MLflow 3 refuses the folder store (maintenance mode) unless this is set.
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        try:
            mlflow.set_tracking_uri(target)
            exp = mlflow.get_experiment_by_name(cfg["experiment"])
            exp_id = (
                exp.experiment_id
                if exp
                else mlflow.create_experiment(
                    cfg["experiment"], artifact_location=(repo / cfg["artifact_root"]).as_uri()
                )
            )
            return Tracking(repo, target, backend, exp_id)
        except Exception:
            if backend == "folder":
                raise
    raise AssertionError("unreachable")


def run_name(
    kind: str, where: str, family: str, candidate: str, horizon: int, feature_set: str
) -> str:
    parts = [kind, where, family, candidate, f"h{horizon}", feature_set]
    if any(not p or " " in p for p in parts):
        raise ValueError(f"run name parts must be non-empty without spaces: {parts}")
    return "-".join(parts)


def base_tags(tr: Tracking, kind: str, data_files: Mapping[str, Path]) -> dict[str, str]:
    repo = tr.repo
    run_kinds = yaml.safe_load((repo / CONFIG).read_text())["tracking"]["run_kinds"]
    if kind not in run_kinds:
        raise ValueError(f"unknown run kind {kind!r}")
    commits = config_commits(repo)
    tags = {
        "run_kind": kind,
        "tracking_backend": tr.backend,
        "git_commit": head_commit(repo),
        "git_dirty": str(tree_dirty(repo)).lower(),
        "config_sha256": sha256_file(repo / CONFIG),
        "preregistration_commit": commits[-1] if commits else "",
        "version.python": platform.python_version(),
        "mlflow.source.name": SOURCE,
    }
    tags |= {f"version.{p}": version(p) for p in PACKAGES}
    tags |= {f"sha256.{name}": sha256_file(path) for name, path in data_files.items()}
    return tags


@contextmanager
def start(
    tr: Tracking,
    kind: str,
    name: str,
    params: Mapping[str, Any],
    tags: Mapping[str, str] | None = None,
    data_files: Mapping[str, Path] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """A tracked run; data_files maps a short name (for example "labels") to a file."""
    all_tags = base_tags(tr, kind, data_files or {}) | dict(tags or {})
    check_no_absolute_paths(all_tags)
    check_no_absolute_paths(params)
    t0 = time.time()
    with mlflow.start_run(experiment_id=tr.experiment_id, run_name=name, tags=all_tags) as run:
        mlflow.set_tag("mlflow.source.name", SOURCE)  # MLflow sets the absolute script path
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        yield run
        mlflow.set_tags(
            {
                "timing.seconds": f"{time.time() - t0:.1f}",
                "timing.peak_rss_gb": f"{peak_rss_gb():.3f}",
            }
        )
    logged = MlflowClient(tr.uri).get_run(run.info.run_id).data
    check_no_absolute_paths(logged.tags)
    check_no_absolute_paths(logged.params)
