"""A throwaway git repository with the real evaluation config, for tracking and guard tests."""

import shutil
import subprocess
from pathlib import Path

CONFIG = Path("configs/evaluation.yaml")


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def make_repo(tmp: Path) -> Path:
    repo = tmp / "repo"
    (repo / "configs").mkdir(parents=True)
    shutil.copy(CONFIG, repo / CONFIG)
    (repo / "README.md").write_text("test repository\n")
    sh(repo, "init", "-q", "-b", "main")
    sh(repo, "config", "user.email", "test@example.com")
    sh(repo, "config", "user.name", "Test")
    sh(repo, "config", "commit.gpgsign", "false")
    commit(repo, "Register config")
    return repo


def commit(repo: Path, message: str) -> str:
    sh(repo, "add", "-A")
    sh(repo, "commit", "-q", "-m", message)
    return sh(repo, "rev-parse", "--short=7", "HEAD")


def tracking_uri(tmp: Path) -> str:
    return f"sqlite:///{tmp / 'store' / 'mlflow.db'}"
