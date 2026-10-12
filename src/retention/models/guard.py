"""Freeze the fold choices, and open the one test session.

freeze() is the only writer of configs/choices.yaml. It records the frozen
choices, the MLflow runs behind them, every commit that changed
configs/evaluation.yaml (the last one is binding) and that file's SHA-256, and
stores the SHA-256 of the choices file itself in a "freeze" run, so an edit by
hand is detected.

open_test_session() refuses unless every check below holds, in this order:
  1. no tracked file differs from HEAD;
  2. configs/choices.yaml is committed, frozen and names every stage;
  3. its SHA-256 equals the one stored in its freeze run (not edited by hand);
  4. the binding commit is an ancestor of HEAD;
  5. configs/evaluation.yaml at the binding commit has the recorded SHA-256;
  6. the working-tree file has the same SHA-256;
  7. no commit after the binding commit changed it;
  8. every cited run exists and carries the same config SHA-256;
  9. no test run exists (a crashed one may be repeated with a stated reason).
Then it records the test run in MLflow and creates the TestUnlock, the only
place in the code base that can. dry_run() runs the same checks and reports
them without recording or unlocking anything; before the freeze, the checks
that need the choices file are reported as "needs choices". This catches
accidental drift; the pushed commit history is the proof of order.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from retention.models.protocol import TestUnlock
from retention.models.tracking import (
    CONFIG,
    Tracking,
    base_tags,
    check_no_absolute_paths,
    config_commits,
    git,
    sha256_file,
    sha256_text,
    start,
    tree_dirty,
)

CHOICES = Path("configs/choices.yaml")
HEADER = (
    "# Written by the freeze command after the fold stages; never edited by hand.\n"
    "# Its SHA-256 is stored in the freeze run, and the test guard checks it.\n"
)


class GuardRefusal(RuntimeError):
    """The test session cannot open; the message says which check failed."""


def required_stages(repo: Path) -> list[str]:
    return yaml.safe_load((repo / CONFIG).read_text())["selection"]["stages"]


def _run_ids(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [r for v in value.values() for r in _run_ids(v)]
    if isinstance(value, list):
        return [r for v in value for r in _run_ids(v)]
    return [str(value)]


def freeze(
    tr: Tracking,
    choices: dict[str, Any],
    run_ids: dict[str, Any],
    binding_commit: str | None = None,
) -> str:
    """Write configs/choices.yaml; returns the freeze run id. Commit the file alone."""
    repo = tr.repo
    missing = [s for s in required_stages(repo) if choices.get(s) in (None, "", {}, [])]
    if missing:
        raise ValueError(f"cannot freeze: no choice for {missing}")
    commits = config_commits(repo)
    doc_base = {
        "frozen": True,
        "preregistration_commits": commits,
        "binding_commit": binding_commit or commits[-1],
        "evaluation_config_sha256": sha256_file(repo / CONFIG),
    }
    with start(tr, "freeze", "freeze", {}) as run:
        doc = {
            **doc_base,
            "freeze_run_id": run.info.run_id,
            "choices": choices,
            "run_ids": run_ids,
        }
        text = HEADER + yaml.safe_dump(doc, sort_keys=False)
        (repo / CHOICES).write_text(text)
        mlflow.set_tag("choices_sha256", sha256_text(text))
    return run.info.run_id


def blob_sha256(repo: Path, commit: str, path: Path) -> str:
    """SHA-256 of a file exactly as stored at a commit (bytes, no stripping)."""
    out = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True, check=True
    )
    return hashlib.sha256(out.stdout).hexdigest()


@dataclass(frozen=True)
class CheckResult:
    check: int
    name: str
    status: str  # "passed", "failed", "needs choices" (before the freeze), "needs commit"
    message: str = ""


@dataclass
class Checks:
    results: list[CheckResult] = field(default_factory=list)
    text: str = ""
    binding: str = ""
    previous: list = field(default_factory=list)

    def add(self, check: int, name: str, ok: bool, message: str) -> bool:
        self.results.append(
            CheckResult(check, name, "passed" if ok else "failed", "" if ok else message)
        )
        return ok

    def pending(self, check: int, name: str, message: str) -> None:
        self.results.append(CheckResult(check, name, "needs choices", message))

    def first_problem(self) -> CheckResult | None:
        return next((r for r in self.results if r.status != "passed"), None)


def run_checks(tr: Tracking, rerun_reason: str | None = None) -> Checks:
    """Every precondition of the test session, in order; never raises, never unlocks.

    Before the freeze (no configs/choices.yaml), checks 2, 3 and 8 need the
    choices file and are reported as such; the binding commit is then the last
    commit that changed the config and the recorded hash is the working tree's.
    """
    repo, client, out = tr.repo, MlflowClient(tr.uri), Checks()
    out.add(1, "clean tree", not tree_dirty(repo), "a tracked file differs from HEAD")

    path = repo / CHOICES
    doc: dict[str, Any] | None = None
    if not path.exists():
        for n, name in ((2, "choices committed and complete"), (3, "choices not edited")):
            out.pending(n, name, "configs/choices.yaml does not exist yet (before the freeze)")
    else:
        tracked = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--error-unmatch", str(CHOICES)],
            capture_output=True,
        )
        out.text = path.read_text()
        doc = yaml.safe_load(out.text)
        missing = [
            s for s in required_stages(repo) if doc.get("choices", {}).get(s) in (None, "", {}, [])
        ]
        if tracked.returncode:
            out.results.append(
                CheckResult(
                    2,
                    "choices committed and complete",
                    "needs commit",
                    "configs/choices.yaml is written but not committed",
                )
            )
        elif doc.get("frozen") is not True:
            out.add(
                2, "choices committed and complete", False, "configs/choices.yaml is not frozen"
            )
        else:
            out.add(
                2,
                "choices committed and complete",
                not missing,
                f"configs/choices.yaml is incomplete: no choice for {missing}",
            )
        try:
            tags = client.get_run(doc["freeze_run_id"]).data.tags
            ok = tags.get("run_kind") == "freeze" and tags.get("choices_sha256") == sha256_text(
                out.text
            )
            out.add(
                3,
                "choices not edited",
                ok,
                "configs/choices.yaml was edited after the freeze command wrote it",
            )
        except (MlflowException, KeyError):
            out.add(
                3,
                "choices not edited",
                False,
                "the freeze run of configs/choices.yaml does not exist",
            )

    if doc is not None and "binding_commit" in doc:
        binding, recorded = doc["binding_commit"], doc["evaluation_config_sha256"]
    else:
        commits = config_commits(repo)
        binding, recorded = (commits[-1] if commits else "HEAD"), sha256_file(repo / CONFIG)
    out.binding = binding
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", binding, "HEAD"],
        capture_output=True,
    )
    is_ancestor = out.add(
        4,
        "binding commit is an ancestor",
        not ancestor.returncode,
        f"the binding commit {binding} is not an ancestor of HEAD",
    )
    out.add(
        5,
        "config at the binding commit",
        is_ancestor and blob_sha256(repo, binding, CONFIG) == recorded,
        f"configs/evaluation.yaml at {binding} does not have the recorded SHA-256",
    )
    out.add(
        6,
        "config in the working tree",
        sha256_file(repo / CONFIG) == recorded,
        "the working-tree configs/evaluation.yaml does not have the recorded SHA-256",
    )
    later = (
        git(repo, "log", "--format=%h", f"{binding}..HEAD", "--", str(CONFIG))
        if is_ancestor
        else "?"
    )
    out.add(
        7,
        "no later config change",
        not later,
        f"configs/evaluation.yaml was changed after the binding commit {binding}",
    )

    if doc is None:
        out.pending(8, "cited runs", "the cited runs are listed in configs/choices.yaml")
    else:
        problem = ""
        for run_id in _run_ids(doc.get("run_ids", {})):
            try:
                if client.get_run(run_id).data.tags.get("config_sha256") != recorded:
                    problem = f"cited run {run_id} was made under a different config"
                    break
            except MlflowException:
                problem = f"cited run {run_id} does not exist"
                break
        out.add(8, "cited runs", not problem, problem)

    out.previous = client.search_runs(
        [tr.experiment_id], filter_string="tags.run_kind = 'test'", max_results=1000
    )
    if any(r.info.status in ("FINISHED", "RUNNING", "SCHEDULED") for r in out.previous):
        out.add(
            9,
            "no test run yet",
            False,
            "a test run already exists; the test months are scored once",
        )
    else:
        out.add(
            9,
            "no test run yet",
            not out.previous or bool(rerun_reason),
            "a crashed test run exists; repeat it only with a stated reason",
        )
    return out


def dry_run(tr: Tracking) -> list[CheckResult]:
    """All precondition checks against the repository, without recording or unlocking anything."""
    return run_checks(tr).results


def open_test_session(tr: Tracking, rerun_reason: str | None = None) -> tuple[TestUnlock, str]:
    """Run every check; on success record the test run and return (unlock, run id)."""
    checks = run_checks(tr, rerun_reason)
    problem = checks.first_problem()
    if problem is not None:
        raise GuardRefusal(f"check {problem.check}: {problem.message}")

    tags = base_tags(tr, "test", {}) | {
        "unlock_issued": "true",
        "choices_sha256": sha256_text(checks.text),
        "binding_commit": checks.binding,
    }
    if checks.previous:
        tags |= {
            "rerun_reason": rerun_reason,
            "rerun_of": ",".join(r.info.run_id for r in checks.previous),
        }
    check_no_absolute_paths(tags)
    run = MlflowClient(tr.uri).create_run(tr.experiment_id, tags=tags, run_name="test")
    unlock = TestUnlock()
    return unlock, run.info.run_id
