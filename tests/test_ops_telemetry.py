"""Nothing leaves the machine when evidently and prefect run (configs/ops.yaml, amendment A13).

A child process runs a tiny Evidently report and a tiny Prefect flow with every connection to a
non-local address blocked and logged (tests/socket_block/sitecustomize.py, inherited by the
subprocesses Prefect starts). The test fails if any attempt is logged. A control child proves the
block would log an attempt. Needs the ops group.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ops_group import needs_ops_group
from retention.ops.telemetry import registered_environment, telemetry_off

BLOCK = Path("tests/socket_block")


def child(script: str, log: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    for k in registered_environment():  # the child must set the three variables itself
        env.pop(k, None)
    env |= {"SOCKET_LOG": str(log), "PYTHONPATH": f"{BLOCK.resolve()}{os.pathsep}src"}
    return subprocess.run(
        [sys.executable, str(BLOCK / script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )


def attempts(log: Path) -> list[dict]:
    return [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []


def test_the_registered_variables_are_set_and_the_prefect_home_exists(monkeypatch):
    for k in registered_environment():
        monkeypatch.setenv(k, "placeholder")  # restored at the end of the test
    env = telemetry_off()
    assert env == {
        "DO_NOT_TRACK": "1",
        "PREFECT_SERVER_ANALYTICS_ENABLED": "false",
        "PREFECT_HOME": "data/interim/ops/prefect",
    }
    assert all(os.environ[k] == v for k, v in env.items())
    assert Path(env["PREFECT_HOME"]).is_dir()
    assert not Path(env["PREFECT_HOME"]).is_absolute()  # inside the repository, not ~/.prefect


@needs_ops_group
def test_the_block_logs_an_attempt_to_leave(tmp_path):
    log = tmp_path / "log.jsonl"
    done = child("try_connect.py", log)
    assert done.returncode == 0 and "refused" in done.stdout
    got = attempts(log)
    assert got and all("192.0.2.1" in a["target"] for a in got)


@needs_ops_group
def test_a_tiny_evidently_report_and_a_tiny_prefect_flow_never_connect_out(tmp_path):
    home = Path.home() / ".prefect"
    before = sorted(p.name for p in home.iterdir()) if home.exists() else None
    log = tmp_path / "log.jsonl"
    done = child("run_tiny.py", log, str(tmp_path / "drift"))
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["prefect"] == 3 and result["evidently_drifted"] > 0
    assert attempts(log) == []
    after = sorted(p.name for p in home.iterdir()) if home.exists() else None
    assert after == before  # nothing written under ~/.prefect
