"""The ops workflow and the dashboard image: static checks (nothing is built or run here).

The workflow has not run yet; these tests read it and the Dockerfiles as text and YAML.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

OPS_YML = Path(".github/workflows/ops.yml")
CI_YML = Path(".github/workflows/ci.yml")
OPS_CONFIG = yaml.safe_load(Path("configs/ops.yaml").read_text())


def workflow() -> dict:
    return yaml.safe_load(OPS_YML.read_text())


def steps_text(job: str) -> str:
    return "\n".join(
        f"{s.get('uses', '')} {s.get('run', '')}" for s in workflow()["jobs"][job]["steps"]
    )


def test_the_workflow_runs_on_pushes_to_main_and_on_path_filtered_pull_requests():
    on = workflow()[True]  # YAML reads the key "on" as true
    assert on["push"] == {"branches": ["main"]}
    paths = on["pull_request"]["paths"]
    for needed in (
        "configs/ops.yaml",
        "pyproject.toml",
        "uv.lock",
        "src/retention/synthetic/**",
        "Dockerfile",
        "Dockerfile.dashboard",
        "docker-compose.yml",
        ".github/workflows/ops.yml",
    ):
        assert needed in paths, needed
    assert "branches" not in on["pull_request"] and len(on) == 2


def test_the_smoke_job_timeout_is_the_registered_ceiling():
    ceiling = OPS_CONFIG["smoke"]["job_ceiling_minutes"]
    assert workflow()["jobs"]["smoke"]["timeout-minutes"] == ceiling == 10
    assert workflow()["jobs"]["images"]["timeout-minutes"] > 0


def test_the_telemetry_off_variables_are_the_registered_ones():
    env = workflow()["env"]
    for k, v in OPS_CONFIG["amendment_A13"]["environment"].items():
        assert env[k] == v, k
    assert env["OPS_TESTS_MUST_RUN"] == "1"


def test_nothing_is_pushed_logged_in_or_secret():
    text = "\n".join(
        ln.split(" #")[0]
        for ln in OPS_YML.read_text().splitlines()
        if not ln.lstrip().startswith("#")
    )
    for banned in ("secrets.", "docker login", "docker push", "--push", "registry", "GITHUB_TOKEN"):
        assert banned not in text, banned
    assert workflow()["permissions"] == {"contents": "read"}


def test_both_images_are_built_from_their_own_dockerfile():
    text = steps_text("images")
    assert "docker build -f Dockerfile -t retention-api:ci ." in text
    assert "docker build -f Dockerfile.dashboard -t retention-dashboard:ci ." in text


def test_actions_are_pinned_to_the_versions_the_main_workflow_uses():
    main = dict(re.findall(r"uses: ([\w./-]+)@([\w.]+)", CI_YML.read_text()))
    used = dict(re.findall(r"uses: ([\w./-]+)@([\w.]+)", OPS_YML.read_text()))
    assert used and all(re.fullmatch(r"v\d+(\.\d+)*", v) for v in used.values())
    assert {k: v for k, v in used.items() if k in main} == {k: main[k] for k in used if k in main}
    assert set(used) <= set(main)  # no new action


def test_every_pytest_run_shows_its_skips():
    runs = re.findall(r"uv run pytest[^\n]*", OPS_YML.read_text())
    assert runs and all(" -rs" in r for r in runs)


def test_only_the_ops_workflow_installs_the_ops_group():
    smoke = steps_text("smoke")
    assert "--group ops" in smoke and "--locked" in smoke
    assert "ops" not in re.findall(r"--group (\w+)", CI_YML.read_text())


def test_the_smoke_job_runs_the_synthetic_path_and_the_ops_tests():
    smoke = steps_text("smoke")
    assert "uv run python -m retention.synthetic all" in smoke
    for t in ("tests/test_synthetic.py", "tests/test_ops_config.py", "tests/test_ops_group.py"):
        assert t in smoke
    checkout = workflow()["jobs"]["smoke"]["steps"][0]
    assert checkout["with"] == {"fetch-depth": 0}  # the config tests read the registered commits


def test_the_workflow_says_no_registry_account_or_hosting():
    head = "\n".join(OPS_YML.read_text().splitlines()[:5])
    assert "No push to any registry" in head and "Nothing here is hosted" in head


# --- the dashboard image ----------------------------------------------------------------------


API_DOCKERFILE = Path("Dockerfile")
DASH_DOCKERFILE = Path("Dockerfile.dashboard")


def test_the_dashboard_image_uses_the_pinned_bases_of_the_api_image():
    def froms(path: Path) -> list[str]:
        return re.findall(r"^FROM (\S+)", path.read_text(), re.M)

    api, dash = froms(API_DOCKERFILE), froms(DASH_DOCKERFILE)
    assert dash[0] == api[0] and dash[1] == api[1] and dash[2] == api[2]
    assert all("@sha256:" in f for f in dash)


def test_the_dashboard_image_installs_only_the_dashboard_group_and_copies_no_data():
    text = DASH_DOCKERFILE.read_text()
    assert "uv sync --frozen --no-dev --group dashboard --no-install-project" in text
    assert "--group api" not in text and "--group ops" not in text
    copies = re.findall(r"^COPY (?:--from=\w+ )?(\S+)", text, re.M)
    assert set(copies) == {
        "/uv",
        "pyproject.toml",
        "/app/.venv",
        "configs",
        "dashboard",
        ".streamlit",
    }
    assert "data" not in copies and "mlruns" not in copies


def test_the_dashboard_image_runs_as_a_user_without_secrets_and_binds_inside_only():
    text = DASH_DOCKERFILE.read_text()
    assert re.search(r"^USER app$", text, re.M) and "useradd --system" in text
    assert not re.search(r"(?i)token|password|secret|api_key", text)
    assert '"--server.address", "0.0.0.0"' in text and "EXPOSE 8501" in text


def test_the_build_context_allows_the_dashboard_and_nothing_under_data():
    allow = [ln for ln in Path(".dockerignore").read_text().splitlines() if ln.startswith("!")]
    assert "!dashboard/" in allow and "!.streamlit/" in allow
    assert not [a for a in allow if a.startswith(("!data", "!mlruns", "!tests"))]


@pytest.mark.skipif(shutil.which("uv") is None, reason="not checked: uv is not on the PATH")
def test_every_package_the_dashboard_imports_is_in_the_dashboard_group():
    """Found while writing the image: PyYAML was in no group an image installs."""
    out = subprocess.run(
        ["uv", "export", "--only-group", "dashboard", "--no-hashes", "--no-emit-project"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    locked = {m.group(1).lower() for m in re.finditer(r"^([A-Za-z0-9_.-]+)==", out, re.M)}
    imports = set()
    for f in Path("dashboard").rglob("*.py"):
        imports |= set(re.findall(r"^(?:from|import) (\w+)", f.read_text(), re.M))
    third_party = {
        "altair": "altair",
        "httpx": "httpx",
        "pandas": "pandas",
        "streamlit": "streamlit",
    }
    third_party["yaml"] = "pyyaml"
    for module, dist in third_party.items():
        assert module in imports, module
        assert dist in locked, f"{module} is imported but {dist} is not in the dashboard group"
