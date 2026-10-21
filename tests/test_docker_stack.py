"""The step 7 stack files against their registration (configs/serving.yaml, step7, step7_images).

No Docker here: the compose file, the Dockerfile, the build-context allowlist and the
Makefile are read as text. One test (local files only) loads the service from a folder
laid out as the container sees it: the configs, a read-only copy of mlruns and the two
pinned files, nothing else.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

from retention.api.config import load_serving
from test_registry_rebase import make_read_only, make_writable, schema_version, snapshot

STEP7 = load_serving()["step7"]
IMAGES = load_serving()["step7_images"]
PINS = load_serving()["step5"]["pins"]
COMPOSE = yaml.safe_load(Path("docker-compose.yml").read_text())
DOCKERFILE = Path("Dockerfile").read_text()
ALLOWED = ["src/", "configs/", "pyproject.toml", "uv.lock", "README.md"]


# --- the compose file ---------------------------------------------------------------------------


def test_the_stack_is_valkey_and_the_api():
    assert sorted(COMPOSE["services"]) == ["api", "valkey"]


def test_every_port_is_bound_to_the_host_loopback_only():
    ports = {name: s["ports"] for name, s in COMPOSE["services"].items()}
    assert ports == {
        "valkey": [f"{STEP7['ports']['valkey']}:6379"],
        "api": [f"{STEP7['ports']['api']}:8000"],
    }
    assert all(p.startswith("127.0.0.1:") for ps in ports.values() for p in ps)


def test_valkey_is_the_pinned_image_with_no_persistence_and_no_eviction():
    v = COMPOSE["services"]["valkey"]
    assert v["image"] == STEP7["images"]["valkey"]
    assert v["command"] == ["valkey-server", "--save", "", "--appendonly", "no"]
    assert "maxmemory" not in " ".join(v["command"])
    assert v["healthcheck"]["test"] == ["CMD", "valkey-cli", "ping"]


def test_the_api_is_read_only_with_tmp_and_reaches_valkey_by_service_name():
    a = COMPOSE["services"]["api"]
    assert a["read_only"] is True and a["tmpfs"] == ["/tmp"]
    assert a["environment"] == {"VALKEY_URL": STEP7["api"]["valkey_url"]}
    assert a["depends_on"] == {"valkey": {"condition": "service_healthy"}}
    assert "env_file" not in a and "deploy" not in a


def test_the_api_memory_limit_is_the_registered_one_and_valkey_has_none():
    limit = load_serving()["api_memory_limit"]
    assert limit["api"] == "1.5GiB" and limit["valkey"] == "none"
    mib = float(limit["api"].removesuffix("GiB")) * 1024
    assert COMPOSE["services"]["api"]["mem_limit"] == f"{int(mib)}m" == "1536m"
    assert "mem_limit" not in COMPOSE["services"]["valkey"]


def test_the_api_mounts_only_mlruns_and_the_two_pinned_files_read_only():
    mounts = COMPOSE["services"]["api"]["volumes"]
    assert all(m.endswith(":ro") for m in mounts)
    pairs = [m.removesuffix(":ro").split(":") for m in mounts]
    assert ["./mlruns", "/app/mlruns"] in pairs
    pinned = {p["path"] for p in PINS.values()}
    assert {src.removeprefix("./") for src, _ in pairs} == {"mlruns"} | pinned
    assert all(dst == "/app/" + src.removeprefix("./") for src, dst in pairs)
    assert not any("data/raw" in m or ".env" in m or "data/interim" in m for m in mounts)


def test_both_services_restart_unless_stopped_and_have_healthchecks():
    for s in COMPOSE["services"].values():
        assert s["restart"] == STEP7["restart"] == "unless-stopped"
        assert s["healthcheck"]["test"]


def test_the_api_healthcheck_treats_200_and_503_as_alive():
    script = COMPOSE["services"]["api"]["healthcheck"]["test"][-1]
    assert "http://127.0.0.1:8000/health" in script
    assert "0 if e.code == 503 else 1" in script


# --- the Dockerfile and the build context --------------------------------------------------------


def from_lines() -> list[str]:
    return re.findall(r"^FROM (\S+)", DOCKERFILE, flags=re.M)


def test_every_base_image_is_the_registered_digest():
    assert from_lines() == [IMAGES["uv"], IMAGES["python"], IMAGES["python"]]
    assert all("@sha256:" in f for f in from_lines())
    assert IMAGES["python"].startswith(STEP7["images"]["python"] + "@")
    assert IMAGES["uv"].startswith(STEP7["images"]["uv"] + "@")


def test_the_image_installs_the_locked_api_group_and_libgomp1():
    assert f"RUN {STEP7['install']} --no-install-project" in DOCKERFILE
    assert f"RUN {STEP7['install']} --no-editable" in DOCKERFILE
    assert STEP7["runtime_packages"] == ["libgomp1"]
    assert "apt-get install -y --no-install-recommends libgomp1" in DOCKERFILE
    assert "GCC Runtime Library Exception 3.1" in DOCKERFILE


def test_the_image_runs_the_registered_command_as_a_non_root_user():
    users = re.findall(r"^USER (\S+)", DOCKERFILE, flags=re.M)
    assert users == ["app"]
    cmd = re.search(r"^CMD (\[.*\])$", DOCKERFILE, flags=re.M)
    assert " ".join(yaml.safe_load(cmd.group(1))) == STEP7["api"]["command"]


def test_the_image_copies_only_code_configs_and_the_lock():
    copies = re.findall(r"^COPY (?!--from)(.+) \S+$", DOCKERFILE, flags=re.M)
    sources = {s for line in copies for s in line.split()}
    assert sources == {"pyproject.toml", "uv.lock", "README.md", "src", "configs"}
    assert "mlruns" not in DOCKERFILE and "data/" not in DOCKERFILE


def test_the_build_context_is_an_allowlist():
    lines = [
        x for x in Path(".dockerignore").read_text().splitlines() if x and not x.startswith("#")
    ]
    assert lines[0] == "*"
    assert [x.removeprefix("!") for x in lines[1:] if x.startswith("!")] == ALLOWED
    assert [x for x in lines[1:] if not x.startswith("!")] == ["**/__pycache__"]


# --- make up / make down -------------------------------------------------------------------------


def target(name: str) -> list[str]:
    text = Path("Makefile").read_text()
    body = re.search(rf"^{name}:\n((?:\t.*\n)+)", text, flags=re.M)
    return [x.strip().lstrip("@") for x in body.group(1).splitlines()]


def test_make_up_waits_for_health_and_prints_it_and_make_down_stops_the_stack():
    up = target("up")
    assert up[0] == STEP7["make"]["up"].split(", then")[0] == "docker compose up -d --wait"
    assert "$(API_URL)/health" in up[1]
    assert target("down") == [STEP7["make"]["down"]]


# --- the container's view of the files (local files only) ---------------------------------------

LOCAL = [Path("mlruns/mlflow.db"), *(Path(p["path"]) for p in PINS.values())]


@pytest.mark.skipif(
    not all(p.is_file() for p in LOCAL), reason="needs the local registry and pinned files"
)
def test_the_service_loads_from_the_container_s_files_alone(tmp_path, monkeypatch):
    """configs, a read-only mlruns copy and the two pinned files: the context loads with
    every registry and pin check, and the registry copy is not written or migrated."""
    from retention.api import service as svc

    root = tmp_path / "app"
    shutil.copytree("configs", root / "configs")
    shutil.copytree("mlruns", root / "mlruns")
    for p in PINS.values():
        (root / p["path"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p["path"], root / p["path"])
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(root)
    make_read_only(root / "mlruns")
    try:
        before = snapshot(root / "mlruns")
        ctx = svc.default_context()
        after = snapshot(root / "mlruns")
    finally:
        make_writable(root / "mlruns")
    ctx.close()
    assert after == before
    assert schema_version(root / "mlruns" / "mlflow.db") == "dc11669786a5"
    assert ctx.models.version == 1 and ctx.config_sha256
