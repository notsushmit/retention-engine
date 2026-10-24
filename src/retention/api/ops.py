"""GET /drift and the /model additions: files written by the drift run and the retraining flow.

configs/serving.yaml ops_endpoints. The ops folder (RETENTION_OPS_DIR, default
data/processed/kkbox/sample_020/ops) is mounted read-only; the files are read at each call, at most
1 MB each, and are not pinned (they do not exist when the endpoint is registered), so the body
carries the file's own SHA-256. A file that is missing answers "not monitored"; one that is
unreadable, too big, not valid, naming a never-used word or holding a customer-ID-shaped string
answers "unreadable". Never a 5xx, never a number in an absent answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from retention.api import schemas as sch
from retention.api.labels import DASH, OPS

OPS_DIR_ENV = "RETENTION_OPS_DIR"
DEFAULT_OPS_DIR = Path("data/processed/kkbox/sample_020/ops")
DRIFT_FILE = "drift_latest.json"
RETRAIN_FILE = "retrain_latest.json"
MAX_BYTES = 1_000_000
ID_SHAPED = re.compile(r"[A-Za-z0-9+/]{43}=|synthetic_[0-9a-f]{16}")


def ops_dir() -> Path:
    return Path(os.environ.get(OPS_DIR_ENV) or DEFAULT_OPS_DIR)


def _read(path: Path) -> tuple[str, bytes | None]:
    """('absent' | 'unreadable' | 'ok', the bytes)."""
    if not path.is_file():
        return "absent", None
    try:
        if path.stat().st_size > MAX_BYTES:
            return "unreadable", None
        return "ok", path.read_bytes()
    except OSError:
        return "unreadable", None


def _clean_text(text: str) -> bool:
    return not sch.never_used_hits(text) and not ID_SHAPED.search(text)


def drift_body(directory: Path | None = None) -> dict[str, Any]:
    state, raw = _read((directory or ops_dir()) / DRIFT_FILE)
    if state != "ok":
        return {
            "status": "not monitored" if state == "absent" else "unreadable",
            "label": DASH.drift,
        }
    try:
        data = json.loads(raw)
        hidden = 0
        kept = []
        for c in data.get("columns_drifted", []):
            if sch.never_used_hits(json.dumps(c.get("column", ""))):
                hidden += 1
            else:
                kept.append(c)
        data["columns_drifted"] = kept
        if isinstance(data.get("columns_compared"), dict):
            data["columns_compared"]["hidden_by_rule"] = (
                int(data["columns_compared"].get("hidden_by_rule", 0)) + hidden
            )
        data["report_sha256"] = hashlib.sha256(raw).hexdigest()
        body = sch.DriftBody(**data).model_dump(mode="json")
    except (ValueError, TypeError, AttributeError, ValidationError):
        return {"status": "unreadable", "label": DASH.drift}
    if not _clean_text(json.dumps(body)):
        return {"status": "unreadable", "label": DASH.drift}
    return body


def drift_line(directory: Path | None = None) -> str:
    """The /model drift line: the pointer to GET /drift when a valid drift file is present."""
    return OPS.drift_see if drift_body(directory)["status"] == "monitored" else DASH.drift


def last_retrain(directory: Path | None = None) -> dict[str, Any] | None:
    state, raw = _read((directory or ops_dir()) / RETRAIN_FILE)
    if state != "ok":
        return None
    try:
        body = sch.LastRetrain(**json.loads(raw)).model_dump(mode="json")
    except (ValueError, TypeError, ValidationError):
        return None
    return body if _clean_text(json.dumps(body)) else None
