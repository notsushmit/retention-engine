"""Downloads are size-checked and retried on stalls; extraction never overwrites raw files."""

import hashlib
import sys
import time
import zipfile
from pathlib import Path

import py7zr
import pytest

from retention.data.download import ensure_archive, extract


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nested_archive(tmp_path: Path) -> Path:
    """A zip holding a 7z that holds a CSV inside a folder, like Kaggle downloads can be."""
    src = tmp_path / "src"
    (src / "data" / "churn_comp_refresh").mkdir(parents=True)
    (src / "data" / "churn_comp_refresh" / "members.csv").write_text("msno,city\na,1\n")
    inner = tmp_path / "members.csv.7z"
    with py7zr.SevenZipFile(inner, "w") as z:
        z.writeall(src / "data", arcname="data")
    outer = tmp_path / "archives" / "members.csv.7z.zip"
    outer.parent.mkdir()
    with zipfile.ZipFile(outer, "w") as z:
        z.write(inner, arcname=inner.name)
    return outer


def test_extracts_nested_archive_flat(tmp_path: Path) -> None:
    archive = _nested_archive(tmp_path)
    out = tmp_path / "raw" / "csv"
    extract(archive, out)
    assert (out / "members.csv").read_text() == "msno,city\na,1\n"
    assert [p.name for p in out.iterdir()] == ["members.csv"]


def test_archive_untouched_and_existing_file_kept(tmp_path: Path) -> None:
    archive = _nested_archive(tmp_path)
    before = _digest(archive)
    out = tmp_path / "raw" / "csv"
    out.mkdir(parents=True)
    (out / "members.csv").write_text("already here\n")
    extract(archive, out)
    assert (out / "members.csv").read_text() == "already here\n"
    assert _digest(archive) == before


# --- ensure_archive: size checks, stall watchdog, retries -------------------

FAKE_KAGGLE = """
import sys, time
from pathlib import Path
dest, name, total, mode, counter = sys.argv[1:]
total = int(total)
counter = Path(counter)
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
out = Path(dest) / name
if mode == "stall_then_ok" and attempt == 1:
    out.write_bytes(b"x" * (total // 2))
    time.sleep(60)  # hangs like the stalled members_v3 download
elif mode == "short":
    out.write_bytes(b"x" * (total - 1))
else:
    out.write_bytes(b"x" * total)
"""


def _fake(tmp_path: Path, mode: str, total: int):
    script = tmp_path / "fake_kaggle.py"
    script.write_text(FAKE_KAGGLE)
    counter = tmp_path / "attempts.txt"

    def command(dest: Path) -> list[str]:
        return [sys.executable, str(script), str(dest), "f.7z", str(total), mode, str(counter)]

    return command, counter


FAST = {"stall_seconds": 1.0, "poll": 0.1, "backoff": 0.0}


def test_complete_archive_is_skipped(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    (archives / "f.7z").write_bytes(b"x" * 100)
    command, counter = _fake(tmp_path, "ok", 100)
    ensure_archive("f.7z", 100, archives, command, **FAST)
    assert not counter.exists()  # downloader never ran


def test_wrong_size_archive_raises_and_is_kept(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    (archives / "f.7z").write_bytes(b"x" * 40)
    command, _ = _fake(tmp_path, "ok", 100)
    with pytest.raises(RuntimeError, match="40 bytes but Kaggle reports 100"):
        ensure_archive("f.7z", 100, archives, command, **FAST)
    assert (archives / "f.7z").stat().st_size == 40


def test_stalled_download_is_killed_and_retried(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    command, counter = _fake(tmp_path, "stall_then_ok", 100)
    start = time.monotonic()
    path = ensure_archive("f.7z", 100, archives, command, retries=2, **FAST)
    assert time.monotonic() - start < 20  # did not wait out the 60s hang
    assert counter.read_text() == "2"
    assert path.stat().st_size == 100
    assert not (archives / ".incoming" / "f.7z").exists()


def test_wrong_final_size_fails_without_touching_archives(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    command, counter = _fake(tmp_path, "short", 100)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        ensure_archive("f.7z", 100, archives, command, retries=2, **FAST)
    assert counter.read_text() == "3"
    assert not (archives / "f.7z").exists()
