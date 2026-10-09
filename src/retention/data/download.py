"""Download KKBox competition files with the Kaggle CLI and extract them.

Downloads land in data/raw/archives/.incoming/ and move to data/raw/archives/
only once their size matches what Kaggle reports. Archives are never modified
or deleted. Extracted files land in data/raw/csv/. Existing files are skipped,
so re-running never overwrites raw data.
"""

from __future__ import annotations

import csv
import os
import shutil
import signal
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import py7zr

ARCHIVE_SUFFIXES = (".7z", ".zip")


@dataclass(frozen=True)
class RawFile:
    name: str  # extracted file name, e.g. "user_logs_v2.csv"
    group: str  # "v2", "v1" or "v1_logs"
    table: str | None = None  # Parquet table name; None for non-tabular files
    remote: str | None = None  # file name on Kaggle, if it differs from `name` + ".7z"
    stream: bool = False  # read straight from the archive, never unpacked to disk

    @property
    def remote_name(self) -> str:
        return self.remote or f"{self.name}.7z"


def parse_files(entries: list[dict[str, str]]) -> list[RawFile]:
    return [RawFile(**entry) for entry in entries]


def remote_sizes(competition: str) -> dict[str, int]:
    """File sizes in bytes as Kaggle reports them (`kaggle competitions files -v`)."""
    out = subprocess.run(
        [_kaggle(), "competitions", "files", "-c", competition, "-v"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lines = out.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("name,size"))
    return {row["name"]: int(row["size"]) for row in csv.DictReader(lines[header:])}


def _kaggle() -> str:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("kaggle CLI not found. Run `make setup` first.")
    return kaggle


def kaggle_command(competition: str, remote_name: str, dest: Path) -> list[str]:
    args = ["competitions", "download", "-c", competition, "-f", remote_name]
    return [_kaggle(), *args, "-p", str(dest), "-q"]


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _run_with_watchdog(cmd: list[str], watched: Path, stall_seconds: float, poll: float) -> bool:
    """Run cmd; kill its process group if `watched` stops growing. True if it exited cleanly."""
    proc = subprocess.Popen(cmd, start_new_session=True)
    last_size, last_change = -1, time.monotonic()
    while proc.poll() is None:
        time.sleep(poll)
        size = _size(watched)
        if size != last_size:
            last_size, last_change = size, time.monotonic()
        elif time.monotonic() - last_change > stall_seconds:
            print(f"no progress for {stall_seconds:.0f}s at {size} bytes, stopping download")
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return False
    return proc.returncode == 0


def ensure_archive(
    remote_name: str,
    expected_size: int,
    archive_dir: Path,
    command: Callable[[Path], list[str]],
    stall_seconds: float = 120,
    retries: int = 3,
    poll: float = 5,
    backoff: float = 10,
) -> Path:
    """Make sure archive_dir/remote_name exists at exactly the size Kaggle reports.

    Downloads go to archive_dir/.incoming/ and are moved into archive_dir only after
    the size matches, so a file in archive_dir is always complete. A partial file is
    kept between attempts so the Kaggle CLI can resume it.
    """
    final = archive_dir / remote_name
    if final.exists():
        if _size(final) == expected_size:
            print(f"ok, already downloaded at expected size: {remote_name}")
            return final
        raise RuntimeError(
            f"{final} is {_size(final)} bytes but Kaggle reports {expected_size}. "
            "Archives are never deleted automatically: check it and remove it by hand."
        )
    incoming = archive_dir / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    staged = incoming / remote_name
    marker = incoming / f"{remote_name}.kaggle-partial"
    for attempt in range(1, retries + 2):
        print(f"downloading {remote_name} ({expected_size} bytes), attempt {attempt}")
        clean_exit = _run_with_watchdog(command(incoming), staged, stall_seconds, poll)
        size = _size(staged)
        if size == expected_size:
            os.replace(staged, final)
            marker.unlink(missing_ok=True)
            return final
        if clean_exit or size > expected_size:
            # The CLI thinks it finished, or overshot: the partial cannot be trusted.
            print(f"wrong size {size}, expected {expected_size}; discarding the partial")
            staged.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
        if attempt <= retries:
            time.sleep(backoff * attempt)
    raise RuntimeError(f"{remote_name}: download failed after {retries + 1} attempts")


def _unpack(archive: Path, dest: Path) -> None:
    if archive.suffix == ".7z":
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=dest)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(path=dest)
    else:
        raise ValueError(f"not an archive: {archive}")


def extract(path: Path, out_dir: Path) -> list[Path]:
    """Extract an archive (nested archives too) into out_dir, flattening folders.

    Works in a temporary folder next to out_dir so the source archive is only read.
    Plain files are copied. Files already in out_dir are left alone.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix not in ARCHIVE_SUFFIXES:
        target = out_dir / path.name
        if not target.exists():
            shutil.copy2(path, target)
        return [target]

    written: list[Path] = []
    with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
        tmp_dir = Path(tmp)
        _unpack(path, tmp_dir)
        for member in sorted(p for p in tmp_dir.rglob("*") if p.is_file()):
            if member.name.startswith(("._", ".DS_Store")) or "__MACOSX" in member.parts:
                continue
            if member.suffix in ARCHIVE_SUFFIXES:
                written.extend(extract(member, out_dir))
                continue
            target = out_dir / member.name
            if target.exists():
                print(f"skip extract, already present: {target.name}")
            else:
                shutil.move(member, target)
            written.append(target)
    return written


def download_all(
    competition: str,
    files: list[RawFile],
    raw_dir: Path,
    skip_groups: set[str],
    stall_seconds: float = 120,
    retries: int = 3,
) -> None:
    archive_dir = raw_dir / "archives"
    csv_dir = raw_dir / "csv"
    sizes = remote_sizes(competition)
    for f in files:
        if f.group in skip_groups:
            print(f"skip group {f.group}: {f.name}")
            continue
        if f.remote_name not in sizes:
            raise RuntimeError(f"{f.remote_name} is not in the Kaggle file list")
        archive = ensure_archive(
            f.remote_name,
            sizes[f.remote_name],
            archive_dir,
            lambda dest, name=f.remote_name: kaggle_command(competition, name, dest),
            stall_seconds=stall_seconds,
            retries=retries,
        )
        if f.stream:
            print(f"not extracting {f.name}: streamed from the archive when sampling")
            continue
        if (csv_dir / f.name).exists():
            print(f"skip, already extracted: {f.name}")
            continue
        print(f"extracting {archive.name}")
        extract(archive, csv_dir)
        if not (csv_dir / f.name).exists():
            raise RuntimeError(f"{f.name} not found after extracting {f.remote_name}")
