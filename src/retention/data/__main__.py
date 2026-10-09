"""Command line entry: `python -m retention.data {download,sample}`."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from retention.data.download import download_all, parse_files
from retention.data.sample import sample_all, sample_dir
from retention.data.stream import compare_logs, stream_sample


def _fraction(args: argparse.Namespace, data_cfg: dict) -> float:
    env_fraction = os.environ.get("SAMPLE_FRACTION") or None
    return args.fraction or float(env_fraction or data_cfg["sample_fraction"])


def verify_stream(raw_dir: Path, interim_dir: Path, fraction: float) -> None:
    """Stream user_logs_v2 from its archive and compare with the extracted-CSV Parquet."""
    archive = raw_dir / "archives" / "user_logs_v2.csv.7z"
    extracted = sample_dir(interim_dir, fraction) / "user_logs_v2.parquet"
    out = interim_dir / "_verify_stream" / "user_logs_v2.parquet"
    stats = stream_sample(archive, "user_logs_v2.csv", out, fraction)
    result = compare_logs(out, extracted)
    mb = 1e6
    print(f"streamed {stats.source_rows} rows, kept {stats.kept_rows}, in {stats.seconds:.1f}s")
    print(f"dates {stats.date_min} to {stats.date_max}")
    print(
        f"throughput: {archive.stat().st_size / mb / stats.seconds:.1f} MB/s compressed, "
        f"{stats.uncompressed_bytes / mb / stats.seconds:.1f} MB/s uncompressed"
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    out.unlink()
    out.parent.rmdir()
    ok = (
        result["rows_streamed"] == result["rows_extracted"]
        and result["customers_only_in_one"] == 0
        and result["num_100_mismatches"] == 0
        and result["total_secs_max_rel_diff"] is not None
        and result["total_secs_max_rel_diff"] < 1e-9
    )
    print("MATCH" if ok else "MISMATCH")
    if not ok:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m retention.data")
    parser.add_argument("command", choices=["download", "sample", "verify-stream"])
    parser.add_argument("--config", type=Path, default=Path("configs/kkbox.yaml"))
    parser.add_argument("--skip-group", action="append", default=[], help="e.g. v1_logs")
    parser.add_argument("--fraction", type=float, default=None)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    files = parse_files(data_cfg["files"])

    if args.command == "download":
        dl = data_cfg.get("download", {})
        download_all(
            config["competition"],
            files,
            raw_dir,
            set(args.skip_group),
            stall_seconds=dl.get("stall_seconds", 120),
            retries=dl.get("retries", 3),
        )
    elif args.command == "verify-stream":
        verify_stream(raw_dir, Path(data_cfg["interim_dir"]), _fraction(args, data_cfg))
    else:
        fraction = _fraction(args, data_cfg)
        wanted = [f for f in files if f.table and f.group not in args.skip_group]
        tables = {f.table: f.name for f in wanted if not f.stream}
        streamed = {
            f.table: (raw_dir / "archives" / f.remote_name, f.name) for f in wanted if f.stream
        }
        interim = Path(data_cfg["interim_dir"])
        out_dir = sample_all(raw_dir / "csv", tables, interim, fraction, streamed)
        print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
