"""SHAP contributions of the saved 7-day and 30-day boosters, one snapshot at a time.

LightGBM's pred_contrib (path-dependent TreeSHAP): one value per model column
plus the base value, adding up to the raw score (log-odds). Each chunk is one
snapshot and one horizon; the checks run on every chunk and any failure stops
the run before any output part is moved into place:

- booster files: their SHA-256 equals the saved one and the registered one;
- row counts: the known P2 main counts per snapshot;
- additivity: |sum of contributions + base value - raw score| <= the limit;
- base value: constant within the chunk and equal (within the limit) to the
  refit's mean raw score over its fit rows;
- tie to the reproduction: on 2016-11-01 the loaded booster's raw-probability
  summary equals the rehearsal's logged one, every field;
- peak memory below the ceiling after every chunk.

Rows come from features.parquet plus in_exp30 and segment (known at T); no
outcome column is read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from retention.explain.config import ExplainConfig
from retention.explain.model import (
    Settings,
    compare_summary,
    logged_summary,
    p2_main_rows,
)
from retention.explain.selection import Selection, select, shown_mask
from retention.models.classifier import Classifier, score_summary
from retention.models.protocol import Protocol
from retention.models.resources import peak_rss_gb

# Registered booster hashes (refit at 6b49305, saved at db585dc).
EXPECTED_SHA256 = {
    7: "e1c985c98c87280cf09bf451515e660f2f4b3fbae722cae97c93a02ed2b0a26d",
    30: "b967381d50a3e1f392b5db5281e2326a22799dde79dc8cdce66adb01b9fcf35d",
}
# P2 main rows per explained snapshot in the 20% sample.
EXPECTED_ROWS = {
    date(2016, 10, 1): 144_996,
    date(2016, 11, 1): 169_673,
    date(2017, 1, 1): 150_672,
    date(2017, 2, 1): 180_412,
}
TIE_SNAPSHOT = date(2016, 11, 1)
KEYS = ["msno", "snapshot_date", "horizon"]


class CheckFailed(RuntimeError):
    """A pass mark failed; the message says which and by how much."""


# --- the saved model -----------------------------------------------------------------


@dataclass
class SavedModel:
    horizon: int
    model: Classifier
    platt_a: float
    platt_b: float
    sha256: str


def load_saved_model(
    model_dir: Path,
    horizon: int,
    settings: Settings,
    protocol: Protocol,
    expected_sha: str,
) -> SavedModel:
    """The saved booster and its Platt parameters; refuses a file whose hash differs."""
    text = (model_dir / f"booster_{horizon}d.txt").read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    meta = json.loads((model_dir / f"model_{horizon}d.json").read_text())
    if not sha == meta["booster_sha256"] == expected_sha:
        raise CheckFailed(
            f"{horizon} days: booster file SHA-256 {sha} does not match the saved "
            f"{meta['booster_sha256']} and the registered {expected_sha}"
        )
    if meta["columns"] != settings.columns:
        raise CheckFailed(f"{horizon} days: saved columns differ from the frozen feature set")
    model = Classifier(
        meta["columns"],
        settings.categorical,
        settings.candidates[horizon],
        settings.fixed,
        protocol.threads,
        protocol.seed,
    )
    model.cats = [c for c in settings.categorical if c in meta["columns"]]
    model.levels = {c: meta["categorical_levels"][c] for c in model.cats}
    model.booster = lgb.Booster(model_str=text)
    return SavedModel(horizon, model, float(meta["platt_a"]), float(meta["platt_b"]), sha)


# --- one chunk ---------------------------------------------------------------------------


def chunk_contributions(model: Classifier, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Raw scores (log-odds) and contributions (rows x (columns + 1), base value last)."""
    x, _ = model._frame(rows)
    raw = model.booster.predict(x, raw_score=True)
    contrib = model.booster.predict(x, pred_contrib=True)
    return np.asarray(raw, dtype=float), np.asarray(contrib, dtype=float)


def check_chunk(
    raw: np.ndarray,
    contrib: np.ndarray,
    base_reference: float,
    max_gap: float,
    max_base_error: float,
) -> dict[str, Any]:
    """Additivity and base-value checks of one chunk; 'failures' lists what failed."""
    base = contrib[:, -1]
    gap = float(np.max(np.abs(contrib.sum(axis=1) - raw))) if len(raw) else 0.0
    constant = bool(len(base) == 0 or np.all(base == base[0]))
    base_value = float(base[0]) if len(base) else float("nan")
    diff = abs(base_value - base_reference)
    failures = []
    if not gap <= max_gap:
        failures.append(f"additivity gap {gap:.3e} above {max_gap:.0e}")
    if not constant:
        failures.append(f"base value not constant (range {float(np.ptp(base)):.3e})")
    if not diff <= max_base_error:
        failures.append(f"base value differs from the refit's by {diff:.3e}")
    return {
        "max_additivity_gap": gap,
        "base_value": base_value,
        "base_value_constant": constant,
        "base_value_difference_from_refit": diff,
        "failures": failures,
    }


def group_sums(contrib: np.ndarray, columns: list[str], groups: dict[str, list[str]]) -> np.ndarray:
    """rows x groups: each group's summed contribution (group order of `groups`)."""
    pos = {c: k for k, c in enumerate(columns)}
    out = np.empty((contrib.shape[0], len(groups)))
    for j, cols in enumerate(groups.values()):
        out[:, j] = contrib[:, [pos[c] for c in cols]].sum(axis=1)
    return out


def chunk_table(
    rows: pd.DataFrame,
    horizon: int,
    raw: np.ndarray,
    contrib: np.ndarray,
    gsum: np.ndarray,
    columns: list[str],
    groups: list[str],
    sel: Selection,
    platt_b: float,
) -> pa.Table:
    data: dict[str, Any] = {
        "msno": rows.msno.astype(str).to_numpy(),
        "snapshot_date": rows.snapshot_date.dt.date.to_numpy(),
        "horizon": np.full(len(rows), horizon, dtype=np.int16),
        "raw_score": raw,
        "base_value": contrib[:, -1],
        "platt_b": np.full(len(rows), platt_b),
        "d": sel.d,
    }
    data |= {f"phi_{c}": contrib[:, k] for k, c in enumerate(columns)}
    data |= {f"g_{g}": gsum[:, j] for j, g in enumerate(groups)}
    return pa.table(data)


# --- the run -------------------------------------------------------------------------------


@dataclass
class ChunkResult:
    horizon: int
    snapshot: date
    rows: pd.DataFrame  # fact columns and keys, for the descriptive tables
    selection: Selection


def run_contributions(
    protocol: Protocol,
    processed: Path,
    model_dir: Path,
    refit_report: dict[str, Any],
    out_dir: Path,
    cfg: ExplainConfig,
    settings: Settings,
    progress: Callable[[str], None] = print,
    expected_sha: dict[int, str] = EXPECTED_SHA256,
    expected_rows: dict[date, int] = EXPECTED_ROWS,
    rehearsal: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[ChunkResult]]:
    """Every chunk with its checks; parts are written as temporary files and moved into
    place only when every chunk has passed. Returns the report and the chunk results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_parts: list[tuple[Path, Path]] = []
    report: dict[str, Any] = {"chunks": [], "tie_to_reproduction": []}
    results: list[ChunkResult] = []
    # A column can be both a fact and the "shown only when known" column (tenure_days):
    # keep each column once.
    kept_cols = sorted(
        {c for cols in cfg.facts.values() for c in cols} | set(cfg.shown_only_when_known.values())
    )
    try:
        for h in cfg.horizons:
            saved = load_saved_model(model_dir, h, settings, protocol, expected_sha[h])
            ref = refit_report["horizons"][str(h)]
            if ref["booster_sha256"] != saved.sha256:
                raise CheckFailed(f"{h} days: refit report hash differs from the booster file")
            base_ref = float(ref["base_value_mean_raw_score"])
            for t in cfg.snapshots:
                rows = p2_main_rows(processed, t, settings.columns, protocol)
                if t in expected_rows and len(rows) != expected_rows[t]:
                    raise CheckFailed(f"{t}: {len(rows)} P2 main rows, expected {expected_rows[t]}")
                raw, contrib = chunk_contributions(saved.model, rows)
                chk = check_chunk(
                    raw, contrib, base_ref, cfg.max_additivity_error, cfg.max_base_value_error
                )
                if t == TIE_SNAPSHOT and rehearsal is not None:
                    tie = tie_to_reproduction(saved.model, rows, h, cfg, rehearsal)
                    report["tie_to_reproduction"].append(tie)
                    if not tie["equal"]:
                        chk["failures"].append(f"tie to the reproduction: {tie['differs']}")
                gsum = group_sums(contrib, settings.columns, cfg.groups)
                sel = select(cfg, gsum, raw - contrib[:, -1], saved.platt_b, shown_mask(cfg, rows))
                closure = float(np.max(np.abs(sel.listed_sum() + sel.other - sel.d)))
                peak = peak_rss_gb()
                chk |= {
                    "horizon": h,
                    "snapshot": t.isoformat(),
                    "rows": len(rows),
                    "max_listed_plus_other_minus_d": closure,
                    "peak_rss_gb": peak,
                }
                if peak > cfg.memory_ceiling_gb:
                    chk["failures"].append(f"peak RSS {peak} GB above {cfg.memory_ceiling_gb} GB")
                report["chunks"].append(chk)
                progress(
                    f"{h} days {t}: {len(rows):,} rows, gap {chk['max_additivity_gap']:.2e}, "
                    f"base {chk['base_value']:.15g}, peak {peak:.2f} GB"
                )
                if chk["failures"]:
                    raise CheckFailed(f"{h} days {t}: " + "; ".join(chk["failures"]))
                final = out_dir / f"h{h}_{t.isoformat()}.parquet"
                tmp = final.with_suffix(".parquet.tmp")
                table = chunk_table(
                    rows,
                    h,
                    raw,
                    contrib,
                    gsum,
                    settings.columns,
                    cfg.group_names,
                    sel,
                    saved.platt_b,
                )
                pq.write_table(table, tmp, compression="zstd")
                tmp_parts.append((tmp, final))
                keep = rows[["msno", "snapshot_date", *kept_cols]].copy()
                results.append(ChunkResult(h, t, keep, sel))
                del raw, contrib, gsum, table
    except BaseException:
        for tmp, _ in tmp_parts:
            tmp.unlink(missing_ok=True)
        raise
    for tmp, final in tmp_parts:
        tmp.replace(final)
    report["passed"] = True
    return report, results


def tie_to_reproduction(
    model: Classifier,
    rows: pd.DataFrame,
    horizon: int,
    cfg: ExplainConfig,
    rehearsal: dict[str, Any],
) -> dict[str, Any]:
    """The loaded booster's raw-probability summary on the rows against the logged one."""
    cell = next(c for c in cfg.reproduction_cells if c[0] == horizon and c[1] == TIE_SNAPSHOT)
    path = cell[2][:-1] if cell[2][-1] == "score_summary_raw" else cell[2]
    p, _ = model.predict(rows)
    table = compare_summary(logged_summary(rehearsal, path), score_summary(p))
    bad = [r for r in table if not r["equal"]]
    largest = 0.0
    for r in bad:
        try:
            largest = max(largest, abs(float(r["logged"]) - float(r["refit"])))
        except (TypeError, ValueError):
            largest = float("inf")
    return {
        "horizon": horizon,
        "snapshot": TIE_SNAPSHOT.isoformat(),
        "fields": len(table),
        "equal": not bad,
        "differs": [r["field"] for r in bad],
        "largest_difference": largest,
    }


# --- comparing two runs ----------------------------------------------------------------------


def read_outputs(out_dir: Path) -> pd.DataFrame:
    parts = sorted(out_dir.glob("h*_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no contribution parts in {out_dir}")
    df = pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)
    return df.sort_values(KEYS, kind="stable").reset_index(drop=True)


def same_outputs(a: Path, b: Path) -> dict[str, Any]:
    """Whether two runs' outputs are identical after sorting by key (values compared exactly)."""
    x, y = read_outputs(a), read_outputs(b)
    same_shape = x.shape == y.shape and list(x.columns) == list(y.columns)
    identical = bool(same_shape and x.equals(y))
    return {"rows": len(x), "columns": x.shape[1], "identical": identical}
