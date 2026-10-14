"""reasons.parquet: the reasons for every explained customer, snapshot and horizon.

Built from the stored contributions, the features (fact columns and
tenure_days), the saved Platt parameters and the committed configs. No labels
are read: the caveats depend only on the snapshot and the customer half, which
comes from the customer id.

One chunk per horizon and snapshot. Every chunk is checked before it is
written (row count, listed contributions plus other factors equal d within
1e-12 recomputed from the written columns, no hidden group in a slot, no hidden
feature named in the text); the file is written under a temporary name and
moved into place only after every chunk has passed.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from retention.explain.config import STRENGTHS, ExplainConfig
from retention.explain.contributions import EXPECTED_ROWS, CheckFailed
from retention.explain.reasons import (
    HIDDEN_WORDS,
    PHRASES,
    Renderer,
    platt_probability,
    probability_text,
)
from retention.explain.selection import POSITIONS, select, shown_mask, strength_codes
from retention.models.data import load_features
from retention.models.frames import half_mask
from retention.models.protocol import Protocol
from retention.models.resources import peak_rss_gb

TOLERANCE = 1e-12
KEYS = ["msno", "snapshot_date", "horizon"]
CALIBRATION_MONTH = date(2016, 10, 1)
SETTINGS_ONLY_MONTH = date(2016, 11, 1)
GAP_MONTH = date(2017, 2, 1)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed_sha256(repo: Path, path: Path) -> str:
    """SHA-256 of a file exactly as committed at HEAD (bytes, no stripping)."""
    out = subprocess.run(
        ["git", "-C", str(repo), "show", f"HEAD:{path}"], capture_output=True, check=True
    )
    return hashlib.sha256(out.stdout).hexdigest()


def caveat_flags(msno: pd.Series, t: date, protocol: Protocol) -> dict[str, np.ndarray]:
    """The three caveat flags of every row of one snapshot."""
    n = len(msno)
    cal = half_mask(msno, "calibration", protocol) if t == CALIBRATION_MONTH else np.zeros(n, bool)
    settings = (np.ones(n, bool) & ~cal) if t == CALIBRATION_MONTH else np.zeros(n, bool)
    if t == SETTINGS_ONLY_MONTH:
        settings = np.ones(n, bool)
    return {
        "october_2016_calibration_half": cal,
        "months_used_for_settings": settings,
        "february_2017_gap": np.full(n, t == GAP_MONTH),
    }


def reason_rows(
    cfg: ExplainConfig,
    renderer: Renderer,
    part: pd.DataFrame,
    facts: pd.DataFrame,
    horizon: int,
    snapshot: date,
    platt: tuple[float, float],
    protocol: Protocol,
    phrases_sha256: str,
) -> pd.DataFrame:
    """The reasons of one horizon and snapshot, in the contributions part's row order."""
    names = cfg.group_names
    df = part.merge(facts, on="msno", how="left", validate="one_to_one")
    if len(df) != len(part):
        raise CheckFailed("facts do not match the contributions rows")
    b = float(df.platt_b.iloc[0])
    raw_minus_base = (df.raw_score - df.base_value).to_numpy()
    sel = select(
        cfg, df[[f"g_{n}" for n in names]].to_numpy(), raw_minus_base, b, shown_mask(cfg, df)
    )
    prob = platt_probability(df.raw_score.to_numpy(), *platt)
    flags = caveat_flags(df.msno, snapshot, protocol)
    fact_cols = sorted({c for cs in cfg.facts.values() for c in cs})
    records = df[fact_cols].astype(object).where(df[fact_cols].notna(), None).to_dict("records")
    keys = list(flags)
    texts, cancel = [], np.zeros(len(df), bool)
    for i, f in enumerate(records):
        caveats = tuple(k for k in keys if flags[k][i])
        out = renderer.render(sel, i, f, horizon, float(prob[i]), caveats)
        texts.append("\n".join(out.lines))
        cancel[i] = out.partly_cancel
    rows = np.arange(len(df))
    pos_s = np.where(sel.position != 0, strength_codes(np.abs(sel.d), cfg.cutoffs()), 0)

    def word(code):
        return [STRENGTHS[k - 1] if k > 0 else None for k in code]

    out = {
        "msno": df.msno.astype(str).to_numpy(),
        "snapshot_date": np.full(len(df), snapshot),
        "horizon": np.full(len(df), horizon, dtype=np.int16),
        "raw_score": df.raw_score.to_numpy(),
        "base_value": df.base_value.to_numpy(),
        "platt_a": np.full(len(df), platt[0]),
        "platt_b": np.full(len(df), b),
        "d": sel.d,
        "probability": prob,
        "probability_text": [probability_text(float(p)) for p in prob],
        "position": [POSITIONS[int(p) + 1] for p in sel.position],
        "position_strength": word(pos_s),
    }
    for kind, idx, strength in (
        ("raising", sel.raising, sel.raising_strength),
        ("lowering", sel.lowering, sel.lowering_strength),
    ):
        for j in range(idx.shape[1]):
            ok = idx[:, j] >= 0
            out[f"{kind}_{j + 1}"] = [
                names[k] if o else None for k, o in zip(idx[:, j], ok, strict=True)
            ]
            out[f"{kind}_{j + 1}_strength"] = word(strength[:, j])
            out[f"{kind}_{j + 1}_c"] = np.where(ok, sel.c[rows, np.maximum(idx[:, j], 0)], np.nan)
    out["other_direction"] = np.where(
        sel.other_strength == 0, "small", np.where(sel.other > 0, "raises", "lowers")
    )
    out["other_strength"] = word(sel.other_strength)
    out["other"] = sel.other
    out["partly_cancel"] = cancel
    out |= {f"caveat_{k}": v for k, v in flags.items()}
    out["text"] = texts
    out["phrases_sha256"] = np.full(len(df), phrases_sha256)
    return pd.DataFrame(out)


def check_rows(rows: pd.DataFrame, cfg: ExplainConfig) -> dict[str, Any]:
    """The per-row pass marks, recomputed from the written columns."""
    c_cols = [c for c in rows.columns if c.endswith("_c")]
    listed = rows[c_cols].fillna(0.0).sum(axis=1).to_numpy()
    closure = float(np.max(np.abs(listed + rows.other.to_numpy() - rows.d.to_numpy())))
    slot_cols = [c for c in rows.columns if re.fullmatch(r"(raising|lowering)_\d", c)]
    hidden_slots = int(rows[slot_cols].isin(list(cfg.hidden)).to_numpy().sum())
    hidden_cols = [c for h in cfg.hidden for c in cfg.groups[h]]
    pattern = re.compile("|".join([HIDDEN_WORDS, *map(re.escape, hidden_cols)]), re.IGNORECASE)
    hidden_text = int(rows.text.map(lambda t: bool(pattern.search(t))).sum())
    failures = []
    if not closure <= TOLERANCE:
        failures.append(f"listed + other - d up to {closure:.2e}")
    if hidden_slots:
        failures.append(f"{hidden_slots} slots hold a hidden group")
    if hidden_text:
        failures.append(f"{hidden_text} texts name something hidden")
    return {
        "max_closure": closure,
        "hidden_slots": hidden_slots,
        "hidden_text": hidden_text,
        "failures": failures,
    }


def run_reasons(
    protocol: Protocol,
    processed: Path,
    contrib_dir: Path,
    model_dir: Path,
    cfg: ExplainConfig,
    renderer: Renderer,
    out_path: Path,
    expected_rows: dict[date, int] = EXPECTED_ROWS,
    phrases_path: Path = PHRASES,
    metadata: dict[str, str] | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    phrases_sha = sha256_file(phrases_path)
    fact_cols = sorted({c for cs in cfg.facts.values() for c in cs} | {"tenure_days"})
    tmp = out_path.with_suffix(".parquet.tmp")
    writer = None
    report: dict[str, Any] = {"chunks": [], "phrases_sha256": phrases_sha}
    try:
        for h in cfg.horizons:
            meta = json.loads((model_dir / f"model_{h}d.json").read_text())
            platt = (float(meta["platt_a"]), float(meta["platt_b"]))
            for t in cfg.snapshots:
                part = pq.read_table(contrib_dir / f"h{h}_{t.isoformat()}.parquet").to_pandas()
                facts = load_features(processed / "features.parquet", [t], fact_cols)
                facts = facts.drop(columns="snapshot_date")
                rows = reason_rows(cfg, renderer, part, facts, h, t, platt, protocol, phrases_sha)
                chk = check_rows(rows, cfg)
                if t in expected_rows and len(rows) != expected_rows[t]:
                    chk["failures"].append(f"{len(rows)} rows, expected {expected_rows[t]}")
                if len(rows) != len(part):
                    chk["failures"].append("row count differs from the contributions part")
                chk |= {
                    "horizon": h,
                    "snapshot": t.isoformat(),
                    "rows": len(rows),
                    "partly_cancel_rows": int(rows.partly_cancel.sum()),
                    "peak_rss_gb": peak_rss_gb(),
                }
                if chk["peak_rss_gb"] > cfg.memory_ceiling_gb:
                    chk["failures"].append(f"peak RSS {chk['peak_rss_gb']} GB")
                report["chunks"].append(chk)
                progress(
                    f"{h} days {t}: {len(rows):,} rows, closure {chk['max_closure']:.1e}, "
                    f"peak {chk['peak_rss_gb']:.2f} GB"
                )
                if chk["failures"]:
                    raise CheckFailed(f"{h} days {t}: " + "; ".join(chk["failures"]))
                table = pa.Table.from_pandas(rows, preserve_index=False)
                if writer is None:
                    schema = table.schema.with_metadata(
                        {b"retention.reasons": json.dumps(metadata or {}).encode()}
                    )
                    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
                writer.write_table(table.cast(writer.schema))
        writer.close()
        writer = None
        tmp.replace(out_path)
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
    report["passed"] = True
    return report


def read_reasons(path: Path) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    return df.sort_values(KEYS, kind="stable").reset_index(drop=True)


def same_reasons(a: Path, b: Path) -> dict[str, Any]:
    x, y = read_reasons(a), read_reasons(b)
    identical = bool(list(x.columns) == list(y.columns) and x.equals(y))
    return {"rows": len(x), "columns": x.shape[1], "identical": identical}
