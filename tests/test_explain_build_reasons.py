"""reasons.parquet: rows, closure, hidden features, caveats, text, determinism, no labels."""

from __future__ import annotations

import dataclasses
import json
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retention.explain.build_reasons import (
    CheckFailed,
    caveat_flags,
    check_rows,
    reason_rows,
    run_reasons,
    same_reasons,
    sha256_file,
)
from retention.explain.config import load_explain_config
from retention.explain.hand_check import expected_caveats
from retention.explain.model import load_settings
from retention.explain.reasons import PHRASES, Renderer, platt_probability
from retention.explain.selection import select, shown_mask
from retention.models import data as data_module
from retention.models.frames import half_mask
from retention.models.protocol import load_protocol

P = load_protocol()
CFG = load_explain_config(load_settings().columns)
R = Renderer.from_files(CFG)
NAMES = CFG.group_names
PLATT = (0.024395167821727723, 0.9988291492427049)
SHA = sha256_file(PHRASES)
OCT, NOV, JAN, FEB = date(2016, 10, 1), date(2016, 11, 1), date(2017, 1, 1), date(2017, 2, 1)


def make_part(n: int, seed: int, t: date, horizon: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.normal(scale=0.45, size=(n, len(NAMES)))
    g[: n // 4, NAMES.index("hidden")] = 3.0  # hidden is often the largest push
    base = -6.2
    raw = base + g.sum(axis=1)
    out = pd.DataFrame(
        {
            "msno": [f"c{seed}_{i:03d}" for i in range(n)],
            "snapshot_date": [t] * n,
            "horizon": horizon,
            "raw_score": raw,
            "base_value": base,
            "platt_b": PLATT[1],
            "d": PLATT[1] * (raw - base),
        }
    )
    for k, name in enumerate(NAMES):
        out[f"g_{name}"] = g[:, k]
    return out


def make_facts(msno: pd.Series, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    n = len(msno)

    def maybe(values, share):
        v = pd.array(values)
        return v.where(rng.random(n) >= share, pd.NA) if hasattr(v, "where") else v

    act = rng.integers(0, 29, n)
    return pd.DataFrame(
        {
            "msno": msno.to_numpy(),
            "days_until_expiry": pd.array(rng.integers(0, 31, n), "Int32"),
            "days_since_last_payment": pd.array(
                np.where(rng.random(n) < 0.1, None, rng.integers(1, 200, n)), "Int32"
            ),
            "active_days_28": pd.array(act, "Int32"),
            "secs_per_day_28": np.where(act == 0, 0.0, rng.uniform(5, 5000, n)),
            "ratio_secs_7_vs_base": pd.array(
                np.where(rng.random(n) < 0.3, None, rng.uniform(0, 3, n)), "Float64"
            ),
            "ratio_active_7_vs_base": pd.array(
                np.where(rng.random(n) < 0.3, None, rng.uniform(0, 3, n)), "Float64"
            ),
            "days_since_last_listen": pd.array(
                np.where(act == 0, None, rng.integers(1, 40, n)), "Int32"
            ),
            "has_any_log_before_T": pd.array((rng.random(n) < 0.9).astype(int), "Int8"),
            "first_log_in_window": pd.array((rng.random(n) < 0.05).astype(int), "Int8"),
            "complete_share_28": pd.array(
                np.where(act == 0, None, rng.uniform(0, 1, n)), "Float64"
            ),
            "last_row_kind": pd.array(
                rng.choice(["paid", "trial", "promo", "discount100"], n, p=[0.9, 0.04, 0.03, 0.03]),
                "string",
            ),
            "last_amount_paid": pd.array(rng.choice([0, 99, 149, 180], n), "Int32"),
            "last_plan_days": pd.array(rng.choice([0, 30, 90], n), "Int32"),
            "last_is_auto_renew": pd.array(rng.integers(0, 2, n), "Int8"),
            "last_row_is_cancel": pd.array((rng.random(n) < 0.05).astype(int), "Int8"),
            "tenure_days": pd.array(
                np.where(rng.random(n) < 0.15, None, rng.integers(5, 3000, n)), "Int32"
            ),
        }
    )


@pytest.fixture(scope="module")
def built():
    part = make_part(60, 1, OCT)
    facts = make_facts(part.msno, 1)
    return part, facts, reason_rows(CFG, R, part, facts, 7, OCT, PLATT, P, SHA)


def test_rows_and_order_follow_the_contributions(built):
    part, _, rows = built
    assert len(rows) == len(part) and rows.msno.tolist() == part.msno.tolist()
    assert set(rows.horizon) == {7} and set(rows.snapshot_date) == {OCT}


def test_listed_plus_other_equals_d_and_nothing_hidden(built):
    _, _, rows = built
    chk = check_rows(rows, CFG)
    assert chk["failures"] == [] and chk["max_closure"] <= 1e-12
    slots = rows[[c for c in rows.columns if c[-2:] in ("_1", "_2", "_3")]]
    assert not slots.isin(["hidden"]).any().any()
    assert set(slots.stack().dropna()) <= set(NAMES) - {"hidden"}


def test_check_rows_catches_tampering(built):
    _, _, rows = built
    bad = rows.copy()
    bad.loc[0, "other"] += 1e-9
    assert any("listed + other" in f for f in check_rows(bad, CFG)["failures"])
    bad = rows.copy()
    bad.loc[0, "raising_1"] = "hidden"
    assert any("hidden group" in f for f in check_rows(bad, CFG)["failures"])
    bad = rows.copy()
    bad.loc[0, "text"] += " via registered channel"
    assert any("name something hidden" in f for f in check_rows(bad, CFG)["failures"])


def test_slots_are_ordered_by_size(built):
    _, _, rows = built
    for kind, k in (("raising", 3), ("lowering", 2)):
        c = rows[[f"{kind}_{j}_c" for j in range(1, k + 1)]].abs().to_numpy()
        for a, b in zip(c[:, :-1].T, c[:, 1:].T, strict=True):
            ok = np.isnan(b) | (a >= b)
            assert ok.all()


def test_text_is_the_renderer_output(built):
    part, facts, rows = built
    df = part.merge(facts, on="msno")
    sel = select(
        CFG,
        df[[f"g_{n}" for n in NAMES]].to_numpy(),
        (df.raw_score - df.base_value).to_numpy(),
        PLATT[1],
        shown_mask(CFG, df),
    )
    prob = platt_probability(df.raw_score.to_numpy(), *PLATT)
    for i in (0, 7, 33):
        f = {c: (None if pd.isna(df.loc[i, c]) else df.loc[i, c]) for c in facts.columns}
        cav = expected_caveats(OCT, df.msno[i], P)
        want = "\n".join(R.render(sel, i, f, 7, float(prob[i]), cav).lines)
        assert rows.text[i] == want
    assert rows.probability.tolist() == prob.tolist()


def test_caveat_flags_match_the_hand_check_rule():
    msno = pd.Series([f"x{i}" for i in range(40)])
    for t in (OCT, NOV, JAN, FEB):
        flags = caveat_flags(msno, t, P)
        for i, m in enumerate(msno):
            got = tuple(k for k, v in flags.items() if v[i])
            assert got == expected_caveats(t, m, P)
    cal = caveat_flags(msno, OCT, P)["october_2016_calibration_half"]
    assert cal.tolist() == half_mask(msno, "calibration", P).tolist()


def test_same_input_same_output_and_phrases_sha(built):
    part, facts, rows = built
    again = reason_rows(CFG, R, part, facts, 7, OCT, PLATT, P, SHA)
    assert rows.equals(again)
    assert set(rows.phrases_sha256) == {SHA}


# --- the whole run on files ----------------------------------------------------------


@pytest.fixture()
def files(tmp_path):
    processed, contrib, model = tmp_path / "processed", tmp_path / "contrib", tmp_path / "model"
    for d in (processed, contrib, model):
        d.mkdir()
    snaps = (OCT, NOV)
    feats = []
    for k, t in enumerate(snaps):
        for h in (7, 30):
            part = make_part(25, 10 + k, t, h)
            pq.write_table(
                pa.Table.from_pandas(part, preserve_index=False),
                contrib / f"h{h}_{t.isoformat()}.parquet",
            )
        f = make_facts(make_part(25, 10 + k, t).msno, 10 + k).assign(snapshot_date=t)
        feats.append(f)
    pq.write_table(
        pa.Table.from_pandas(pd.concat(feats, ignore_index=True), preserve_index=False),
        processed / "features.parquet",
    )
    for h in (7, 30):
        (model / f"model_{h}d.json").write_text(
            json.dumps({"platt_a": PLATT[0], "platt_b": PLATT[1]})
        )
    cfg = dataclasses.replace(CFG, snapshots=snaps)
    return processed, contrib, model, cfg, {OCT: 25, NOV: 25}


def test_run_writes_reads_no_labels_and_repeats_exactly(files, tmp_path, monkeypatch):
    processed, contrib, model, cfg, counts = files

    def refuse(*a, **k):
        raise PermissionError("no labels")

    monkeypatch.setattr(data_module, "load_labels", refuse)
    out = tmp_path / "reasons.parquet"
    report = run_reasons(P, processed, contrib, model, cfg, R, out, counts, progress=lambda m: None)
    assert report["passed"] and len(report["chunks"]) == 4
    assert pq.read_metadata(out).num_rows == 100
    assert not list(tmp_path.glob("*.tmp"))
    out2 = tmp_path / "reasons2.parquet"
    run_reasons(P, processed, contrib, model, cfg, R, out2, counts, progress=lambda m: None)
    assert same_reasons(out, out2)["identical"]


def test_a_failed_chunk_writes_nothing(files, tmp_path):
    processed, contrib, model, cfg, _ = files
    out = tmp_path / "reasons.parquet"
    with pytest.raises(CheckFailed, match="rows, expected"):
        run_reasons(
            P, processed, contrib, model, cfg, R, out, {OCT: 25, NOV: 24}, progress=lambda m: None
        )
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))
