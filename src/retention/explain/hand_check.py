"""The hand check: 20 customer-snapshot rows, rendered, with their raw rows and automatic checks.

The sample follows configs/explain.yaml (hand_check): 16 rows from the behaviour
strata of 2016-10 and 2016-11 (assigned in order, first match wins) and 4 rows
from 2017 (2 from 2017-01, 2 from 2017-02), drawn once with the seed there.
Every row is rendered at 7 and 30 days.

Each fact is recomputed independently from the raw tables (clean transactions,
daily listening logs, the member row, and the known-at-T expiry from the
labels), in pandas, not with the feature code. A row passes when, at both
horizons:
  1. every shown fact equals its raw value, its text equals the text rendered
     from the raw value, and every "none / missing" claim holds in the raw rows;
  2. nothing hidden is named and no hidden group is listed;
  3. listed contributions plus other factors equal d within 1e-12;
  4. every label matches the stored contributions (reason strength and
     direction, position, other factors);
  5. exactly the caveats that apply are present.
Features, stored contributions and the saved model only; no outcome column.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from retention.explain.config import STRENGTHS, ExplainConfig
from retention.explain.reasons import (
    HIDDEN_WORDS,
    Renderer,
    platt_probability,
    render_fact,
)
from retention.explain.selection import Selection, select, shown_mask, strength_codes
from retention.features.spec import BASELINE_MIN_ACTIVE_DAYS, SECS_PER_DAY_MAX, SECS_SENTINEL_ABS
from retention.labels.timeline import order_rows
from retention.models.data import load_features, load_labels
from retention.models.frames import half_mask
from retention.models.protocol import Protocol

LUNAR_NEW_YEAR = date(2017, 1, 28)
EXPLAIN_CONFIG = Path("configs/explain.yaml")
TOLERANCE = 1e-12
NOT_NUMBER = {
    "has_any_log_before_T",
    "first_log_in_window",
    "last_is_auto_renew",
    "last_row_is_cancel",
    "last_row_kind",
}
# Kept rows: every fact column, plus the stratum columns.
STRATUM_COLUMNS = [
    "last_row_kind",
    "last_is_auto_renew",
    "last_row_is_cancel",
    "has_any_log_before_T",
    "tenure_days",
]
LAST_ROW_ORDER = [
    "d",
    "expiry",
    "actual_amount_paid",
    "plan_list_price",
    "payment_plan_days",
    "payment_method_id",
    "is_auto_renew",
    "row_kind",
]


# --- the draw -----------------------------------------------------------------------------------


def load_hand_check_config(path: Path = EXPLAIN_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())["hand_check"]


def window_has(t: date, day: date, days: int = 28) -> bool:
    """Whether `day` falls in the window T - days to T - 1."""
    return t - timedelta(days=days) <= day <= t - timedelta(days=1)


def assign_strata(rows: pd.DataFrame, strata: list[dict[str, Any]], close_cut: float) -> pd.Series:
    """First matching stratum name per row. Needs columns last_row_kind, never_seen,
    last_is_auto_renew, top_decile_30, last_row_is_cancel, d_30, has_any_log_before_T."""
    rules = {
        "zero_paid_last_row": rows.last_row_kind.isin(["trial", "promo", "discount100"]),
        "never_seen": rows.never_seen & (rows.last_row_kind == "paid"),
        "auto_renew_high_risk": (rows.last_is_auto_renew == 1) & rows.top_decile_30,
        "manual_payer": rows.last_is_auto_renew == 0,
        "last_row_cancel": rows.last_row_is_cancel == 1,
        "close_to_average": rows.d_30.abs() < close_cut,
        "no_listening_before_t": rows.has_any_log_before_T == 0,
        "any_other": pd.Series(True, index=rows.index),
    }
    out = pd.Series(pd.NA, index=rows.index, dtype="object")
    for s in strata:
        hit = rules[s["name"]].fillna(False).astype(bool) & out.isna()
        out[hit] = s["name"]
    return out


def draw(rows: pd.DataFrame, cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """The sample: behaviour strata in order, then the month rows; random within each."""
    rng = np.random.default_rng(seed)
    rows = rows.sort_values(["snapshot_date", "msno"], kind="stable").reset_index(drop=True)
    behaviour = rows.snapshot_date.dt.date.isin(
        [date.fromisoformat(str(t)) for t in cfg["behaviour_strata_snapshots"]]
    )
    picked = []
    for s in cfg["strata"]:
        idx = np.flatnonzero(behaviour & (rows.stratum == s["name"]))
        take = np.sort(rng.choice(idx, size=s["n"], replace=False))
        picked += [(i, s["name"]) for i in take]
    for m in cfg["month_rows"]:
        t = date.fromisoformat(str(m["snapshot"]))
        idx = np.flatnonzero(rows.snapshot_date.dt.date == t)
        take = np.sort(rng.choice(idx, size=m["n"], replace=False))
        picked += [(i, f"month {t}") for i in take]
    out = rows.iloc[[i for i, _ in picked]].copy()
    out["sample_stratum"] = [s for _, s in picked]
    return out.reset_index(drop=True)


# --- raw values --------------------------------------------------------------------------------


@dataclass
class RawCustomer:
    transactions: pd.DataFrame  # rows dated before T
    daily: pd.DataFrame  # one row per listening day, all history before T
    member: dict[str, Any] | None
    expiry_date: date | None  # known at T, from the labels


def daily_listening(logs: pd.DataFrame) -> pd.DataFrame:
    """One row per day: cleaned seconds (sentinels as 0, capped per row), plays, full plays."""
    secs = logs.total_secs.astype(float)
    bad = (secs < 0) | (secs.abs() >= SECS_SENTINEL_ABS)
    cleaned = np.where(bad, 0.0, np.round(np.minimum(secs, SECS_PER_DAY_MAX) * 1000) / 1000)
    plays = logs[["num_25", "num_50", "num_75", "num_985", "num_100"]].sum(axis=1)
    df = pd.DataFrame({"day": logs.day, "secs": cleaned, "plays": plays, "full": logs.num_100})
    return df.groupby("day", as_index=False).sum().sort_values("day")


def raw_facts(t: date, raw: RawCustomer) -> dict[str, Any]:
    """Every fact column recomputed from the raw rows at snapshot T."""
    out: dict[str, Any] = {}
    out["days_until_expiry"] = (raw.expiry_date - t).days if raw.expiry_date else None
    tx = raw.transactions
    nc = tx[tx.is_cancel == 0]
    if len(nc):
        last = nc.sort_values(LAST_ROW_ORDER, ascending=False, na_position="last").iloc[0]
        out["last_row_kind"] = last.row_kind
        out["last_amount_paid"] = int(last.actual_amount_paid)
        out["last_plan_days"] = int(last.payment_plan_days)
        out["last_is_auto_renew"] = int(last.is_auto_renew)
    else:
        for c in ("last_row_kind", "last_amount_paid", "last_plan_days", "last_is_auto_renew"):
            out[c] = None
    paid = nc[nc.actual_amount_paid > 0]
    out["days_since_last_payment"] = (t - paid.d.max()).days if len(paid) else None
    out["last_row_is_cancel"] = int(tx.sort_values("seq").iloc[-1].is_cancel) if len(tx) else None

    daily = raw.daily
    has_log = len(daily) > 0
    off = np.array([(t - d).days for d in daily.day]) if has_log else np.array([], int)
    in28, last7, base21 = (
        (off >= 1) & (off <= 28),
        (off >= 1) & (off <= 7),
        (off >= 8) & (off <= 28),
    )
    out["has_any_log_before_T"] = int(has_log)
    out["days_since_last_listen"] = int(off.min()) if has_log else None
    out["first_log_in_window"] = int(has_log and off.max() <= 28)
    out["active_days_28"] = int(in28.sum())
    secs28, secs7, secs21 = (float(daily.secs[m].sum()) for m in (in28, last7, base21))
    out["secs_per_day_28"] = secs28 / 28
    a7, a21 = int(last7.sum()), int(base21.sum())
    enough = a21 >= BASELINE_MIN_ACTIVE_DAYS
    out["ratio_secs_7_vs_base"] = (secs7 / 7) / (secs21 / 21) if enough and secs21 > 0 else None
    out["ratio_active_7_vs_base"] = (a7 / 7) / (a21 / 21) if enough else None
    plays28 = int(daily.plays[in28].sum())
    out["complete_share_28"] = int(daily.full[in28].sum()) / plays28 if plays28 else None
    reg = raw.member.get("registration_init_time") if raw.member else None
    out["tenure_days"] = (t - reg).days if has_log and reg is not None and reg < t else None
    out["_active_days_base21"] = a21
    out["_plays_28"] = plays28
    return out


def claim_holds(col: str, text: str | None, spec: Any, raw: dict[str, Any]) -> tuple[bool, str]:
    """Whether a 'none / missing' text states something true about the raw rows."""
    if not isinstance(spec, dict) or text is None:
        return True, ""
    if text == spec.get("when_missing"):
        if col == "ratio_secs_7_vs_base":
            ok = raw["_active_days_base21"] == 0
            return ok, f"listening days in the three weeks before: {raw['_active_days_base21']}"
        if col == "days_since_last_payment":
            return raw[col] is None, "no paid row before T" if raw[
                col
            ] is None else "a paid row exists"
        if col == "complete_share_28":
            return raw["_plays_28"] == 0, f"plays in 28 days: {raw['_plays_28']}"
    return True, ""


def same(a: Any, b: Any) -> bool:
    if a is None or (isinstance(a, float) and math.isnan(a)) or a is pd.NA:
        return b is None or (isinstance(b, float) and math.isnan(b)) or b is pd.NA
    if b is None or b is pd.NA or (isinstance(b, float) and math.isnan(b)):
        return False
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))


# --- one rendered row with its checks ------------------------------------------------------------


@dataclass
class HorizonCheck:
    horizon: int
    lines: list[str]
    facts: list[dict[str, Any]]
    contributions: dict[str, float]
    d: float
    other: float
    position_strength: str
    closure: float
    failures: list[str] = field(default_factory=list)


def expected_caveats(t: date, msno: str, protocol: Protocol) -> tuple[str, ...]:
    if t == date(2016, 10, 1):
        cal = bool(half_mask(pd.Series([msno]), "calibration", protocol)[0])
        return ("october_2016_calibration_half",) if cal else ("months_used_for_settings",)
    if t == date(2016, 11, 1):
        return ("months_used_for_settings",)
    if t == date(2017, 2, 1):
        return ("february_2017_gap",)
    return ()


def check_horizon(
    renderer: Renderer,
    cfg: ExplainConfig,
    horizon: int,
    row: pd.Series,
    raw: dict[str, Any],
    platt: tuple[float, float],
    caveats: tuple[str, ...],
) -> HorizonCheck:
    names = cfg.group_names
    b = float(row[f"platt_b_{horizon}"])
    g = np.array([[float(row[f"g_{horizon}_{n}"]) for n in names]])
    raw_minus_base = np.array([float(row[f"raw_{horizon}"]) - float(row[f"base_{horizon}"])])
    sel: Selection = select(cfg, g, raw_minus_base, b, shown_mask(cfg, pd.DataFrame([row])))
    prob = float(platt_probability(np.array([row[f"raw_{horizon}"]]), *platt)[0])
    facts = {c: row[c] for c in {c for cs in cfg.facts.values() for c in cs}}
    out = renderer.render(sel, 0, facts, horizon, prob, caveats)
    p = renderer.p
    c_stored = {n: b * float(row[f"g_{horizon}_{n}"]) for n in names}
    d_stored = b * float(raw_minus_base[0])
    pos_s = (
        int(strength_codes(np.array([abs(d_stored)]), cfg.cutoffs())[0]) if sel.position[0] else 0
    )
    hc = HorizonCheck(
        horizon,
        out.lines,
        [],
        c_stored,
        out.d,
        out.other,
        STRENGTHS[pos_s - 1] if pos_s else "none",
        abs(sum(x[2] for x in out.listed) + out.other - out.d),
    )
    text = "\n".join(out.lines)
    # 1. facts
    for grp, _, _ in out.listed:
        spec = p["groups"][grp]["facts"]
        for col in cfg.facts[grp]:
            shown = render_fact(spec[col], row[col], p["missing_value"], col)
            from_raw = render_fact(spec[col], raw[col], p["missing_value"], col)
            claim_ok, claim = claim_holds(col, shown, spec[col], raw)
            ok = same(row[col], raw[col]) and shown == from_raw and claim_ok
            hc.facts.append(
                {
                    "group": grp,
                    "fact": col,
                    "shown": shown,
                    "feature": _plain(row[col]),
                    "raw": raw[col],
                    "claim": claim,
                    "ok": ok,
                }
            )
            if not ok:
                hc.failures.append(
                    f"{horizon} days: fact {col} shown {shown!r}, feature "
                    f"{_plain(row[col])!r}, raw {raw[col]!r} {claim}".rstrip()
                )
    # 2. nothing hidden named
    hidden_cols = [c for h in cfg.hidden for c in cfg.groups[h]]
    if any(grp in cfg.hidden for grp, _, _ in out.listed):
        hc.failures.append(f"{horizon} days: a hidden group is listed")
    if any(c in text for c in hidden_cols) or re.search(HIDDEN_WORDS, text, re.IGNORECASE):
        hc.failures.append(f"{horizon} days: the text names something hidden")
    # 3. listed plus other equals d
    if not hc.closure <= TOLERANCE:
        hc.failures.append(f"{horizon} days: listed + other - d = {hc.closure:.2e}")
    # 4. labels against the stored contributions
    for grp, s, c in out.listed:
        want = int(strength_codes(np.array([abs(c_stored[grp])]), cfg.cutoffs())[0])
        if c != c_stored[grp] or s != want:
            hc.failures.append(f"{horizon} days: {grp} label {s} vs stored {want}")
    if abs(out.d - d_stored) > TOLERANCE:
        hc.failures.append(f"{horizon} days: d differs from the stored contributions")
    pos = p["position"]
    want_pos = (
        pos["close"]
        if pos_s == 0 or not sel.position[0]
        else pos["higher" if d_stored > 0 else "lower"].format(strength=STRENGTHS[pos_s - 1])
    )
    if out.lines[0] != want_pos:
        hc.failures.append(f"{horizon} days: position line {out.lines[0]!r} vs {want_pos!r}")
    if out.listed:
        unlisted = sum(v for k, v in c_stored.items() if k not in {x[0] for x in out.listed})
        os_ = int(strength_codes(np.array([abs(unlisted)]), cfg.cutoffs())[0])
        of = p["other_factors"]
        want_other = (
            of["small"]
            if os_ == 0
            else of["raises" if unlisted > 0 else "lowers"].format(strength=STRENGTHS[os_ - 1])
        )
        if want_other not in out.lines:
            hc.failures.append(f"{horizon} days: other-factors line does not match {want_other!r}")
    # 5. caveats
    present = tuple(k for k, v in p["caveats"].items() if v in out.lines)
    if present != caveats:
        hc.failures.append(f"{horizon} days: caveats {present} vs expected {caveats}")
    return hc


def _plain(v: Any) -> Any:
    if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else v


# --- the run ----------------------------------------------------------------------------------


def candidate_rows(
    protocol: Protocol, processed: Path, contrib_dir: Path, cfg: ExplainConfig
) -> pd.DataFrame:
    """Every explained P2 main row with its facts, stored contributions and stratum columns."""
    fact_cols = sorted({c for cs in cfg.facts.values() for c in cs} | set(STRATUM_COLUMNS))
    frames = []
    for t in cfg.snapshots:
        df = load_features(processed / "features.parquet", [t], fact_cols)
        df = df.assign(snapshot_date=df.snapshot_date.astype("datetime64[ns]"))
        for h in cfg.horizons:
            part = pq.read_table(contrib_dir / f"h{h}_{t.isoformat()}.parquet").to_pandas()
            keep = {
                "raw_score": f"raw_{h}",
                "base_value": f"base_{h}",
                "platt_b": f"platt_b_{h}",
                "d": f"d_{h}",
                **{f"g_{n}": f"g_{h}_{n}" for n in cfg.group_names},
            }
            part = part[["msno", *keep]].rename(columns=keep)
            df = df.merge(part, on="msno", how="inner", validate="one_to_one")
        prior = [
            date(t.year + (t.month - k - 1) // 12, (t.month - k - 1) % 12 + 1, 1) for k in (1, 2, 3)
        ]
        seen = set(load_labels(processed / "labels.parquet", prior, ["segment"], protocol).msno)
        df["never_seen"] = ~df.msno.isin(seen)
        df["top_decile_30"] = df.raw_30 >= df.raw_30.quantile(0.9)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_raw(
    processed: Path, interim: Path, sample: pd.DataFrame, protocol: Protocol
) -> dict[tuple[str, date], RawCustomer]:
    ids = sorted(set(sample.msno))
    con = duckdb.connect()
    con.register("_ids", pd.DataFrame({"msno": ids}))
    tx = con.sql(
        f"""select * from read_parquet('{processed / "transactions.parquet"}')
            join _ids using (msno)"""
    ).df()
    # The timeline order of Phase 1 (date, cancel rank, expiry, tie keys): "last row of
    # any kind" is the highest seq. Rows dated on or after T are dropped below.
    tx = order_rows(tx).rename(columns={"transaction_date": "d"})
    tx["expiry"] = tx.expiry_date
    logs = con.sql(
        f"""select msno, make_date(date // 10000, (date // 100) % 100, date % 100) as day,
              num_25, num_50, num_75, num_985, num_100, total_secs
            from read_parquet('{interim / "user_logs.parquet"}') join _ids using (msno)"""
    ).df()
    members = con.sql(
        f"""select msno, registration_init_time, registered_via, city, bd, gender
            from read_parquet('{processed / "members.parquet"}') join _ids using (msno)"""
    ).df()
    out = {}
    for t, g in sample.groupby(sample.snapshot_date.dt.date):
        exp = load_labels(processed / "labels.parquet", [t], ["expiry_date"], protocol)
        exp = dict(zip(exp.msno, exp.expiry_date, strict=True))
        for m in g.msno:
            ctx = tx[(tx.msno == m)].assign(
                d=lambda x: pd.to_datetime(x.d).dt.date,
                expiry=lambda x: pd.to_datetime(x.expiry).dt.date,
            )
            ctx = ctx[
                [
                    "seq",
                    "d",
                    "row_kind",
                    "payment_plan_days",
                    "plan_list_price",
                    "actual_amount_paid",
                    "is_auto_renew",
                    "payment_method_id",
                    "membership_expire_date",
                    "expiry",
                    "is_cancel",
                ]
            ]
            ctx = ctx[ctx.d < t]
            lg = logs[logs.msno == m].assign(day=lambda x: pd.to_datetime(x.day).dt.date)
            lg = lg[lg.day < t]
            mem = members[members.msno == m]
            member = None
            if len(mem):
                member = {k: _plain(v) for k, v in mem.iloc[0].items() if k != "msno"}
                reg = member.get("registration_init_time")
                member["registration_init_time"] = (
                    pd.Timestamp(reg).date() if reg is not None else None
                )
            e = exp.get(m)
            out[(m, t)] = RawCustomer(
                ctx,
                daily_listening(lg)
                if len(lg)
                else pd.DataFrame(columns=["day", "secs", "plays", "full"]),
                member,
                pd.Timestamp(e).date() if e is not None else None,
            )
    return out


def weekly(daily: pd.DataFrame, t: date) -> list[dict[str, Any]]:
    out = []
    for name, lo, hi in (
        ("days 22-28 before", 22, 28),
        ("days 15-21 before", 15, 21),
        ("days 8-14 before", 8, 14),
        ("days 1-7 before", 1, 7),
    ):
        m = [(lo <= (t - d).days <= hi) for d in daily.day] if len(daily) else []
        part = daily[m] if len(daily) else daily
        out.append(
            {
                "week": name,
                "active_days": int(len(part)),
                "minutes": round(float(part.secs.sum()) / 60, 1) if len(part) else 0.0,
                "plays": int(part.plays.sum()) if len(part) else 0,
                "full_plays": int(part.full.sum()) if len(part) else 0,
            }
        )
    return out


def write_markdown(results: list[dict[str, Any]], path: Path, header: list[str]) -> None:
    lines = [*header, ""]
    for k, r in enumerate(results, 1):
        lines += [
            f"## {k}. {r['stratum']} | customer {r['msno']} | snapshot {r['snapshot']} | "
            f"{'PASS' if r['passed'] else 'FAIL'}",
            f"28-day listening window {r['window']} includes 28 January 2017: "
            f"{'yes' if r['lunar_new_year'] else 'no'}",
            "",
        ]
        for hc in r["horizons"]:
            lines += [f"### {hc['horizon']} days", "", "What the manager sees:", ""]
            lines += [f"> {x}" for x in hc["lines"]]
            lines += ["", "| group | c (calibrated log-odds) |", "|---|---|"]
            lines += [f"| {g} | {v:+.4f} |" for g, v in hc["contributions"].items()]
            lines += [
                "",
                f"d = {hc['d']:+.4f}; position strength: {hc['position_strength']}; "
                f"other factors net = {hc['other']:+.4f}; listed + other - d = "
                f"{hc['closure']:.1e}",
                "",
            ]
            if hc["facts"]:
                lines += [
                    "| fact | shown | feature | raw | claim | ok |",
                    "|---|---|---|---|---|---|",
                ]
                lines += [
                    f"| {f['fact']} | {f['shown']} | {f['feature']} | {f['raw']} | "
                    f"{f['claim']} | {'yes' if f['ok'] else 'NO'} |"
                    for f in hc["facts"]
                ]
            lines += [f"- FAIL: {x}" for x in hc["failures"]] + [""]
        lines += [
            "Last transactions before T (newest first):",
            "",
            "| date | kind | plan days | list price | paid | auto-renew | expiry (raw) | "
            "expiry (used) | cancel |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {x['d']} | {x['row_kind']} | {x['payment_plan_days']} | "
            f"{x['plan_list_price']} | {x['actual_amount_paid']} | {x['is_auto_renew']} | "
            f"{x['membership_expire_date']} | {x['expiry']} | {x['is_cancel']} |"
            for x in r["transactions"]
        ]
        lines += [
            "",
            "Listening in the 28 days before T, by week:",
            "",
            "| week | active days | minutes | plays | played to the end |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {w['week']} | {w['active_days']} | {w['minutes']} | {w['plays']} | "
            f"{w['full_plays']} |"
            for w in r["weeks"]
        ]
        lines += [
            "",
            f"Listening history before T: first day {r['first_day']}, last day {r['last_day']}.",
            "",
            f"Member row ({r['member_note']}): {r['member']}",
            "",
            "---",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def run_hand_check(
    protocol: Protocol,
    processed: Path,
    interim: Path,
    contrib_dir: Path,
    model_dir: Path,
    cfg: ExplainConfig,
    renderer: Renderer,
    out_path: Path,
) -> dict[str, Any]:
    hcfg = load_hand_check_config()
    rows = candidate_rows(protocol, processed, contrib_dir, cfg)
    rows["stratum"] = assign_strata(rows, hcfg["strata"], cfg.position_cutoff)
    sample = draw(rows, hcfg, int(hcfg["seed"]))
    raw_all = load_raw(processed, interim, sample, protocol)
    platt = {
        h: (
            json.loads((model_dir / f"model_{h}d.json").read_text())["platt_a"],
            json.loads((model_dir / f"model_{h}d.json").read_text())["platt_b"],
        )
        for h in cfg.horizons
    }
    results = []
    for _, row in sample.iterrows():
        t = row.snapshot_date.date()
        raw_c = raw_all[(row.msno, t)]
        raw = raw_facts(t, raw_c)
        caveats = expected_caveats(t, row.msno, protocol)
        hcs = [check_horizon(renderer, cfg, h, row, raw, platt[h], caveats) for h in cfg.horizons]
        member = raw_c.member
        masked = row.has_any_log_before_T == 0
        results.append(
            {
                "stratum": row.sample_stratum,
                "msno": row.msno,
                "snapshot": t.isoformat(),
                "window": f"{t - timedelta(days=28)} to {t - timedelta(days=1)}",
                "lunar_new_year": window_has(t, LUNAR_NEW_YEAR),
                "horizons": [hc.__dict__ for hc in hcs],
                "passed": not any(hc.failures for hc in hcs),
                "transactions": raw_c.transactions.sort_values("seq", ascending=False)
                .head(8)
                .astype(str)
                .to_dict("records"),
                "weeks": weekly(raw_c.daily, t),
                "first_day": str(raw_c.daily.day.min()) if len(raw_c.daily) else "none",
                "last_day": str(raw_c.daily.day.max()) if len(raw_c.daily) else "none",
                "member": member,
                "member_note": "member columns masked in the features: no listening before T"
                if masked
                else "member columns used",
            }
        )
    passed = sum(r["passed"] for r in results)
    header = [
        "# Hand check of the reasons",
        "",
        f"Seed {hcfg['seed']}; strata from configs/explain.yaml; {len(results)} rows; "
        f"passed the automatic checks: {passed} of {len(results)}. Features, stored contributions "
        "and the saved model only; no outcome. Not tracked.",
    ]
    write_markdown(results, out_path, header)
    out_path.with_suffix(".json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    return {"rows": len(results), "passed": passed, "results": results}
