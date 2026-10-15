"""Command line entry: `python -m retention.trends <command>`.

clean-months: the daily batch indicator over March to November 2016 (the registered
clean months): weekday test, false flags against the registered limit. Writes
batch_clean.json. Reads payment rows only; no outcome is read, and no day of
2016-12 or 2017-02 is loaded (the count window ends 2016-11-30).

known-days: scores the registered known days with the weekday decision and baselines
of the clean-month run. Refuses to run unless batch_clean.json exists and a tracked
file matches HEAD.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from retention.trends import batch

OUT = Path("data/processed/kkbox/sample_020/trends")
TIMELINE = Path("data/processed/kkbox/sample_020/timeline.parquet")
LABELS = Path("data/processed/kkbox/sample_020/labels.parquet")
FEATURES = Path("data/processed/kkbox/sample_020/features.parquet")


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def clean_months() -> dict:
    cfg = batch.load_indicator_config()
    # three months of history before March 2016 so the first clean month has a baseline
    counts = batch.daily_counts(TIMELINE, "2015-12-01", "2016-11-30")
    wk = batch.weekday_multipliers(counts, cfg)
    mult = wk["multipliers"] if wk["use_weekday"] else None
    ind = batch.indicator(counts, cfg, mult)
    judged = ind[ind.month.astype(str).isin(cfg.clean_months) & ind.baseline.notna()]
    flagged = judged[judged.flag]
    return {
        "commit": _head(),
        "weekday": {k: v for k, v in wk.items() if k != "multipliers"},
        "weekday_multipliers_used": wk["multipliers"] if wk["use_weekday"] else None,
        "units_judged": int(len(judged)),
        "false_flags": [
            {
                "start": str(r.start.date()),
                "end": str(r.end.date()),
                "kind": r.kind,
                "count": r.count,
                "baseline": r.baseline,
                "ratio": round(float(r.ratio), 3),
            }
            for r in flagged.itertuples()
        ],
        "limit": cfg.max_false_flags,
        "within_limit": int(len(flagged)) <= cfg.max_false_flags,
        "min_ratio_clean": round(float(judged.ratio.min()), 3),
        "ratio_quantiles_clean": {
            q: round(float(judged.ratio.quantile(q)), 3) for q in (0.01, 0.05, 0.5)
        },
    }


def known_days() -> dict:
    cfg = batch.load_indicator_config()
    clean = json.loads((OUT / "batch_clean.json").read_text())
    counts = batch.daily_counts(TIMELINE, "2016-09-01", "2017-03-31")
    # baselines need three months; load from 2016-09 so 2016-12 and 2017-02 have them
    mult = clean["weekday_multipliers_used"]
    mult = {int(k): v for k, v in mult.items()} if mult else None
    ind = batch.indicator(counts, cfg, mult)

    def find(day: str):
        d = pd.Timestamp(day)
        hit = ind[(ind.start <= d) & (ind.end >= d)]
        if hit.empty:
            return {"date": day, "unit": None, "note": "not judged (day 1)"}
        r = hit.iloc[0]
        return {
            "date": day,
            "unit": f"{r.start.date()}..{r.end.date()}",
            "kind": r.kind,
            "count": r["count"],
            "baseline": r.baseline,
            "ratio": round(float(r.ratio), 3),
            "flag": bool(r.flag),
        }

    around: dict[str, list[dict]] = {}
    for d in (*cfg.known_gap_days, *cfg.known_non_gap_days):
        t = pd.Timestamp(d)
        around[d] = [find(str((t + pd.Timedelta(days=k)).date())) for k in (-2, -1, 1, 2)]
    blocks = [
        {
            "unit": f"{r.start.date()}..{r.end.date()}",
            "count": r["count"],
            "baseline": r.baseline,
            "ratio": round(float(r.ratio), 3),
            "flag": bool(r.flag),
        }
        for _, r in ind[
            (ind.kind == "block") & ind.month.astype(str).isin(["2016-12", "2017-02"])
        ].iterrows()
    ]
    gaps = [find(d) for d in cfg.known_gap_days]
    non = [find(d) for d in cfg.known_non_gap_days]
    return {
        "commit": _head(),
        "known_gap_days": gaps,
        "known_non_gap_days": non,
        "two_days_either_side": around,
        "month_end_blocks": blocks,
        "flags_known_gap_days": all(g["flag"] for g in gaps),
        "does_not_flag_moved_batch_day": not any(n["flag"] for n in non),
        "false_flags_within_limit": clean["within_limit"],
    }


def month_end_check() -> dict:
    """Paid weekly rate 2016-02-15..2016-12-19 (snapshots before 2017-01-01), then the test."""
    from datetime import date

    from retention.models.data import MAX_SNAPSHOTS_PER_READ, load_labels
    from retention.models.protocol import load_protocol
    from retention.trends.series import COLUMNS, month_end_test, weekly_rates

    protocol = load_protocol()
    snaps = [date(2016, m, 1) for m in range(2, 13)]
    assert max(snaps) < protocol.locked_from
    cols = [c for c in COLUMNS if c not in ("msno", "snapshot_date")]
    frames = [
        load_labels(LABELS, snaps[i : i + MAX_SNAPSHOTS_PER_READ], cols, protocol)
        for i in range(0, len(snaps), MAX_SNAPSHOTS_PER_READ)
    ]
    lab = pd.concat(frames, ignore_index=True)
    w = weekly_rates(lab, "paid")
    w = w[(w.index >= "2016-02-15") & (w.index <= "2016-12-19")]
    res = month_end_test(w)
    return {
        "commit": _head(),
        "rows_read": int(len(lab)),
        "weeks": int(len(w)),
        "weeks_with_unobserved_rows": int((w.unobserved > 0).sum()),
        "unobserved_rows": int(w.unobserved.sum()),
        "all_weeks_complete": bool(w.complete.all()),
        "test": res,
        "table": [
            {
                "week": str(i.date()),
                **{
                    k: (float(v) if k == "rate" else int(v))
                    for k, v in r.items()
                    if k != "complete"
                },
            }
            for i, r in w.iterrows()
        ],
    }


def study_run() -> dict:
    """The registered synthetic study; real weekly paid event counts only (no outcome)."""
    import resource
    import time
    from datetime import date

    import yaml

    from retention.models.data import MAX_SNAPSHOTS_PER_READ, load_labels
    from retention.models.protocol import load_protocol
    from retention.trends import study as st

    protocol = load_protocol()
    snaps = [date(2016, m, 1) for m in range(2, 13)]
    cols = ["expiry_date", "in_expiry_month", "segment"]
    lab = pd.concat(
        [
            load_labels(LABELS, snaps[i : i + MAX_SNAPSHOTS_PER_READ], cols, protocol)
            for i in range(0, len(snaps), MAX_SNAPSHOTS_PER_READ)
        ]
    )
    d = lab[lab.in_expiry_month.astype(bool) & (lab.segment == "paid")]
    wk = d.expiry_date.dt.to_period("W-SUN").dt.start_time
    n = d.groupby(wk).size()
    n = n[(n.index >= "2016-02-15") & (n.index <= "2016-12-19")]
    cfg = yaml.safe_load(Path("configs/changepoints.yaml").read_text())["detector"]
    t0 = time.time()
    res = st.run_study(n.to_numpy(dtype=float), cfg)
    res["commit"] = _head()
    res["seconds"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    res["real_count_weeks"] = int(len(n))
    res["real_counts_min_median_max"] = [int(n.min()), int(n.median()), int(n.max())]
    return res


def _bucket(x: pd.Series, edges: list[tuple[str, float, float]]) -> pd.Series:
    out = pd.Series("missing", index=x.index, dtype="object")
    for name, lo, hi in edges:
        out[(x >= lo) & (x <= hi)] = name
    return out


def alerts_run() -> dict:
    """The one detector run on the real paid weekly series, with every registered rule."""
    import json as _json
    from datetime import date

    import numpy as np
    import yaml

    from retention.models.data import MAX_SNAPSHOTS_PER_READ, load_features, load_labels
    from retention.models.protocol import load_protocol
    from retention.trends import alerts as al
    from retention.trends import study as st
    from retention.trends.series import COLUMNS, weekly_rates

    full = yaml.safe_load(Path("configs/changepoints.yaml").read_text())
    det, acfg = full["detector"], full["alerts"]
    study = _json.loads((OUT / "detector_study.json").read_text())
    c = float(study["chosen_c"])  # the c the registered selection rule chose
    protocol = load_protocol()
    snaps = [date(2016, m, 1) for m in range(2, 13)]
    chunks = [
        snaps[i : i + MAX_SNAPSHOTS_PER_READ] for i in range(0, len(snaps), MAX_SNAPSHOTS_PER_READ)
    ]
    lcols = [x for x in COLUMNS if x not in ("msno", "snapshot_date")]
    lab = pd.concat([load_labels(LABELS, ch, lcols, protocol) for ch in chunks], ignore_index=True)
    w = weekly_rates(lab, "paid")
    w = w[(w.index >= "2016-02-15") & (w.index <= "2016-12-19")]
    rates, events = w.rate.to_numpy(), w.events.to_numpy(dtype=float)
    cps = st.detect(rates, events, c=c, min_size=int(det["min_size_weeks"]))
    segs = al.segments_from_cps(len(w), cps)

    # data-quality inputs (payment rows and the month-quality table; no outcome)
    icfg = batch.load_indicator_config()
    counts = batch.daily_counts(TIMELINE, "2015-12-01", "2017-03-31")
    ind = batch.indicator(counts, icfg)
    gaps = batch.gap_dates(ind)
    mq = pd.read_parquet("data/processed/kkbox/sample_020/month_quality.parquet")
    status = {str(m): s for m, s in zip(mq.month, mq.status, strict=True)}
    weeks = list(w.index)
    wflags = [al.week_flags(x, gaps, status, acfg) for x in weeks]
    flagged = np.array(
        [
            bool(f["renewal_gap_dates"] or f["unreliable_months"] or f["uncertain_months"])
            for f in wflags
        ]
    )
    quiet = al.quiet_week_sd(rates, segs, flagged)

    # drill-down frames: paid expiry events with features at the snapshot's first day
    feats = [
        "last_plan_days",
        "last_payment_method_id",
        "last_is_auto_renew",
        "city",
        "registered_via",
        "tenure_days",
    ]
    fr = pd.concat([load_features(FEATURES, ch, feats) for ch in chunks], ignore_index=True)
    d = lab[lab.in_expiry_month.astype(bool) & (lab.segment == "paid")].merge(
        fr, on=["msno", "snapshot_date"], how="left"
    )
    d["week"] = d.expiry_date.dt.to_period("W-SUN").dt.start_time
    d = d[d.week.isin(weeks) & d.churned_classic.notna()].copy()
    d["widx"] = d.week.map({x: i for i, x in enumerate(weeks)})
    d["churned"] = d.churned_classic.astype(int)
    dims = al.segment_columns(d)

    def window(a: int, b: int) -> pd.DataFrame:
        m = (d.widx >= a) & (d.widx < b)
        parts = []
        for name, col in dims.items():
            g = (
                pd.DataFrame({"value": col[m], "churned": d.churned[m]})
                .groupby("value")
                .churned.agg(events="size", churned="sum")
            )
            g.insert(0, "dimension", name)
            parts.append(g.reset_index())
        return pd.concat(parts, ignore_index=True)

    maxw = int(acfg["windows"]["max_weeks_each_side"])
    small_pts = float(acfg["size_small_points"])
    out_alerts = []
    for i, cp in enumerate(cps):
        (b0, b1) = next(s for s in segs if s[1] == cp)
        (a0, a1) = next(s for s in segs if s[0] == cp)
        b0w, a1w = max(b0, b1 - maxw), min(a1, a0 + maxw)
        min_w = int(det["min_size_weeks"])
        short = (b1 - b0w) < min_w or (a1w - a0) < min_w
        if short:
            rb = ra = size = None
            is_note = "a window had fewer than the minimum weeks; no rate is stated"
        else:
            rb = float(d[(d.widx >= b0w) & (d.widx < b1)].churned.mean())
            ra = float(d[(d.widx >= a0) & (d.widx < a1w)].churned.mean())
            size = 100 * (ra - rb)
            is_note = None
        nxt = next((s for s in segs if s[0] == a1), None)
        m_next = float(rates[nxt[0] : nxt[1]].mean()) if nxt is not None else None
        if short:
            band, returned = [None, None], None
        else:
            band, returned = al.returned_to_level(rb, quiet["sd"], m_next)
        ret_week = str(weeks[nxt[0]].date()) if returned and nxt is not None else None
        win = range(max(0, cp - 1), min(len(weeks), cp + 2))
        flags = al.merge_flags([wflags[j] for j in win])
        is_small = (not short) and abs(size) < small_pts
        ranked: list[dict] = []
        too_small: list[dict] = []
        totals: dict = {}
        if not is_small and not short:
            rows, too_small, totals = al.contributions(
                window(b0w, b1), window(a0, a1w), int(acfg["drill_down"]["min_events_each_period"])
            )
            ranked = sorted(rows, key=lambda r: -abs(r["rate_contribution"]))[
                : int(acfg["drill_down"]["top_segments"])
            ]
        alert = {
            "alert_id": f"cp{i + 1}",
            "week": str(weeks[cp].date()),
            "series": "the paid series",
            "before_rate": rb,
            "after_rate": ra,
            "size_points": None if size is None else round(size, 3),
            "note": is_note,
            "direction": None if short else ("up" if ra > rb else "down"),
            "size_class": "not_computable" if short else ("small" if is_small else "shift"),
            "windows": {
                "rule": "detector segments, at most 8 weeks each side",
                "before_weeks": [str(weeks[b0w].date()), str(weeks[b1 - 1].date())]
                if b1 > b0w
                else None,
                "after_weeks": [str(weeks[a0].date()), str(weeks[a1w - 1].date())],
                "segment_after": [str(weeks[a0].date()), str(weeks[a1 - 1].date())],
            },
            "series_returned_to_level": returned,
            "return_week": ret_week,
            "band": band,
            "flags": {
                **flags,
                "renewal_gap_dates": [str(x.date()) for x in flags["renewal_gap_dates"]],
                "gap_same_weeks": [str(x.date()) for x in flags["gap_same_weeks"]],
            },
            "data_quality_flags": al.flag_names(flags),
            "top_segments": ranked,
            "too_small_segments": too_small,
            "dimension_totals": totals,
            "noise_assumption": {
                "validated_at": "1.5x binomial noise",
                "c": c,
                "min_segment_weeks": int(det["min_size_weeks"]),
                "real_series_false_alarm_rate": "not validated",
                "status": "exploratory",
            },
        }
        alert["sentence"] = al.render(alert, acfg)
        out_alerts.append(alert)
    result = {
        "commit": _head(),
        "series_weeks": [str(weeks[0].date()), str(weeks[-1].date())],
        "weeks": len(weeks),
        "change_points": [str(weeks[k].date()) for k in cps],
        "quiet_week_sd": {
            **quiet,
            "sd": None if not np.isfinite(quiet["sd"]) else quiet["sd"],
            "computable": bool(np.isfinite(quiet["sd"])),
        },
        "small_threshold_points": small_pts,
        "alerts": out_alerts,
        "weekly": [
            {"week": str(x.date()), "events": int(e), "rate": float(r), "flagged": bool(f)}
            for x, e, r, f in zip(weeks, events, rates, flagged, strict=True)
        ],
    }
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/processed/alerts.json").write_text(_json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    p = argparse.ArgumentParser(prog="retention.trends")
    p.add_argument(
        "command", choices=["clean-months", "known-days", "month-end-check", "study", "alerts"]
    )
    a = p.parse_args()
    if _dirty():
        raise SystemExit("a tracked file differs from HEAD; commit first")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.command == "alerts":
        res, name = alerts_run(), "alerts_run.json"
    elif a.command == "study":
        res, name = study_run(), "detector_study.json"
    elif a.command == "month-end-check":
        res, name = month_end_check(), "month_end_check.json"
    elif a.command == "clean-months":
        res, name = clean_months(), "batch_clean.json"
    else:
        if not (OUT / "batch_clean.json").exists():
            raise SystemExit("run clean-months first and look at it")
        res, name = known_days(), "batch_known_days.json"
    (OUT / name).write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
