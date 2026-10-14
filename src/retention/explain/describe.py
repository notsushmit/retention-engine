"""Descriptive tables of the contributions; they decide nothing.

B: per horizon and month, how often reasons are listed, their strength mix,
the position mix, the "no group qualifies" shares, the other-factors mix, and
how often the unnamed groups alone would have reached the slight cutoff; per
group, how often it is listed in each direction.

C: for each group, direction and horizon, the distribution of the group's fact
columns among rows where the group is listed in that direction, next to the
same distribution over all rows (the reference). Reference points are fixed in
advance; every share carries its denominator.

Illustrative rows: a few listed rows per group and direction, drawn with a
fixed seed, for reading. Features and contributions only; no outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from retention.explain.config import STRENGTHS, ExplainConfig
from retention.explain.contributions import ChunkResult
from retention.explain.selection import POSITIONS

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
# Registered reference points: (operator, value) pairs per numeric fact column.
REFERENCE_POINTS: dict[str, list[tuple[str, float]]] = {
    "ratio_secs_7_vs_base": [("<", 1.0)],
    "ratio_active_7_vs_base": [("<", 1.0)],
    "days_until_expiry": [("<=", 7)],
    "days_since_last_payment": [(">", 31)],
    "active_days_28": [("==", 0)],
    "secs_per_day_28": [("==", 0)],
    "days_since_last_listen": [(">=", 7)],
    "complete_share_28": [("<", 0.5)],
    "last_amount_paid": [("==", 0)],
    "last_plan_days": [("==", 30), ("==", 0)],
    "tenure_days": [("<", 365)],
}
BOOLEAN_FACTS = (
    "has_any_log_before_T",
    "first_log_in_window",
    "last_is_auto_renew",
    "last_row_is_cancel",
)
CATEGORY_FACTS = {"last_row_kind": ("paid", "discount100", "promo", "trial")}
ILLUSTRATIVE_SEED = 20161013
ILLUSTRATIVE_PER_CELL = 3
OPS = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
}


def _share(k: int, n: int) -> dict[str, Any]:
    return {"share": k / n if n else None, "count": int(k), "of": int(n)}


# --- fact summaries --------------------------------------------------------------------


def fact_summary(values: pd.Series, column: str) -> dict[str, Any]:
    """Distribution of one fact column; shares are of the non-missing values."""
    n = len(values)
    known = values.dropna()
    out: dict[str, Any] = {"rows": n, "missing": _share(n - len(known), n)}
    if column in BOOLEAN_FACTS:
        out["true"] = _share(int((known.astype(int) == 1).sum()), len(known))
        return out
    if column in CATEGORY_FACTS:
        s = known.astype(str)
        out["levels"] = {lv: _share(int((s == lv).sum()), len(s)) for lv in CATEGORY_FACTS[column]}
        return out
    x = known.to_numpy(dtype=float)
    out["quantiles"] = (
        {f"p{int(q * 100)}": float(np.quantile(x, q)) for q in QUANTILES} if len(x) else {}
    )
    out["reference_points"] = {
        f"{op} {v:g}": _share(int(OPS[op](x, v).sum()), len(x))
        for op, v in REFERENCE_POINTS.get(column, [])
    }
    return out


# --- per-row frame ------------------------------------------------------------------------


def listed_mask(sel_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """rows x groups: True where the group is in one of the listed slots."""
    out = np.zeros((sel_idx.shape[0], n_groups), dtype=bool)
    for j in range(sel_idx.shape[1]):
        ok = sel_idx[:, j] >= 0
        out[np.flatnonzero(ok), sel_idx[ok, j]] = True
    return out


def _cells(results: Sequence[ChunkResult], horizon: int) -> list[ChunkResult]:
    return [r for r in results if r.horizon == horizon]


# --- table B ------------------------------------------------------------------------------


def table_b_month(cfg: ExplainConfig, r: ChunkResult) -> dict[str, Any]:
    s, n = r.selection, len(r.selection.d)
    slight = cfg.cutoffs()[0]
    out: dict[str, Any] = {
        "horizon": r.horizon,
        "snapshot": r.snapshot.isoformat(),
        "rows": n,
        "any_raising": _share(int((s.n_raising > 0).sum()), n),
        "any_lowering": _share(int((s.n_lowering > 0).sum()), n),
        "position": {
            name: _share(int((s.position == code).sum()), n)
            for code, name in zip((-1, 0, 1), POSITIONS, strict=True)
        },
        "higher_but_no_raising_group": _share(int(s.many_small_raising().sum()), n),
        "lower_but_no_lowering_group": _share(int(s.many_small_lowering().sum()), n),
        "none_stand_out": _share(int(s.none_stand_out().sum()), n),
    }
    for name, strength in (("raising", s.raising_strength), ("lowering", s.lowering_strength)):
        listed = strength[strength > 0]
        out[f"{name}_strength_mix"] = {
            STRENGTHS[k - 1]: _share(int((listed == k).sum()), len(listed)) for k in (1, 2, 3)
        }
    other_dir = np.sign(s.other).astype(int)
    mix = {"small": _share(int((s.other_strength == 0).sum()), n)}
    for k in (1, 2, 3):
        for sign, word in ((1, "raises"), (-1, "lowers")):
            hit = (s.other_strength == k) & (other_dir == sign)
            mix[f"{word} {STRENGTHS[k - 1]}"] = _share(int(hit.sum()), n)
    out["other_factors"] = mix
    names = cfg.group_names
    for g in cfg.hidden:
        out[f"hidden_{g}_alone_reaches_slight"] = _share(
            int((np.abs(s.c[:, names.index(g)]) >= slight).sum()), n
        )
    for g, col in cfg.shown_only_when_known.items():
        missing = r.rows[col].isna().to_numpy()
        big = np.abs(s.c[:, names.index(g)]) >= slight
        out[f"missing_{g}_alone_reaches_slight"] = _share(int((missing & big).sum()), n)
    return out


def table_b_groups(cfg: ExplainConfig, results: Sequence[ChunkResult]) -> dict[str, Any]:
    """Per horizon and group, pooled over the snapshots given."""
    slight = cfg.cutoffs()[0]
    names, out = cfg.group_names, {}
    for h in cfg.horizons:
        cells = _cells(results, h)
        n = sum(len(r.selection.d) for r in cells)
        up = np.vstack([listed_mask(r.selection.raising, len(names)) for r in cells])
        down = np.vstack([listed_mask(r.selection.lowering, len(names)) for r in cells])
        c = np.vstack([r.selection.c for r in cells])
        rows = {}
        for k, g in enumerate(names):
            listed_abs = np.abs(c[up[:, k] | down[:, k], k])
            rows[g] = {
                "listed_raising": _share(int(up[:, k].sum()), n),
                "listed_lowering": _share(int(down[:, k].sum()), n),
                "c_at_or_above_slight": _share(int((c[:, k] >= slight).sum()), n),
                "c_at_or_below_minus_slight": _share(int((c[:, k] <= -slight).sum()), n),
                "median_abs_c_when_listed": float(np.median(listed_abs))
                if len(listed_abs)
                else None,
            }
        out[str(h)] = rows
    return out


# --- table C: facts check ----------------------------------------------------------------------


def facts_check(cfg: ExplainConfig, results: Sequence[ChunkResult]) -> dict[str, Any]:
    names, out = cfg.group_names, {}
    for h in cfg.horizons:
        cells = _cells(results, h)
        frame = pd.concat([r.rows for r in cells], ignore_index=True)
        up = np.vstack([listed_mask(r.selection.raising, len(names)) for r in cells])
        down = np.vstack([listed_mask(r.selection.lowering, len(names)) for r in cells])
        per_h: dict[str, Any] = {}
        for g, cols in cfg.facts.items():
            k = names.index(g)
            per_g: dict[str, Any] = {}
            for direction, mask in (("raises", up[:, k]), ("lowers", down[:, k])):
                per_g[direction] = {
                    "listed_rows": _share(int(mask.sum()), len(frame)),
                    "facts": {
                        col: {
                            "listed": fact_summary(frame.loc[mask, col], col),
                            "all_rows": fact_summary(frame[col], col),
                        }
                        for col in cols
                    },
                }
            per_h[g] = per_g
        out[str(h)] = per_h
    return out


def facts_reference_by_snapshot(
    cfg: ExplainConfig, results: Sequence[ChunkResult]
) -> dict[str, Any]:
    """The all-rows reference distribution of every fact column, per horizon and snapshot."""
    cols = sorted({c for cs in cfg.facts.values() for c in cs})
    return {
        f"{r.horizon}|{r.snapshot.isoformat()}": {c: fact_summary(r.rows[c], c) for c in cols}
        for r in results
    }


# --- illustrative rows ------------------------------------------------------------------------


def illustrative_rows(
    cfg: ExplainConfig,
    results: Sequence[ChunkResult],
    seed: int = ILLUSTRATIVE_SEED,
    per_cell: int = ILLUSTRATIVE_PER_CELL,
) -> list[dict[str, Any]]:
    """`per_cell` listed rows per horizon, group and direction, drawn once with `seed`."""
    rng = np.random.default_rng(seed)
    names, out = cfg.group_names, []
    for h in cfg.horizons:
        cells = _cells(results, h)
        frame = pd.concat([r.rows for r in cells], ignore_index=True)
        sels = [r.selection for r in cells]
        up = np.vstack([listed_mask(s.raising, len(names)) for s in sels])
        down = np.vstack([listed_mask(s.lowering, len(names)) for s in sels])
        c = np.vstack([s.c for s in sels])
        d = np.concatenate([s.d for s in sels])
        pos = np.concatenate([s.position for s in sels])
        raising = np.vstack([s.raising for s in sels])
        lowering = np.vstack([s.lowering for s in sels])
        rs = np.vstack([s.raising_strength for s in sels])
        ls = np.vstack([s.lowering_strength for s in sels])
        other = np.concatenate([s.other for s in sels])
        for g, cols in cfg.facts.items():
            k = names.index(g)
            for direction, mask in (("raises", up[:, k]), ("lowers", down[:, k])):
                idx = np.flatnonzero(mask)
                take = np.sort(rng.choice(idx, size=min(per_cell, len(idx)), replace=False))
                for i in take:
                    row = frame.iloc[i]
                    out.append(
                        {
                            "horizon": h,
                            "group": g,
                            "direction": direction,
                            "msno": str(row.msno),
                            "snapshot": str(pd.Timestamp(row.snapshot_date).date()),
                            "facts": {col: _plain(row[col]) for col in cols},
                            "c_group": float(c[i, k]),
                            "d": float(d[i]),
                            "position": POSITIONS[int(pos[i]) + 1],
                            "listed_raising": [
                                (names[j], STRENGTHS[s - 1])
                                for j, s in zip(raising[i], rs[i], strict=True)
                                if j >= 0
                            ],
                            "listed_lowering": [
                                (names[j], STRENGTHS[s - 1])
                                for j, s in zip(lowering[i], ls[i], strict=True)
                                if j >= 0
                            ],
                            "other": float(other[i]),
                        }
                    )
    return out


def _plain(v: Any) -> Any:
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else v
