"""Which groups are listed for each row, with what strength; position and other factors.

All on the calibrated log-odds scale: c = b x group contribution and
d = b x (raw score - base value), b the horizon's Platt slope.

- A group can be listed only if it is shown (never a hidden group, never a
  group shown only when known whose column is missing for that row).
- Raising: shown groups with c >= ln(slight multiplier), largest c first, at
  most top_raising. Lowering: shown groups with c <= -ln(slight), most negative
  first, at most top_lowering. Ties keep the group order.
- Strength code per listed group: 3 strong, 2 moderate, 1 slight.
- Position: higher if d >= ln(position multiplier), lower if d <= -ln(...),
  close otherwise.
- Other factors: the sum of c over every group not listed (hidden groups,
  groups not shown for the row, groups below the cut or beyond the caps).
  Listed plus other equals d up to floating-point rounding.

Groups and labels only; the wording comes from configs/reason_phrases.yaml in
a later step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retention.explain.config import ExplainConfig

POSITIONS = ("lower", "close", "higher")  # code -1, 0, 1


@dataclass
class Selection:
    c: np.ndarray  # rows x groups
    d: np.ndarray
    raising: np.ndarray  # rows x top_raising group indices, -1 = empty slot
    raising_strength: np.ndarray  # same shape, 0 for empty
    lowering: np.ndarray
    lowering_strength: np.ndarray
    position: np.ndarray  # -1, 0, 1
    other: np.ndarray
    other_strength: np.ndarray  # 0 = small net effect, else 1 to 3 (direction from sign)

    @property
    def n_raising(self) -> np.ndarray:
        return (self.raising >= 0).sum(axis=1)

    @property
    def n_lowering(self) -> np.ndarray:
        return (self.lowering >= 0).sum(axis=1)

    def many_small_raising(self) -> np.ndarray:
        return (self.position == 1) & (self.n_raising == 0)

    def many_small_lowering(self) -> np.ndarray:
        return (self.position == -1) & (self.n_lowering == 0)

    def none_stand_out(self) -> np.ndarray:
        return (self.position == 0) & (self.n_raising == 0) & (self.n_lowering == 0)

    def listed_sum(self) -> np.ndarray:
        rows = np.arange(len(self.d))[:, None]
        out = np.zeros(len(self.d))
        for idx in (self.raising, self.lowering):
            vals = np.where(idx >= 0, self.c[rows, np.maximum(idx, 0)], 0.0)
            out += vals.sum(axis=1)
        return out


def strength_codes(abs_c: np.ndarray, cutoffs: tuple[float, float, float]) -> np.ndarray:
    """0 below slight, 1 slight, 2 moderate, 3 strong (cutoffs are inclusive)."""
    slight, moderate, strong = cutoffs
    out = np.zeros(abs_c.shape, dtype=np.int8)
    out[abs_c >= slight] = 1
    out[abs_c >= moderate] = 2
    out[abs_c >= strong] = 3
    return out


def shown_mask(cfg: ExplainConfig, rows: pd.DataFrame) -> np.ndarray:
    """rows x groups: True where the group may be listed for that row."""
    mask = np.ones((len(rows), len(cfg.groups)), dtype=bool)
    for k, g in enumerate(cfg.group_names):
        if g in cfg.hidden:
            mask[:, k] = False
        elif g in cfg.shown_only_when_known:
            mask[:, k] = rows[cfg.shown_only_when_known[g]].notna().to_numpy()
    return mask


def _pick(key: np.ndarray, top: int) -> np.ndarray:
    """Indices of the `top` largest finite keys per row (stable: ties keep group order)."""
    order = np.argsort(-key, axis=1, kind="stable")[:, :top]
    picked = np.take_along_axis(key, order, axis=1)
    return np.where(np.isfinite(picked), order, -1)


def select(
    cfg: ExplainConfig,
    group_sums: np.ndarray,
    raw_minus_base: np.ndarray,
    platt_b: float,
    shown: np.ndarray,
) -> Selection:
    if not platt_b > 0:
        raise ValueError(f"Platt slope b = {platt_b} is not positive")
    cutoffs = cfg.cutoffs()
    slight = cutoffs[0]
    c = platt_b * np.asarray(group_sums, dtype=float)
    d = platt_b * np.asarray(raw_minus_base, dtype=float)
    up = np.where(shown & (c >= slight), c, -np.inf)
    down = np.where(shown & (c <= -slight), -c, -np.inf)
    raising = _pick(up, cfg.top_raising)
    lowering = _pick(down, cfg.top_lowering)
    rows = np.arange(len(d))[:, None]

    def strength(idx: np.ndarray) -> np.ndarray:
        vals = np.abs(c[rows, np.maximum(idx, 0)])
        return np.where(idx >= 0, strength_codes(vals, cutoffs), 0).astype(np.int8)

    listed = np.zeros(c.shape, dtype=bool)
    for idx in (raising, lowering):
        for j in range(idx.shape[1]):
            ok = idx[:, j] >= 0
            listed[np.flatnonzero(ok), idx[ok, j]] = True
    other = np.where(listed, 0.0, c).sum(axis=1)
    pos_cut = cfg.position_cutoff
    position = np.where(d >= pos_cut, 1, np.where(d <= -pos_cut, -1, 0)).astype(np.int8)
    other_strength = np.where(
        np.abs(other) >= cfg.other_small_cutoff, strength_codes(np.abs(other), cutoffs), 0
    ).astype(np.int8)
    return Selection(
        c=c,
        d=d,
        raising=raising,
        raising_strength=strength(raising),
        lowering=lowering,
        lowering_strength=strength(lowering),
        position=position,
        other=other,
        other_strength=other_strength,
    )
