"""Churn archetypes, as registered in configs/archetypes.yaml (no outcome is read here).

Representation: for every P2 main row of the explained snapshots, the 30-day
group contributions on the calibrated log-odds scale (b x group sum) of the 11
shown groups; a missing tenure counts as 0; the hidden group is left out. Rows
whose last plan was free (trial, promo, discount100) form the archetype defined
by rule and are taken out before clustering.

Stages, each a function so it can be run and inspected on its own:
  1. one_fit_timing: one HDBSCAN fit on the real fit sample and the vote for
     every other row, with time and memory;
  2. select_setting: the registered grid, chosen by resample stability (ARI);
  3. expiry_diagnostic: adjusted mutual information with days_until_expiry on
     2016-10, and the registered branches;
  4. judge: the seven criteria, "holds up", the share series and profiles.

Every sample is drawn with a seed fixed in advance (base + offset, see SEEDS)
from rows sorted by customer within each snapshot, snapshot by snapshot. Labels
on an evaluation set always come from the 10-nearest-neighbour vote of the
clustering judged; for shares and profiles, rows in the reference fit sample
keep their HDBSCAN label and every other row is voted.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from retention.explain.config import ExplainConfig
from retention.models.data import load_features
from retention.models.resources import peak_rss_gb

ARCHETYPES_CONFIG = Path("configs/archetypes.yaml")
BASE_SEED = 20161001
# Offsets from the base seed, fixed before any clustering ran (also in the config).
SEEDS = {
    "fit_sample": 0,
    "evaluation_set": 100,
    "resamples": (1, 20),
    "month_fit_2016_10": 201,
    "month_fit_2016_11": 202,
    "fit_2017": 301,
    "evaluation_set_2017": 302,
    "expiry_diagnostic": 401,
}
NOISE = -1
RULE = -2  # the rule-defined zero-paid archetype
ZERO_PAID_KINDS = ("trial", "promo", "discount100")
OCT16, NOV16, JAN17, FEB17 = (
    date(2016, 10, 1),
    date(2016, 11, 1),
    date(2017, 1, 1),
    date(2017, 2, 1),
)


def load_archetypes_config(path: Path = ARCHETYPES_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


# --- representation --------------------------------------------------------------------------


@dataclass
class Rows:
    """Every P2 main row of the explained snapshots, in snapshot then customer order."""

    msno: np.ndarray
    snapshot: np.ndarray  # datetime.date per row
    rule: np.ndarray  # zero-paid last row (the rule-defined archetype)
    x: np.ndarray  # rows x groups, calibrated 30-day contributions
    groups: list[str]
    days_until_expiry: np.ndarray
    x7: np.ndarray | None = None  # the same at 7 days (for the diagnostic)

    def where(self, snapshot: date, non_rule: bool = True) -> np.ndarray:
        m = self.snapshot == snapshot
        return np.flatnonzero(m & ~self.rule) if non_rule else np.flatnonzero(m)


def group_matrix(part: pd.DataFrame, groups: list[str], tenure_known: np.ndarray) -> np.ndarray:
    b = float(part.platt_b.iloc[0])
    x = b * part[[f"g_{g}" for g in groups]].to_numpy(dtype=float)
    if "tenure" in groups:
        k = groups.index("tenure")
        x[:, k] = np.where(tenure_known, x[:, k], 0.0)
    return x


def load_rows(
    processed: Path,
    contrib_dir: Path,
    cfg: ExplainConfig,
    groups: list[str],
    with_7_days: bool = True,
) -> Rows:
    parts = []
    for t in cfg.snapshots:
        p30 = pq.read_table(contrib_dir / f"h30_{t.isoformat()}.parquet").to_pandas()
        f = load_features(
            processed / "features.parquet",
            [t],
            ["last_row_kind", "tenure_days", "days_until_expiry"],
        ).drop(columns="snapshot_date")
        df = p30.merge(f, on="msno", how="left", validate="one_to_one")
        if with_7_days:
            p7 = pq.read_table(contrib_dir / f"h7_{t.isoformat()}.parquet").to_pandas()
            p7 = p7[["msno", "platt_b", *[f"g_{g}" for g in groups]]]
            df = df.merge(p7, on="msno", suffixes=("", "_h7"), validate="one_to_one")
        df = df.sort_values("msno", kind="stable").reset_index(drop=True)
        known = df.tenure_days.notna().to_numpy()
        x = group_matrix(df, groups, known)
        x7 = None
        if with_7_days:
            d7 = df[["platt_b_h7", *[f"g_{g}_h7" for g in groups]]].copy()
            d7.columns = ["platt_b", *[f"g_{g}" for g in groups]]
            x7 = group_matrix(d7, groups, known)
        parts.append((df, x, x7, t))
    return Rows(
        msno=np.concatenate([df.msno.astype(str).to_numpy() for df, *_ in parts]),
        snapshot=np.concatenate([np.full(len(df), t) for df, _, _, t in parts]),
        rule=np.concatenate(
            [df.last_row_kind.isin(ZERO_PAID_KINDS).to_numpy() for df, *_ in parts]
        ),
        x=np.vstack([x for _, x, _, _ in parts]),
        groups=list(groups),
        days_until_expiry=np.concatenate(
            [df.days_until_expiry.to_numpy(dtype=float) for df, *_ in parts]
        ),
        x7=np.vstack([x7 for *_, x7, _ in parts]) if with_7_days else None,
    )


# --- samples -------------------------------------------------------------------------------------


def draw(
    rows: Rows, offset: int, per_snapshot: dict[date, int], exclude: np.ndarray | None = None
) -> np.ndarray:
    """n non-rule rows per snapshot (date order), without replacement, seed base + offset."""
    rng = np.random.default_rng(BASE_SEED + offset)
    out = []
    for t in sorted(per_snapshot):
        pool = rows.where(t)
        if exclude is not None:
            pool = np.setdiff1d(pool, exclude, assume_unique=True)
        out.append(np.sort(rng.choice(pool, size=per_snapshot[t], replace=False)))
    return np.concatenate(out)


@dataclass
class Samples:
    fit: np.ndarray
    evaluation: np.ndarray
    resamples: list[np.ndarray]
    month_fit: dict[date, np.ndarray]
    fit_2017: np.ndarray
    evaluation_2017: np.ndarray
    diagnostic: np.ndarray


def draw_samples(rows: Rows, acfg: dict[str, Any]) -> Samples:
    c = acfg["clustering"]["fit_sample"]["per_snapshot"]
    fit_n = {date.fromisoformat(str(k)): v for k, v in c.items()}
    e_n = {
        date.fromisoformat(str(k)): v
        for k, v in acfg["evaluation_sets"]["E"]["per_snapshot"].items()
    }
    e17_n = {
        date.fromisoformat(str(k)): v
        for k, v in acfg["evaluation_sets"]["E_2017"]["per_snapshot"].items()
    }
    n_fit = acfg["clustering"]["fit_sample"]["rows"]
    fit = draw(rows, SEEDS["fit_sample"], fit_n)
    lo, hi = SEEDS["resamples"]
    fit17_n = {JAN17: n_fit // 2, FEB17: n_fit // 2}
    fit_2017 = draw(rows, SEEDS["fit_2017"], fit17_n)
    return Samples(
        fit=fit,
        evaluation=draw(rows, SEEDS["evaluation_set"], e_n, exclude=fit),
        resamples=[draw(rows, k, fit_n) for k in range(lo, hi + 1)],
        month_fit={
            OCT16: draw(rows, SEEDS["month_fit_2016_10"], {OCT16: n_fit}),
            NOV16: draw(rows, SEEDS["month_fit_2016_11"], {NOV16: n_fit}),
        },
        fit_2017=fit_2017,
        evaluation_2017=draw(rows, SEEDS["evaluation_set_2017"], e17_n, exclude=fit_2017),
        diagnostic=draw(
            rows, SEEDS["expiry_diagnostic"], {OCT16: acfg["expiry_diagnostic"]["rows"]}
        ),
    )


# --- fitting and the vote ----------------------------------------------------------------------


@dataclass
class Fit:
    sample: np.ndarray  # row indices of the fit sample
    x: np.ndarray  # the fit sample's representation
    labels: np.ndarray  # HDBSCAN labels, NOISE for noise
    nn: NearestNeighbors

    @property
    def n_clusters(self) -> int:
        return int(len(set(self.labels.tolist()) - {NOISE}))


def fit_hdbscan(
    x: np.ndarray, sample: np.ndarray, acfg: dict[str, Any], min_cluster_size: int, threads: int = 4
) -> Fit:
    c = acfg["clustering"]
    labels = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(c["min_samples"]),
        metric=c["metric"],
        cluster_selection_method=c["cluster_selection_method"],
        n_jobs=threads,
        copy=True,
    ).fit_predict(x[sample])
    k = int(c["assignment"]["k"])
    nn = NearestNeighbors(n_neighbors=k, n_jobs=threads).fit(x[sample])
    return Fit(sample, x[sample], labels.astype(int), nn)


def vote(fit: Fit, x: np.ndarray) -> np.ndarray:
    """Majority label of the k nearest fit-sample rows (noise counts); a tie goes to the label of
    the nearest row among the tied labels."""
    if len(x) == 0:
        return np.zeros(0, int)
    _, idx = fit.nn.kneighbors(x)
    lab = fit.labels[idx] + 1  # 0 = noise
    n, k = lab.shape
    m = int(fit.labels.max()) + 2
    counts = np.zeros((n, m), int)
    first = np.full((n, m), k, int)
    rows = np.arange(n)
    for j in range(k - 1, -1, -1):
        counts[rows, lab[:, j]] += 1
        first[rows, lab[:, j]] = j
    score = counts * (k + 1) - first
    return score.argmax(axis=1) - 1


def assign(fit: Fit, x: np.ndarray, rows_idx: np.ndarray) -> np.ndarray:
    """Labels for rows_idx: HDBSCAN label for rows in the fit sample, the vote for the others."""
    out = np.empty(len(rows_idx), int)
    pos = {r: i for i, r in enumerate(fit.sample.tolist())}
    in_fit = np.array([r in pos for r in rows_idx.tolist()], bool)
    out[in_fit] = fit.labels[[pos[r] for r in rows_idx[in_fit].tolist()]]
    out[~in_fit] = vote(fit, x[rows_idx[~in_fit]])
    return out


# --- comparing clusterings ---------------------------------------------------------------------


def jaccard_matrix(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    """Jaccard overlap of every cluster of a (rows) with every cluster of b (noise excluded)."""
    ca = sorted(set(a.tolist()) - {NOISE})
    cb = sorted(set(b.tolist()) - {NOISE})
    m = np.zeros((len(ca), len(cb)))
    for i, u in enumerate(ca):
        au = a == u
        for j, v in enumerate(cb):
            bv = b == v
            union = int((au | bv).sum())
            m[i, j] = (au & bv).sum() / union if union else 0.0
    return m, ca, cb


def match(a: np.ndarray, b: np.ndarray) -> dict[int, tuple[int | None, float]]:
    """One-to-one (Hungarian) match of every cluster of a to a cluster of b, maximising Jaccard;
    a cluster of a left without a partner gets (None, 0.0)."""
    m, ca, cb = jaccard_matrix(a, b)
    out: dict[int, tuple[int | None, float]] = {u: (None, 0.0) for u in ca}
    if m.size:
        ri, ci = linear_sum_assignment(-m)
        for i, j in zip(ri, ci, strict=True):
            out[ca[i]] = (cb[j], float(m[i, j]))
    return out


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index with noise as one more label."""
    return float(adjusted_rand_score(a, b))


# --- stage 1: one fit, timed -------------------------------------------------------------------


def one_fit_timing(
    rows: Rows, samples: Samples, acfg: dict[str, Any], min_cluster_size: int, threads: int = 4
) -> dict[str, Any]:
    t0 = time.time()
    fit = fit_hdbscan(rows.x, samples.fit, acfg, min_cluster_size, threads)
    t_fit = time.time() - t0
    others = np.setdiff1d(np.flatnonzero(~rows.rule), samples.fit, assume_unique=True)
    t1 = time.time()
    voted = vote(fit, rows.x[others])
    t_vote = time.time() - t1
    return {
        "min_cluster_size": min_cluster_size,
        "fit_rows": len(samples.fit),
        "fit_seconds": round(t_fit, 1),
        "voted_rows": len(others),
        "vote_seconds": round(t_vote, 1),
        "peak_rss_gb": peak_rss_gb(),
        "clusters": fit.n_clusters,
        "noise_share_fit": float((fit.labels == NOISE).mean()),
        "noise_share_voted": float((voted == NOISE).mean()),
    }


# --- stage 2: grid selection -------------------------------------------------------------------


@dataclass
class Setting:
    min_cluster_size: int
    reference: Fit
    e_labels: np.ndarray
    resample_e_labels: list[np.ndarray]
    aris: list[float]
    qualifies: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def median_ari(self) -> float:
        return float(np.median(self.aris))

    def summary(self) -> dict[str, Any]:
        return {
            "min_cluster_size": self.min_cluster_size,
            "archetypes": self.reference.n_clusters,
            "unassigned_share_on_E": float((self.e_labels == NOISE).mean()),
            "median_ari": self.median_ari,
            "ari": self.aris,
            "qualifies": self.qualifies,
            "reasons": self.reasons,
        }


def evaluate_setting(
    x: np.ndarray, samples: Samples, acfg: dict[str, Any], size: int, threads: int = 4
) -> Setting:
    sel = acfg["clustering"]["selection"]
    ref = fit_hdbscan(x, samples.fit, acfg, size, threads)
    e = vote(ref, x[samples.evaluation])
    res = [
        vote(fit_hdbscan(x, s, acfg, size, threads), x[samples.evaluation])
        for s in samples.resamples
    ]
    aris = [ari(e, r) for r in res]
    reasons = []
    if not sel["min_archetypes"] <= ref.n_clusters <= sel["max_archetypes"]:
        reasons.append(
            f"{ref.n_clusters} archetypes, outside {sel['min_archetypes']} to "
            f"{sel['max_archetypes']}"
        )
    if (e == NOISE).mean() > sel["max_unassigned_share"]:
        reasons.append(
            f"unassigned on E {(e == NOISE).mean():.3f} above {sel['max_unassigned_share']}"
        )
    return Setting(size, ref, e, res, aris, not reasons, reasons)


def choose(settings: list[Setting]) -> Setting | None:
    """Highest median ARI among qualifying settings; a tie goes to the larger min_cluster_size."""
    ok = [s for s in settings if s.qualifies]
    if not ok:
        return None
    return max(ok, key=lambda s: (s.median_ari, s.min_cluster_size))


def select_setting(
    x: np.ndarray, samples: Samples, acfg: dict[str, Any], threads: int = 4
) -> tuple[Setting | None, list[Setting]]:
    settings = [
        evaluate_setting(x, samples, acfg, int(m), threads)
        for m in acfg["clustering"]["min_cluster_size_grid"]
    ]
    return choose(settings), settings


# --- stage 3: the expiry diagnostic -----------------------------------------------------------


def ami_with_expiry(labels: np.ndarray, days: np.ndarray) -> float:
    """Adjusted mutual information between labels (noise as a label) and expiry day categories."""
    return float(adjusted_mutual_info_score(days.astype(int), labels))


def expiry_diagnostic(
    rows: Rows, samples: Samples, acfg: dict[str, Any], chosen: Setting, threads: int = 4
) -> dict[str, Any]:
    size = chosen.min_cluster_size
    diag = samples.diagnostic
    days = rows.days_until_expiry[diag]
    ref_labels = assign(chosen.reference, rows.x, diag)
    oct_fit = fit_hdbscan(rows.x, samples.month_fit[OCT16], acfg, size, threads)
    a30 = vote(oct_fit, rows.x[diag])
    fit7 = fit_hdbscan(rows.x7, samples.month_fit[OCT16], acfg, size, threads)
    a7 = vote(fit7, rows.x7[diag])
    out = {
        "rows": len(diag),
        "AMI_30_reference": ami_with_expiry(ref_labels, days),
        "AMI_30": ami_with_expiry(a30, days),
        "AMI_7": ami_with_expiry(a7, days),
        "clusters_30_oct_fit": oct_fit.n_clusters,
        "clusters_7_oct_fit": fit7.n_clusters,
    }
    b = acfg["expiry_diagnostic"]["branches"]
    out["restates_expiry"] = out["AMI_30_reference"] >= 0.5
    out["assumption_unsupported"] = out["AMI_7"] < out["AMI_30"] + 0.10
    out["branches"] = {k: v for k, v in b.items()}
    return out


# --- stage 4: the criteria ---------------------------------------------------------------------


def profile(x: np.ndarray, labels: np.ndarray, archetypes: list[int]) -> dict[int, np.ndarray]:
    return {
        a: x[labels == a].mean(axis=0) if (labels == a).any() else np.full(x.shape[1], np.nan)
        for a in archetypes
    }


def profile_ok(ref: np.ndarray, month: np.ndarray, crit: dict[str, Any]) -> tuple[bool, dict]:
    if np.isnan(month).any():
        return False, {"reason": "no rows in this month"}
    if np.linalg.norm(ref) < crit["near_zero_norm"]:
        dist = float(np.linalg.norm(month - ref))
        return dist <= crit["near_zero_max_distance"], {"rule": "distance", "distance": dist}
    cos = float(ref @ month / (np.linalg.norm(ref) * np.linalg.norm(month)))
    same_top = int(np.argmax(np.abs(ref))) == int(np.argmax(np.abs(month)))
    return cos >= crit["min_cosine"] and same_top, {
        "rule": "cosine",
        "cosine": cos,
        "same_largest_group": same_top,
    }


def share_ok(ref: float, month: float, crit: dict[str, Any]) -> tuple[bool, dict]:
    diff = abs(month - ref) * 100
    ratio = month / ref if ref > 0 else math.inf
    lo, hi = crit["ratio_band"]
    ok_abs = diff <= crit["max_absolute_difference_points"]
    ok_ratio = lo <= ratio <= hi
    ok_min = month >= crit["min_share_every_month"]
    return ok_abs and ok_ratio and ok_min, {
        "difference_points": diff,
        "ratio": ratio,
        "absolute_ok": ok_abs,
        "ratio_ok": ok_ratio,
        "minimum_ok": ok_min,
    }


def judge(
    rows: Rows, samples: Samples, acfg: dict[str, Any], chosen: Setting, threads: int = 4
) -> dict[str, Any]:
    crit = acfg["criteria"]
    ref = chosen.reference
    arche = sorted(set(ref.labels.tolist()) - {NOISE})
    size = chosen.min_cluster_size
    out: dict[str, Any] = {"min_cluster_size": size, "archetypes": arche}

    # 1. resamples
    per = {a: [] for a in arche}
    for r in chosen.resample_e_labels:
        for a, (_, j) in match(chosen.e_labels, r).items():
            per[a].append(j)
    c1 = {
        a: {
            "median_jaccard": float(np.median(v)),
            "passed": float(np.median(v)) >= crit["1_resamples"]["min_median_jaccard"],
        }
        for a, v in per.items()
    }
    # 2. the two 2016 months
    c2 = {a: {} for a in arche}
    for t, s in samples.month_fit.items():
        mfit = fit_hdbscan(rows.x, s, acfg, size, threads)
        m = match(chosen.e_labels, vote(mfit, rows.x[samples.evaluation]))
        for a, (_, j) in m.items():
            c2[a][t.isoformat()] = j
    for a in arche:
        c2[a]["passed"] = all(
            v >= crit["2_two_2016_months"]["min_jaccard_each_month"]
            for k, v in c2[a].items()
            if k != "passed"
        )
    # labels for every row of every explained month
    labels = np.full(len(rows.msno), RULE, int)
    non_rule = np.flatnonzero(~rows.rule)
    labels[non_rule] = assign(ref, rows.x, non_rule)
    ref_profile = profile(ref.x, ref.labels, arche)
    months = sorted(set(rows.snapshot.tolist()))
    # 3. profiles
    c3 = {a: {} for a in arche}
    for t in months:
        m = rows.snapshot == t
        mp = profile(rows.x[m], labels[m], arche)
        for a in arche:
            ok, info = profile_ok(ref_profile[a], mp[a], crit["3_shap_profile"])
            c3[a][t.isoformat()] = {"passed": ok, **info}
    for a in arche:
        c3[a]["passed"] = all(v["passed"] for k, v in c3[a].items() if k != "passed")
    # 4. shares (all P2 main rows, rule rows included in the denominator)
    pooled = np.isin(rows.snapshot, [OCT16, NOV16])
    ref_share = {a: float((labels[pooled] == a).mean()) for a in arche}
    shares = {
        t.isoformat(): {
            str(a): float((labels[rows.snapshot == t] == a).mean()) for a in [*arche, NOISE, RULE]
        }
        for t in months
    }
    c4 = {a: {} for a in arche}
    for t in months:
        for a in arche:
            ok, info = share_ok(ref_share[a], shares[t.isoformat()][str(a)], crit["4_shares"])
            c4[a][t.isoformat()] = {"passed": ok, **info}
    for a in arche:
        c4[a]["passed"] = all(v["passed"] for k, v in c4[a].items() if k != "passed")
    # 5. re-found in 2017
    fit17 = fit_hdbscan(rows.x, samples.fit_2017, acfg, size, threads)
    e17_ref = vote(ref, rows.x[samples.evaluation_2017])
    m17 = match(e17_ref, vote(fit17, rows.x[samples.evaluation_2017]))
    c5 = {
        a: {
            "jaccard": m17.get(a, (None, 0.0))[1],
            "passed": m17.get(a, (None, 0.0))[1] >= crit["5_refound_in_2017"]["min_jaccard"],
        }
        for a in arche
    }
    # 6. unassigned, reported three ways
    in_fit = np.zeros(len(rows.msno), bool)
    in_fit[ref.sample] = True
    c6 = {}
    for t in months:
        m = rows.snapshot == t
        nr = m & ~rows.rule
        c6[t.isoformat()] = {
            "non_rule_rows": float((labels[nr] == NOISE).mean()),
            "fit_sample_rows": float((labels[nr & in_fit] == NOISE).mean())
            if (nr & in_fit).any()
            else None,
            "voted_rows": float((labels[nr & ~in_fit] == NOISE).mean()),
            "all_rows_with_rule": float((labels[m] == NOISE).mean()),
            "passed": float((labels[nr] == NOISE).mean())
            <= crit["6_unassigned"]["max_share_every_month"],
        }
    c7 = {
        "archetypes": len(arche),
        "passed": crit["7_count"]["min"] <= len(arche) <= crit["7_count"]["max"],
    }
    holds = {a: all(c[a]["passed"] for c in (c1, c2, c3, c4, c5)) for a in arche}
    out |= {
        "1_resamples": c1,
        "2_two_2016_months": c2,
        "3_shap_profile": c3,
        "4_shares": c4,
        "5_refound_in_2017": c5,
        "6_unassigned": c6,
        "7_count": c7,
        "holds_up": holds,
        "set_holds_up": all(holds.values())
        and all(v["passed"] for v in c6.values())
        and c7["passed"],
        "reference_profile": {
            a: dict(zip(rows.groups, ref_profile[a].tolist(), strict=True)) for a in arche
        },
        "reference_share": ref_share,
        "shares": shares,
        "share_note": "2017 shares and profiles are assignments from a 2016 definition; the "
        "2017-02 share may move for a Lunar New Year reason (2017-01-28 falls in "
        "its 28-day listening window).",
    }
    out["_labels"] = labels
    return out


def apply_failure_rule(judgement: dict[str, Any], rows: Rows) -> dict[str, Any]:
    """As registered: an archetype that does not hold up is not named and its rows count as
    unassigned in the share series; then criteria 6 and 7 are read again."""
    labels = judgement["_labels"].copy()
    failed = [a for a, ok in judgement["holds_up"].items() if not ok]
    labels[np.isin(labels, failed)] = NOISE
    months = sorted(set(rows.snapshot.tolist()))
    kept = [a for a, ok in judgement["holds_up"].items() if ok]
    return {
        "failed_archetypes": failed,
        "kept_archetypes": kept,
        "unassigned_after": {
            t.isoformat(): float((labels[(rows.snapshot == t) & ~rows.rule] == NOISE).mean())
            for t in months
        },
        "count_after": len(kept),
        "_labels": labels,
    }


def without_group(rows: Rows, group: str) -> Rows:
    """The same rows with one group removed from the representation (the registered branch)."""
    keep = [i for i, g in enumerate(rows.groups) if g != group]
    return Rows(
        rows.msno,
        rows.snapshot,
        rows.rule,
        rows.x[:, keep],
        [rows.groups[i] for i in keep],
        rows.days_until_expiry,
        rows.x7[:, keep] if rows.x7 is not None else None,
    )


def labels_sha256(labels: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(labels, dtype=np.int64).tobytes()).hexdigest()
