"""The service layer: the batch path, stored rows, requests and readiness, on hand-made models.

The models are small and fitted here on the real 40 feature columns, so the real
reason config, phrases, actions config and store layout are used unchanged. They
are registered in a temporary MLflow store and loaded through the alias, as the
service loads them. Their file hashes differ from the registered bundle, so the
tests that build responses swap the bundle the response schemas check for the
hand-made one (a test double; configs/serving.yaml is untouched), and one test
shows that without the swap the context is refused.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import redis
import yaml
from pydantic import ValidationError

from retention.api import dt01 as d
from retention.api import registry as reg
from retention.api import schemas as sch
from retention.api import scoring as sc
from retention.api import service as svc
from retention.api import store as st
from retention.api.config import file_sha256
from retention.api.labels import LABELS
from retention.api.parity import compare_frames
from retention.api.saved import ModelRefused
from retention.explain.model import Refit, Settings, booster_sha256, fit_models, save_models
from retention.models.discrete_time import DiscreteTimeModel
from retention.models.protocol import load_protocol
from retention.optimize.value import INTERVAL_ENDS_ALL
from test_explain_build_reasons import make_facts

P = load_protocol()
RAW = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
REAL = svc.load_settings(horizons=svc.C01_HORIZONS)
COLUMNS = list(REAL.columns)
SMALL = {"num_leaves": 7, "n_estimators": 20, "learning_rate": 0.1, "min_child_samples": 20}
MODEL_SETTINGS = Settings(
    columns=COLUMNS,
    categorical=list(REAL.categorical),
    fixed=RAW["classifier"]["fixed_params"],
    candidates={h: SMALL for h in svc.C01_HORIZONS},
)
NOV, DEC = date(2016, 11, 1), date(2016, 12, 1)
LEVELS = {
    "last_payment_method_id": [36, 38, 41],
    "registered_via": [3, 7, 9],
    "segment": ["paid", "discount100_only"],
    "paid_state": ["paid", "unpaid"],
}
N_SCORED = 60
N_RANDOM, RANDOM_SEED = 2000, 2026  # the random 2,000-row set of the equality test
UNKNOWN = "never-transacted"


class Dt01Columns:
    """The part of Dt01Settings that dt01's model JSON records."""

    columns = COLUMNS


# --- hand-made rows and models -------------------------------------------------------------------


def frame(n: int, seed: int, prefix: str = "c") -> pd.DataFrame:
    """n rows of the 40 columns, typed as the batch loader types them, plus msno."""
    rng = np.random.default_rng(seed)
    msno = pd.Series([f"{prefix}{i:03d}" for i in range(n)])
    facts = make_facts(msno, seed)
    schema = st.schema_for(COLUMNS)
    data: dict[str, list] = {}
    for c in COLUMNS:
        t = schema.field(c).type
        if c in facts:
            data[c] = [st.to_json_value(v) for v in facts[c].astype(object)]
        elif c in LEVELS:
            data[c] = [st.to_json_value(v) for v in rng.choice(LEVELS[c], n)]
        elif c == "days_until_paid_expiry":
            data[c] = [int(v) for v in rng.integers(0, 31, n)]
        elif pa.types.is_floating(t):
            data[c] = [None if rng.random() < 0.1 else float(rng.normal()) for _ in range(n)]
        elif pa.types.is_integer(t):
            hi = 2 if t == pa.int8() else 60
            data[c] = [None if rng.random() < 0.1 else int(rng.integers(0, hi)) for _ in range(n)]
        else:
            data[c] = [str(v) for v in rng.choice(["a", "b"], n)]
    records = [{c: data[c][i] for c in COLUMNS} for i in range(n)]
    rows = st.rows_frame(records, COLUMNS)
    rows.insert(0, "msno", msno.to_numpy())
    return rows


def outcome(rows: pd.DataFrame, shift: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    days = rows.days_until_expiry.astype("float64").fillna(15).to_numpy()
    auto = rows.last_is_auto_renew.astype("float64").fillna(0).to_numpy()
    z = shift - 0.06 * days - 1.4 * auto
    return (rng.random(len(rows)) < 1 / (1 + np.exp(-z))).astype(int)


def write_bundle(out: Path) -> dict[str, str]:
    """c01 at 7, 30 and 59 days and dt01, fitted on hand-made rows; returns the hashes."""
    for k, h in enumerate(svc.C01_HORIZONS):
        fit, cal = frame(1500, 10 + k), frame(600, 20 + k)
        fit["y"], cal["y"] = outcome(fit, k - 1.0, k), outcome(cal, k - 1.0, 50 + k)
        model, platt = fit_models(fit, cal, MODEL_SETTINGS, h, P)
        save_models(out, {h: Refit(h, model, platt, 1500, 600)}, {h: booster_sha256(model)})
    surv = frame(1500, 30)
    lapse = outcome(surv, 0.0, 31).astype(bool)
    surv["time"] = np.where(lapse, surv.days_until_expiry.astype("float64").fillna(10), 90)
    surv["event"] = lapse.astype(int)
    dt = DiscreteTimeModel(
        False,
        list(INTERVAL_ENDS_ALL),
        SMALL,
        RAW["classifier"]["fixed_params"],
        RAW["classifier"]["categorical"],
        P.threads,
        P.seed,
    ).fit(surv, COLUMNS)
    d.save(dt, Dt01Columns, out, d.text_sha256(d.booster_text(dt)))
    return {n: file_sha256(out / n) for n in sorted(LABELS.bundle)}


@pytest.fixture(scope="module")
def registered(tmp_path_factory) -> reg.RegistrySettings:
    """The hand-made bundle registered once in a temporary MLflow store."""
    tmp = tmp_path_factory.mktemp("registry")
    hashes = write_bundle(tmp / "src")
    settings = reg.RegistrySettings(
        name="test-scoring",
        alias="serving",
        bundle=hashes,
        tracking_uri=f"sqlite:///{tmp / 'mlflow.db'}",
        artifact_root=tmp / "artifacts",
    )
    before = mlflow.get_tracking_uri()
    reg.register(settings, {n: tmp / "src" / n for n in hashes}, tmp / "staging", "abc", "f" * 64)
    yield settings
    mlflow.set_tracking_uri(before)


@pytest.fixture(scope="module")
def bundle_labels(registered):
    """The response schemas check the hand-made bundle instead of the registered one."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sch, "LABELS", dataclasses.replace(sch.LABELS, bundle=registered.bundle))
        yield


@pytest.fixture(scope="module")
def ctx(registered, bundle_labels, tmp_path_factory) -> svc.Context:
    c = svc.load_context(registered, MODEL_SETTINGS, P, tmp_path_factory.mktemp("work"))
    yield c
    c.close()


# --- one hand-made snapshot ----------------------------------------------------------------------


@dataclasses.dataclass
class Snapshot:
    t: date
    rows: pd.DataFrame  # the scored customers, batch-typed, with snapshot_date
    prices: pd.DataFrame
    status: pd.DataFrame
    known: list[str]
    finals: pd.DataFrame
    counts: dict[str, int]


def snapshot(ctx: svc.Context, t: date = NOV, n: int = N_SCORED, seed: int = 7) -> Snapshot:
    rows = frame(n, seed)
    rows["snapshot_date"] = pd.Timestamp(t)
    has = np.arange(n) % 7 != 3  # every seventh customer has no usable price
    prices = pd.DataFrame(
        {
            "msno": rows.msno,
            "snapshot_date": pd.Timestamp(t),
            "price_per_day": np.where(has, np.resize([4.97, 4.3, 3.3], n), 0.0),
            "has_price": has,
            "price_row_date": [date(2016, 10, 2) if h else None for h in has],
        }
    )
    outside = pd.DataFrame(
        {
            "msno": ["o-trial", "o-beyond", "o-unseen"],
            "status": sc.OUTSIDE,
            "code": [sc.TRIAL, sc.BEYOND, "unseen_level:last_payment_method_id"],
            "next_expiry_in_days": [12, 45, 3],
        }
    )
    scored = pd.DataFrame(
        {"msno": rows.msno, "status": sc.SCORED, "code": None, "next_expiry_in_days": 5}
    )
    status = pd.concat([scored, outside], ignore_index=True)
    status["next_expiry_in_days"] = status.next_expiry_in_days.astype("Int32")
    known = sorted([*status.msno, "n-inactive-1", "n-inactive-2"])
    pr = sc.probabilities(ctx.models.c01, rows, P.seed)
    finals = pd.DataFrame({"msno": rows.msno.astype(str)})
    for k, h in enumerate(svc.C01_HORIZONS):
        finals[f"days_{h}"] = pr.final[:, k]
    counts = sc.population_counts(status) | {"not_active": len(known) - len(status)}
    return Snapshot(t, rows, prices, status, known, finals, counts)


def load(r, s: Snapshot) -> None:
    st.load_snapshot(r, s.t, s.status, s.rows, s.prices, COLUMNS, s.known, s.finals, s.counts)


# --- the batch path ------------------------------------------------------------------------------


def test_scored_bodies_equal_the_step_4_functions_on_the_batch(ctx):
    s = snapshot(ctx)
    bodies = svc.scored_bodies(ctx, s.t, s.rows, s.prices)
    pr = sc.probabilities(ctx.models.c01, s.rows, P.seed)
    facts = s.rows[["msno", *ctx.fact_columns]]
    served = {}
    for h in (7, 30):
        part = sc.contribution_part(ctx.models.c01[h], s.rows, ctx.explain_cfg, COLUMNS, h)
        platt = (ctx.models.c01[h].platt_a, ctx.models.c01[h].platt_b)
        served[h] = sc.service_reasons(
            ctx.explain_cfg, ctx.renderer, part, facts, h, s.t, platt, P, ctx.phrases_sha256
        )
    table = sc.customers(s.rows, sc.survival(ctx.models.dt01, s.rows), served[30], s.prices)
    blocks = sc.action_blocks(sc.actions(table, ctx.actions_cfg), ctx.actions_cfg.base)
    assert [b["msno"] for b in bodies] == s.rows.msno.tolist()
    for i, b in enumerate(bodies):
        got = [b["risk"][f"days_{h}"]["probability"] for h in (7, 30, 59)]
        assert got == pr.final[i].tolist()
        assert b["risk"]["moved_by_ordering_fix"] == bool(pr.moved[i])
        for k, h in enumerate((7, 30)):
            assert b["reasons"][k]["text"] == served[h].text.iloc[i]
            assert b["reasons"][k]["probability_before_fix"] == served[h].probability.iloc[i]
        assert b["paid_days"]["value"] == table.days_30_90.iloc[i]
        assert b["action"] == sch.ActionBlock(**blocks[i]).model_dump(mode="json")
        assert b["model"]["files"] == ctx.models.files and b["model"]["version"] == 1
        assert b["risk"]["days_59"]["grade"] == "demonstration"
    assert any(b["risk"]["moved_by_ordering_fix"] for b in bodies)  # the fix is exercised
    assert any(b["action"].get("price") == "no usable price" for b in bodies)


def one(s: Snapshot, i: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return s.rows.iloc[[i]].reset_index(drop=True), s.prices.iloc[[i]].reset_index(drop=True)


def test_score_one_equals_its_row_in_the_batch_for_every_customer(ctx):
    s = snapshot(ctx)
    batch = svc.scored_bodies(ctx, s.t, s.rows, s.prices)
    differ = [i for i in range(N_SCORED) if svc.score_one(ctx, s.t, *one(s, i)) != batch[i]]
    assert differ == []


def test_the_workaround_rests_on_numpy_summing_one_row_differently(ctx):
    """D-199's evidence: a one-row array alone still differs from the batch on these rows,
    and the copy in the pair is what removes the difference. If NumPy changes this,
    this test fails and the workaround is to be reviewed."""
    from retention.optimize.value import expected_days

    s = snapshot(ctx)
    surv = sc.survival(ctx.models.dt01, s.rows)
    batch = expected_days(surv).days_30_90.to_numpy()
    alone = np.array([expected_days(surv[[i]]).days_30_90.iloc[0] for i in range(N_SCORED)])
    paired = np.array([expected_days(surv[[i, i]]).days_30_90.iloc[0] for i in range(N_SCORED)])
    assert int((alone != batch).sum()) == 16
    assert int((paired != batch).sum()) == 0


def test_score_one_refuses_the_placeholder_and_more_than_one_row(ctx):
    s = snapshot(ctx)
    rows, prices = one(s, 0)
    with pytest.raises(ValueError, match="placeholder"):
        svc.score_one(ctx, s.t, rows.assign(msno=svc.PAIR_MSNO), prices)
    with pytest.raises(ValueError, match="one row"):
        svc.score_one(ctx, s.t, s.rows.head(2), s.prices.head(2))


def test_december_2016_withholds_reasons_and_warns(ctx):
    s = snapshot(ctx, DEC)
    for b in svc.scored_bodies(ctx, s.t, s.rows.head(5), s.prices):
        assert b["reasons"] is None
        assert b["reasons_withheld"] == LABELS.reasons_withheld_code
        assert b["data_warning"] == LABELS.december_warning


@pytest.mark.valkey
def test_a_stored_row_rebuilds_the_batch_row_exactly(ctx, valkey):
    s = snapshot(ctx)
    load(valkey, s)
    for i in (0, 1, 4, 17):
        m = s.rows.msno.iloc[i]
        rows, prices = svc.stored_inputs(m, st.customer(valkey, m), s.t)
        want = s.rows.iloc[[i]].reset_index(drop=True)[rows.columns]
        assert compare_frames(rows, want)["equal"]
        assert rows.dtypes.tolist() == want.dtypes.tolist()
        assert prices.price_per_day.iloc[0] == s.prices.price_per_day.iloc[i]


def test_files_other_than_the_registered_bundle_are_refused(registered, tmp_path, monkeypatch):
    monkeypatch.setattr(sch, "LABELS", LABELS)  # the registered bundle, for this test only
    assert registered.bundle != LABELS.bundle
    with pytest.raises(ModelRefused, match="registered bundle"):
        svc.load_context(registered, MODEL_SETTINGS, P, tmp_path / "work")


# --- requests ------------------------------------------------------------------------------------


@pytest.mark.parametrize("body", [{"msno": "x"}, {"msno": "x", "as_of": "2016-11-01"}])
def test_a_well_formed_request_is_accepted(body):
    assert svc.ScoreRequest(**body).msno == "x"


MALFORMED = [
    {},
    {"msno": ""},
    {"msno": 12},
    {"msno": None},
    {"msno": "x", "as_of": "2016-13-01"},
    {"msno": "x", "as_of": "20161101"},
    {"msno": "x", "as_of": "2016-11-01T00:00:00"},
    {"msno": "x", "as_of": 1477958400},
    {"msno": "x", "city": "1"},
]


@pytest.mark.parametrize("body", MALFORMED)
def test_a_malformed_request_is_refused(body):
    with pytest.raises(ValidationError):
        svc.ScoreRequest(**body)


# --- readiness -----------------------------------------------------------------------------------


def unreachable() -> redis.Redis:
    """A client on a port nothing listens on (no container is touched)."""
    return redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2)


@pytest.mark.valkey
def test_health_is_ready_only_with_matching_files_a_store_date_and_valkey(ctx, valkey):
    h = svc.health(ctx, valkey)
    assert not h.ready and h.reason == "no snapshot is loaded (the store has no date)"
    load(valkey, snapshot(ctx))
    h = svc.health(ctx, valkey)
    assert h.ready and h.reason is None and h.store_date == NOV
    assert h.replay_progress.model_dump() == {
        "produced": 0,
        "consumed": 0,
        "expected_events": 3727472,
    }
    assert h.model_checksums_match and h.valkey_reachable
    assert h.config_sha256 == file_sha256(Path("configs/serving.yaml"))


def test_health_names_valkey_when_it_cannot_be_reached(ctx):
    h = svc.health(ctx, unreachable())
    assert not h.ready and not h.valkey_reachable
    assert h.reason == "Valkey cannot be reached"
    assert h.store_date is None and h.replay_progress is None
    with pytest.raises(svc.NotReady, match="Valkey cannot be reached"):
        svc.ready_date(ctx, unreachable())


# --- amendment A5: unseen-level codes ------------------------------------------------------------


def test_a5_and_a6_registered_values_are_read():
    assert LABELS.unseen_generic_code == "unseen_level:hidden_feature"
    assert LABELS.msno_exempt_field == "msno"


def test_a5_maps_a_never_used_feature_to_the_generic_code():
    assert sc.unseen_code("registered_via") == "unseen_level:hidden_feature"
    assert sc.unseen_code("last_payment_method_id") == "unseen_level:last_payment_method_id"


def test_a5_through_refuse_unseen_on_the_hand_made_models(ctx):
    rows = frame(3, 41)
    rows.loc[0, "registered_via"] = 13  # a level the models never saw
    rows.loc[1, "last_payment_method_id"] = 99
    status = sc.population_status(
        pd.DataFrame(
            {"msno": rows.msno, "in_exp30": True, "segment": "paid", "days_until_expiry": 4}
        ),
        tuple(P.main_segments),
    )
    status, keep = sc.refuse_unseen(status, rows, [m.model for m in ctx.models.c01.values()])
    assert status.code.tolist() == [
        "unseen_level:hidden_feature",
        "unseen_level:last_payment_method_id",
        None,
    ]
    assert keep.tolist() == [False, False, True]


def test_a5_an_outside_code_naming_a_never_used_attribute_cannot_be_built():
    base = {"status": sc.OUTSIDE, "msno": "m", "features_as_of": NOV, "next_expiry_in_days": 3}
    assert sch.Outside(code="unseen_level:hidden_feature", **base).code.endswith("hidden_feature")
    with pytest.raises(ValidationError, match="never shown"):
        sch.Outside(code="unseen_level:registered_via", **base)


# --- amendment A6: the msno identifier exemption -------------------------------------------------

OWN = "x/age+01"  # an ID holding a never-used word as a whole word


def body(**fields) -> str:
    return json.dumps(fields)


def test_a6_exempts_only_the_top_level_msno_equal_to_the_own_id():
    assert sch.never_used_hits(body(status="scored", msno=OWN)) == ["age"]
    assert sch.response_hits(body(status="scored", msno=OWN), 200, OWN) == []


def test_a6_scans_the_same_string_in_any_other_field_and_key():
    assert sch.response_hits(body(msno=OWN, code=OWN), 200, OWN) == ["age"]
    assert sch.response_hits(body(msno=OWN, row={"msno": OWN}), 200, OWN) == ["age"]
    assert sch.response_hits(body(msno=OWN, age=1), 200, OWN) == ["age"]
    assert sch.response_hits(body(msno=OWN, reasons=[OWN]), 200, OWN) == ["age"]


def test_a6_scans_another_customer_s_id_in_msno():
    assert sch.response_hits(body(msno=OWN), 200, "c000") == ["age"]
    assert sch.response_hits(body(msno=OWN), 200, None) == ["age"]
    assert sch.response_hits(body(msno=OWN.upper()), 200, OWN) == ["AGE"]


@pytest.mark.parametrize("status", [404, 409, 422, 500, 503])
def test_a6_scans_every_error_body_in_full(status):
    assert sch.response_hits(body(msno=OWN), status, OWN) == ["age"]


@pytest.mark.parametrize("text", [json.dumps([OWN]), json.dumps(OWN), f"plain {OWN}"])
def test_a6_scans_a_body_that_is_not_a_json_object_in_full(text):
    assert sch.response_hits(text, 200, OWN) == ["age"]
