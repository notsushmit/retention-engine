"""GET /summary, /plans and /model (configs/serving.yaml, dashboard_endpoints).

On the hand-made models, store and pinned files of test_service: every number equals
a batch computation on the same hand-made data, every label is the registered one,
the 59-day bucket carries "demonstration", small cells are suppressed and counted,
and no body names a never-used attribute.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from retention.api import dashboard as db
from retention.api import schemas as sch
from retention.api import scoring as sc
from retention.api import service as svc
from retention.api.config import file_sha256, load_serving
from retention.api.labels import DASH, LABELS
from retention.api.schemas import never_used_hits
from test_app import flip_last_byte, make_client  # noqa: F401 (fixture)
from test_service import (
    CALIBRATION,
    COLUMNS,
    DEC,
    NOV,
    PLAN_VALUES,
    P,
    bundle_labels,  # noqa: F401 (fixture)
    ctx,  # noqa: F401 (fixture)
    load,
    pins,  # noqa: F401 (fixture)
    registered,  # noqa: F401 (fixture)
    snapshot,
    summary_rows,
    unreachable,
)

CFG = load_serving()["dashboard_endpoints"]
MAIN = tuple(P.main_segments)


# --- the registered values ------------------------------------------------------------------------


def test_the_registered_values_are_read():
    assert (DASH.top_n, DASH.min_cell_size) == (1000, 20)
    assert DASH.outside_label == LABELS_FIXED[1]
    assert DASH.dates == LABELS_FIXED[2]
    assert DASH.effect_prefix == "Under the assumed relative base effect"
    assert DASH.drift == "drift: not monitored (Phase 10)"
    assert DASH.no_group == "no group listed"


LABELS_FIXED = load_serving()["wording"]["fixed_disclaimers"]


def test_the_real_pinned_files_are_registered_beside_step5s():
    real = svc.registered_pins()
    assert sorted(real) == ["alerts", "calibration", "plans", "summary"]
    assert real["summary"][1] == CFG["pins"]["summary"]["sha256"]


@pytest.mark.parametrize("name", ["summary", "calibration"])
def test_the_real_dashboard_pins_have_their_registered_hash(name):
    path, sha = svc.registered_pins()[name]
    if not path.exists():
        pytest.skip(f"needs the local {path.name}")
    assert file_sha256(path) == sha


# --- batch computations on the hand-made snapshot -------------------------------------------------


def batch_order(finals: pd.DataFrame, h: int) -> list[str]:
    """The /at-risk order: final probability descending, ties by msno."""
    f = finals.assign(neg=-finals[f"days_{h}"]).sort_values(["neg", "msno"])
    return f.msno.tolist()


def batch_first_groups(c, s, h: int, msnos: list[str]) -> pd.DataFrame:
    rows = s.rows.set_index("msno").loc[msnos].reset_index()
    part = sc.contribution_part(c.models.c01[h], rows, c.explain_cfg, COLUMNS, h)
    facts = rows[["msno", *c.fact_columns]]
    platt = (c.models.c01[h].platt_a, c.models.c01[h].platt_b)
    served = sc.service_reasons(
        c.explain_cfg, c.renderer, part, facts, h, s.t, platt, P, c.phrases_sha256
    )
    return pd.DataFrame(
        {"segment": rows.segment.astype(str), "group": served.raising_1.fillna("no group listed")}
    )


def rankings_of(finals: pd.DataFrame) -> dict[int, list[tuple[str, float]]]:
    return {
        h: [(m, float(finals.set_index("msno").at[m, f"days_{h}"])) for m in batch_order(finals, h)]
        for h in (7, 30, 59)
    }


@pytest.fixture()
def summary_loaded(make_client, valkey):  # noqa: F811
    c = make_client()
    s = snapshot(c.app.state.ctx)
    load(valkey, s)
    return c, s


@pytest.mark.valkey
def test_summary_counts_and_buckets_equal_the_batch(summary_loaded):
    c, s = summary_loaded
    r = c.get("/summary")
    assert r.status_code == 200
    body = r.json()
    counts = body["counts"]
    assert counts["scored"] == s.counts["scored"] == len(s.rows)
    assert counts["not_active"] == s.counts["not_active"]
    outside = {
        k.removeprefix("outside_population_"): v for k, v in s.counts.items() if "outside" in k
    }
    assert counts["outside_population"] == outside
    active = counts["scored"] + sum(outside.values())
    assert counts["outside_share_of_active"] == outside[sc.BEYOND] / active
    assert counts["outside_label"] == LABELS_FIXED[1]
    f = s.finals.sort_values("msno")
    b = body["buckets"]
    assert b["population"] == len(f)
    assert b["days_1_7"] == math.fsum(f.days_7)
    assert b["days_8_30"] == math.fsum(f.days_30 - f.days_7)
    assert b["days_31_59"] == {
        "value": math.fsum(f.days_59 - f.days_30),
        "grade": "demonstration",
    }
    assert b["label"] == "expected counts from the final probabilities, scored customers only"


def tally(groups: pd.DataFrame, top_n: int, min_cell: int) -> dict:
    out = {}
    for seg in ("all", *MAIN):
        part = groups if seg == "all" else groups[groups.segment == seg]
        counts = part.group.value_counts()
        out[seg] = (
            None
            if len(part) < min_cell
            else {
                g: (int(n), int(n) / len(part)) if n >= min_cell else None
                for g, n in counts.items()
            }
        )
    return out


def served_tally(reasons: dict) -> dict:
    out = {}
    for seg in reasons["segments"]:
        out[seg["segment"]] = (
            None
            if seg["suppressed"]
            else {
                c["group"]: None if c["suppressed"] else (c["customers"], c["share"])
                for c in seg["cells"]
            }
        )
    return out


@pytest.mark.valkey
@pytest.mark.parametrize("top_n,min_cell", [(1000, 20), (40, 5), (25, 1)])
def test_reason_shares_equal_the_batch_reasons_of_the_top_n(
    summary_loaded,
    ctx,  # noqa: F811
    top_n,
    min_cell,
):
    c, s = summary_loaded
    app_ctx = c.app.state.ctx
    body = db.summary_body(app_ctx, c.app.state.valkey, s.t, rankings_of(s.finals), top_n, min_cell)
    assert [x["horizon"] for x in body["reasons"]] == [7, 30]
    for reasons in body["reasons"]:
        h = reasons["horizon"]
        top = batch_order(s.finals, h)[:top_n]
        groups = batch_first_groups(ctx, s, h, top)
        assert reasons["population"] == len(top) == min(top_n, len(s.rows))
        assert served_tally(reasons) == tally(groups, top_n, min_cell)
        assert reasons["label"] == f"among the 1,000 customers most at risk at {h} days"
        assert reasons["explains"] == LABELS.reasons_fixed_line
        assert reasons["hidden_by_rule"] == {"groups": 0, "segments": 0, "customers": 0}
        expected = sum(
            len(part.group.unique())
            if len(part) < min_cell
            else int((part.group.value_counts() < min_cell).sum())
            for part in [groups, *[groups[groups.segment == m] for m in MAIN]]
        )
        assert reasons["suppressed_cells"] == expected


@pytest.mark.valkey
def test_the_http_summary_uses_the_registered_top_n_and_cell_size(summary_loaded):
    c, s = summary_loaded
    body = c.get("/summary").json()
    expected = db.summary_body(c.app.state.ctx, c.app.state.valkey, s.t, rankings_of(s.finals))
    assert body == expected
    for reasons in body["reasons"]:
        assert (reasons["top_n"], reasons["min_cell_size"]) == (1000, 20)


def test_cells_under_the_minimum_are_suppressed_and_counted():
    seg = pd.Series(["paid"] * 30 + ["discount100_only"] * 4)
    grp = pd.Series(["renewal_timing"] * 22 + ["listening_amount"] * 8 + [None] * 4)
    out = db.reason_shares(30, seg, grp, MAIN, 1000, 20).model_dump(mode="json")
    by = {x["segment"]: x for x in out["segments"]}
    assert by["all"]["customers"] == 34
    assert [(c["group"], c["customers"]) for c in by["all"]["cells"]] == [
        ("renewal_timing", 22),
        ("listening_amount", None),
        ("no group listed", None),
    ]
    assert by["paid"]["cells"][0] == {
        "group": "renewal_timing",
        "customers": 22,
        "share": 22 / 30,
        "suppressed": False,
    }
    assert by["discount100_only"] == {
        "segment": "discount100_only",
        "customers": None,
        "suppressed": True,
        "cells": [],
    }
    # all: 2 small cells; paid: listening_amount; discount100_only: 1 group, suppressed whole
    assert out["suppressed_cells"] == 2 + 1 + 1


def test_a_shown_cell_under_the_minimum_cannot_be_built():
    cell = sch.ReasonCell(group="renewal_timing", customers=3, share=0.1, suppressed=False)
    seg = sch.ReasonSegment(segment="all", customers=30, suppressed=False, cells=(cell,))
    with pytest.raises(ValidationError):
        sch.SummaryReasons(
            horizon=30,
            top_n=1000,
            population=30,
            label=DASH.reasons_label.replace("<horizon>", "30"),
            explains=LABELS.reasons_fixed_line,
            min_cell_size=20,
            suppressed_cells=0,
            hidden_by_rule=sch.HiddenReasons(groups=0, segments=0, customers=0),
            segments=(seg,),
        )


@pytest.mark.parametrize("word", ["tenure", "city", "registration_channel", "age"])
def test_a_group_or_segment_naming_a_hidden_dimension_is_removed_and_counted(word):
    seg = pd.Series(["paid"] * 25 + [f"{word}"] * 25)
    grp = pd.Series(["renewal_timing"] * 25 + [word] * 20 + ["payment_setup"] * 5)
    out = db.reason_shares(7, seg, grp, ("paid", word), 1000, 1).model_dump(mode="json")
    assert out["hidden_by_rule"] == {"groups": 1, "segments": 1, "customers": 25}
    assert never_used_hits(json.dumps(out)) == []
    assert word not in json.dumps(out)


@pytest.mark.valkey
def test_december_2016_withholds_the_reasons_and_carries_the_warning(ctx, valkey):  # noqa: F811
    s = snapshot(ctx)
    load(valkey, s)
    body = db.summary_body(ctx, valkey, DEC, rankings_of(s.finals))
    assert body["reasons"] is None
    assert body["reasons_withheld"] == LABELS.reasons_withheld_code
    assert body["data_warning"] == LABELS.december_warning
    assert body["buckets"]["days_31_59"]["grade"] == "demonstration"


def test_the_summary_is_built_once_per_store_date():
    calls = []

    def build(context, r, t, rankings):
        calls.append(t)
        return {"t": t.isoformat()}

    cache = db.SummaryCache(build)
    for t in (NOV, NOV, date(2016, 10, 1), date(2016, 10, 1), NOV):
        cache.get(None, None, t, {})
    assert calls == [NOV, date(2016, 10, 1), NOV]


# --- /plans ---------------------------------------------------------------------------------------


def stored(t: str) -> list[dict]:
    return [x for x in summary_rows() if x["snapshot"] == t and x["scenario"] == "base"]


@pytest.mark.valkey
def test_plans_equal_the_pinned_rows_of_the_store_date_and_base_scenario(summary_loaded):
    c, _ = summary_loaded
    body = c.get("/plans").json()
    assert body["plan_status"] == DASH.plan_label
    assert body["plan_status"] == (
        "precomputed in Phase 7 from the 30-day probability before the ordering fix; not re-planned"
    )
    rows = stored("2016-11-01")
    assert len(body["rows"]) == len(rows) == 3
    for got, want in zip(body["rows"], rows, strict=True):
        assert (got["strategy"], got["budget_label"]) == (want["strategy"], want["budget_label"])
        assert {k: got[k] for k in PLAN_VALUES} == {k: want[k] for k in PLAN_VALUES}
        if want["strategy"] == "random":
            assert got["range"]["label"] == "spread of random targeting, not a confidence interval"
            assert got["range"]["p2_5"] == {k: want[f"{k}_p2_5"] for k in PLAN_VALUES}
            assert got["range"]["p97_5"] == {k: want[f"{k}_p97_5"] for k in PLAN_VALUES}
        else:
            assert got["range"] is None
    assert 999.0 not in [r["net_value"] for r in body["rows"]]  # the other scenario's row


@pytest.mark.valkey
def test_plans_carry_every_registered_label(summary_loaded):
    c, _ = summary_loaded
    body = c.get("/plans").json()
    caption = "optimizer maximizes this quantity by design; the comparison is arithmetic"
    assert body["caption"] == f"{caption} on assumed effects"
    assert body["effect_basis"] == "assumed"
    assert body["instability_note"] == "The plan is unstable under 5 of 11 variants."
    assert body["no_estimate_line"] == LABELS.no_estimate_line
    assert body["dates"] == LABELS_FIXED[2]
    assert (body["scenario"], body["money_unit"]) == ("base", "price units")
    assert body["effect_prefix"] == "Under the assumed relative base effect"


def test_a_date_without_plan_rows_reads_no_plan_for_this_date():
    body = db.plans_body(pd.DataFrame(summary_rows()), date(2016, 9, 1))
    assert body["plan_status"] == "no plan for this date" and body["rows"] == []


def test_a_missing_value_is_null_not_nan():
    rows = summary_rows()
    rows[0]["expected_cost"] = float("nan")
    body = db.plans_body(pd.DataFrame(rows), NOV)
    assert body["rows"][0]["expected_cost"] is None
    json.dumps(body, allow_nan=False)


def test_a_planted_never_used_word_in_a_plan_field_is_caught_by_the_scan():
    for field in ("strategy", "budget_label"):
        rows = summary_rows()
        rows[0][field] = "by age"
        body = db.plans_body(pd.DataFrame(rows), NOV)
        assert never_used_hits(json.dumps(body)) == ["age"]


# --- /model ---------------------------------------------------------------------------------------


@pytest.mark.valkey
def test_model_serves_the_files_the_calibration_and_the_labels_with_an_empty_store(
    make_client,  # noqa: F811
):
    c = make_client()
    assert c.get("/health").status_code == 503  # no store loaded
    r = c.get("/model")
    assert r.status_code == 200
    body = r.json()
    ctx_ = c.app.state.ctx
    assert body["files"] == dict(ctx_.models.files) == sch.LABELS.bundle
    assert body["version"] == ctx_.models.version
    assert body["config_sha256"] == ctx_.config_sha256
    assert body["model_checksums_match"] is True
    cal = body["calibration"]
    assert (cal["rows"], cal["rows_changed"]) == (CALIBRATION["rows"], CALIBRATION["rows_changed"])
    for row in cal["horizons"]:
        assert row == {"horizon": row["horizon"], **CALIBRATION[str(row["horizon"])]}
    assert [row["horizon"] for row in cal["horizons"]] == [7, 30]
    assert (
        cal["label"]
        == "descriptive, no pass mark; selection half of 2016-10-01, P2 main; 2016 outcomes only"
    )
    assert cal["no_59"].startswith("no 59-day row")
    assert body["drift"] == "drift: not monitored (Phase 10)"
    assert body["retraining"] == "none in this phase (Phase 10)"
    assert body["grade_59"] == "demonstration"


@pytest.mark.valkey
def test_model_reports_a_changed_model_file(make_client):  # noqa: F811
    c = make_client()
    flip_last_byte(c.app.state.ctx.bundle_dir / "model_7d.json")
    body = c.get("/model").json()
    assert body["model_checksums_match"] is False


# --- all three over HTTP --------------------------------------------------------------------------


ENDPOINTS = ("/summary", "/plans", "/model")
HEX_RUN = re.compile(r"[0-9a-f]{32,}")  # a hash, not a name


@pytest.mark.valkey
@pytest.mark.parametrize("path", ENDPOINTS)
def test_every_endpoint_takes_no_parameters(summary_loaded, path):
    c, _ = summary_loaded
    r = c.get(path, params={"horizon": "30"})
    assert r.status_code == 422 and r.json()["detail"] == LABELS.errors[422]


@pytest.mark.valkey
@pytest.mark.parametrize("path", ["/summary", "/plans"])
def test_summary_and_plans_are_503_without_a_store_or_valkey(make_client, path):  # noqa: F811
    assert make_client().get(path).status_code == 503
    assert make_client(connect=unreachable).get(path).status_code == 503


@pytest.mark.valkey
def test_a_pinned_dashboard_file_changed_after_startup_makes_summary_and_plans_503(
    make_client,  # noqa: F811
    valkey,
    tmp_path,
):
    from test_service import write_pins

    copy = write_pins(tmp_path / "pins")
    c = make_client(pinned=copy)
    load(valkey, snapshot(c.app.state.ctx))
    assert c.get("/plans").status_code == 200
    flip_last_byte(copy["calibration"][0])
    for path in ("/summary", "/plans", "/health"):
        r = c.get(path)
        assert r.status_code == 503 and "calibration_before_after.json" in r.json()["reason"]


@pytest.mark.valkey
def test_no_body_of_the_three_names_a_never_used_attribute_or_a_customer(summary_loaded):
    c, s = summary_loaded
    for path in ENDPOINTS:
        text = c.get(path).text
        assert never_used_hits(text) == [], path
        # A SHA-256 is 64 hex characters; a short ID such as "c022" can sit inside one by chance
        # (found after a CI run, D-265). Hashes are taken out; every other character is still
        # searched, so an ID anywhere else in a body fails.
        searched = HEX_RUN.sub(" ", text)
        assert not any(m in searched for m in s.rows.msno), path


def test_a_planted_id_is_found_and_one_inside_a_hash_is_not():
    hashed = f'{{"config_sha256":"{"0" * 20}c022{"b" * 40}"}}'
    assert HEX_RUN.sub(" ", hashed) == '{"config_sha256":" "}'
    for leaked in ('{"x":"c022"}', '{"x":"id-c022-q"}', '{"x":"c022' + "f" * 8 + '"}'):
        assert "c022" in HEX_RUN.sub(" ", leaked), leaked


def test_startup_is_refused_without_the_dashboard_pins(
    registered,  # noqa: F811
    bundle_labels,  # noqa: F811
    pins,  # noqa: F811
    tmp_path,
):
    from test_service import MODEL_SETTINGS

    two = {k: v for k, v in pins.items() if k in ("plans", "alerts")}
    with pytest.raises(svc.PinRefused):
        svc.load_context(registered, MODEL_SETTINGS, P, tmp_path / "w", pins=two)


def test_startup_is_refused_when_a_dashboard_pin_differs(
    registered,  # noqa: F811
    bundle_labels,  # noqa: F811
    pins,  # noqa: F811
    tmp_path,
):
    from test_service import MODEL_SETTINGS

    changed = dict(pins) | {"summary": (pins["summary"][0], "0" * 64)}
    with pytest.raises(svc.PinRefused, match="summary.parquet"):
        svc.load_context(registered, MODEL_SETTINGS, P, tmp_path / "w", pins=changed)


def test_the_compose_file_mounts_the_two_dashboard_pins_read_only():
    import yaml

    mounts = yaml.safe_load(Path("docker-compose.yml").read_text())["services"]["api"]["volumes"]
    for name in ("summary", "calibration"):
        path = CFG["pins"][name]["path"]
        assert f"./{path}:/app/{path}:ro" in mounts
