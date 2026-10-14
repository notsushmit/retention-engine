"""Reason text: the structure check of the phrases file, fact rendering, and full explanations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from retention.explain.config import load_explain_config
from retention.explain.model import load_settings
from retention.explain.reasons import (
    PHRASES,
    Renderer,
    facts_text,
    platt_probability,
    private_names,
    probability_text,
    render_fact,
    structure_problems,
    tenure_words,
)
from retention.explain.selection import select, shown_mask
from retention.models.calibration import Platt

CFG = load_explain_config(load_settings().columns)
TEXT = PHRASES.read_text()
R = Renderer(CFG, TEXT)
FACTS = R.p["groups"]
LN12, LN15, LN2 = math.log(1.2), math.log(1.5), math.log(2.0)


def below(x: float) -> float:
    return float(np.nextafter(x, 0.0))


def one_row(c: dict[str, float], tenure_known: bool = True, b: float = 1.0):
    """A one-row Selection from calibrated group contributions given by group name."""
    vec = np.array([[c.get(g, 0.0) for g in CFG.group_names]]) / b
    rows = pd.DataFrame({"tenure_days": pd.array([400 if tenure_known else None], "Int32")})
    return select(CFG, vec, vec.sum(axis=1), b, shown_mask(CFG, rows))


# --- the structure check -------------------------------------------------------------------------


def test_the_phrases_file_has_no_problems():
    assert structure_problems(TEXT, CFG) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            'close: "Risk is close to a typical customer in the training months."',
            'close: ""',
            "position.close is not filled",
        ),
        (
            '      tenure_days: "member for {tenure}"',
            '      tenure_months: "member for {tenure}"',
            "facts",
        ),
        (
            'format: "last plan ran for {days}"',
            'format: "last plan ran for {weeks}"',
            "unsupported placeholder",
        ),
        (
            'raises: "Plan length is pushing risk up"',
            'raises: "Plan length 2 is pushing risk up"',
            "has a digit",
        ),
        (
            'raises: "Plan length is pushing risk up"',
            'raises: "More plan length is pushing risk up"',
            "comparison word",
        ),
        (
            'raises: "Plan length is pushing risk up"',
            'raises: "Plan length pushes risk up"',
            "does not have the shape",
        ),
        (
            'may be undercounted in the data."',
            'may be undercounted because of the data."',
            "causal word",
        ),
        (
            'raising: "Nothing in particular is pushing risk up."',
            'raising: "Nothing in particular is pushing risk up; you should call."',
            "advice word",
        ),
        (
            'when_true: "auto-renew on"',
            'when_true: "auto-renew on, registered by app"',
            "names something hidden",
        ),
        ("# Strength words are fixed", "# See D-081. Strength words are fixed", "decision ID"),
        (
            'raising: "Nothing in particular is pushing risk up."',
            'raising: "Nothing makes risk higher than typical."',
            "'higher than'",
        ),
        (
            'higher: "Risk is higher than a typical customer in the training months ({strength})."',
            'higher: "Risk is higher than a typical customer in the training months."',
            "placeholders",
        ),
    ],
)
def test_each_rule_is_caught(old, new, expected):
    assert old in TEXT, old
    problems = structure_problems(TEXT.replace(old, new, 1), CFG)
    assert any(expected in p for p in problems), problems
    with pytest.raises(ValueError):
        Renderer(CFG, TEXT.replace(old, new, 1))


def test_private_file_names_come_from_the_exclude_file(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git" / "info").mkdir(parents=True)
    (repo / "notes").mkdir()
    (repo / "notes" / "SCRATCHBOOK.md").write_text("x")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.write_text("# comment\n/SECRETPLAN.md\n/notes/\n")
    names = private_names(exclude)
    assert names == ["SCRATCHBOOK", "SECRETPLAN", "notes"]
    text = TEXT.replace(
        "# Strength words are fixed", "# See SECRETPLAN. Strength words are fixed", 1
    )
    assert any("private file" in p for p in structure_problems(text, CFG, names))
    assert structure_problems(TEXT, CFG, names) == []
    assert private_names(tmp_path / "missing") == []


def test_the_phrases_file_names_no_local_private_file():
    assert structure_problems(TEXT, CFG, private_names()) == []


def test_every_fact_key_is_in_explain_yaml():
    for g, spec in FACTS.items():
        assert sorted(spec["facts"]) == sorted(CFG.facts[g])


# --- facts ---------------------------------------------------------------------------------------


def fact(group: str, col: str, value):
    return render_fact(FACTS[group]["facts"][col], value, R.p["missing_value"], col)


def test_days_placeholder_and_zero():
    assert fact("renewal_timing", "days_until_expiry", 1) == "expires in 1 day"
    assert fact("renewal_timing", "days_until_expiry", 3) == "expires in 3 days"
    assert fact("renewal_timing", "days_until_expiry", 0) == "expires today"
    assert fact("renewal_timing", "days_since_last_payment", 27) == "last paid 27 days ago"


def test_tenure_in_years_and_months():
    assert tenure_words(213) == "7 months"
    assert tenure_words(825) == "2 years and 3 months"
    assert tenure_words(365) == "1 year"
    assert tenure_words(20) == "less than a month"
    assert fact("tenure", "tenure_days", 825) == "member for 2 years and 3 months"


def test_minutes_round_half_up_with_their_unit_tiny_and_zero():
    def m(v):
        return fact("listening_amount", "secs_per_day_28", v)

    assert m(600.0) == "about 10 minutes a day"
    assert m(60.0) == "about 1 minute a day"
    assert m(30.0) == "about 1 minute a day"  # half up, not half to even
    assert m(90.0) == "about 2 minutes a day"
    assert m(29.99) == "under half a minute a day"
    assert m(20.0) == "under half a minute a day"
    assert m(0.0) is None  # null: left out


def test_value_format_specs():
    assert fact("listening_trend", "ratio_secs_7_vs_base", 0.6) == (
        "listening in the last 7 days was about 60% of the usual (the three weeks before)"
    )
    assert fact("listening_trend", "ratio_secs_7_vs_base", 1.5).endswith(
        "about 150% of the usual (the three weeks before)"
    )
    assert render_fact({"format": "ratio {value:.1f}"}, 1.26, "n/a") == "ratio 1.3"
    assert fact("plan_kind_and_price", "last_amount_paid", 149) == "paid 149 for it"


def test_missing_values():
    assert fact("listening_trend", "ratio_secs_7_vs_base", None) == (
        "too little listening in the three weeks before to compare with"
    )
    assert fact("renewal_timing", "days_until_expiry", None) == "not available"
    assert fact("listening_trend", "ratio_active_7_vs_base", pd.NA) is None  # null: left out
    assert fact("tenure", "tenure_days", float("nan")) == "not available"


def test_booleans_and_categories():
    assert fact("payment_setup", "last_is_auto_renew", 1) == "auto-renew on"
    assert fact("payment_setup", "last_is_auto_renew", 0) == "auto-renew off"
    assert fact("listening_recency", "has_any_log_before_T", 1) is None
    assert fact("plan_kind_and_price", "last_row_kind", "trial") == "last plan was a free trial"


def test_facts_text_capitalises_ends_with_a_full_stop_and_skips_left_out_facts():
    assert facts_text(["expires in 3 days", None, "last paid 27 days ago"], ", ") == (
        "Expires in 3 days, last paid 27 days ago."
    )
    assert facts_text(["already ends."], ", ") == "Already ends."
    assert facts_text([None, None], ", ") == ""


def test_no_fact_is_said_twice():
    line = R.reason_line(
        "listening_amount", "raises", 1, {"active_days_28": 0, "secs_per_day_28": 0.0}
    )
    assert (
        line
        == "How much they listen is pushing risk up (slight). No listening in the last 28 days."
    )
    line = R.reason_line(
        "listening_trend", "raises", 1, {"ratio_secs_7_vs_base": 0.0, "ratio_active_7_vs_base": 0.0}
    )
    assert (
        line
        == "Their listening trend is pushing risk up (slight). No listening in the last 7 days."
    )
    line = R.reason_line(
        "listening_recency",
        "raises",
        1,
        {"days_since_last_listen": None, "has_any_log_before_T": 0},
    )
    assert line == "How recently they listened is pushing risk up (slight). No listening on record."


# --- position, other factors, partly-cancel -------------------------------------------------------


def lines(c, **kw):
    s = one_row(c, kw.pop("tenure_known", True))
    return R.render(s, 0, kw.pop("facts", {}), kw.pop("horizon", 7), kw.pop("p", 0.05), **kw)


def test_position_strength_label():
    assert lines({"renewal_timing": LN2}).lines[0].endswith("months (strong).")
    assert lines({"renewal_timing": below(LN2)}).lines[0].endswith("months (moderate).")
    assert lines({"payment_setup": -LN12}).lines[0] == (
        "Risk is lower than a typical customer in the training months (slight)."
    )
    assert lines({"payment_setup": -below(LN12)}).lines[0] == (
        "Risk is close to a typical customer in the training months."
    )


@pytest.mark.parametrize(
    ("other", "label"),
    [
        (LN12, "push risk up (slight)."),
        (below(LN12), "roughly cancel out."),
        (LN15, "push risk up (moderate)."),
        (below(LN15), "push risk up (slight)."),
        (LN2, "push risk up (strong)."),
        (below(LN2), "push risk up (moderate)."),
        (-LN12, "pull risk down (slight)."),
        (-below(LN12), "roughly cancel out."),
    ],
)
def test_other_factors_boundaries(other, label):
    # One listed reason (billing timing), everything else in the hidden group.
    out = lines({"renewal_timing": 0.9, "hidden": other})
    assert any(x.startswith("Other factors that did not make the list") for x in out.lines)
    assert [x for x in out.lines if x.startswith("Other factors")][0].endswith(label)


def test_other_factors_line_is_left_out_when_nothing_is_listed():
    out = lines({"hidden": 0.8})  # higher, but nothing may be listed
    assert not any(x.startswith("Other factors") for x in out.lines)
    assert out.lines[1] == "No single thing stands out; many smaller things together push risk up."


PARTLY = {
    "close with a strong reason up and one down": (
        {"renewal_timing": 0.8, "payment_setup": -0.7},
        "close",
    ),
    "weaker higher": ({"renewal_timing": 0.8, "payment_setup": -0.3}, "higher"),
    "weaker lower, strongest pulls down": (
        {"renewal_timing": 0.8, "payment_setup": -1.0, "plan_kind_and_price": -0.2},
        "lower",
    ),
    "one way listed, visible opposite other factors": (
        {"renewal_timing": 0.9, "hidden": -0.4},
        "higher",
    ),
}
NO_PARTLY = {
    "equal strength": {"renewal_timing": 0.8},
    "nothing listed": {"hidden": 0.1},
    "one way listed, other factors roughly cancel": {"renewal_timing": 0.75, "hidden": -0.1},
    "stronger position than every reason": {"renewal_timing": 0.5, "plan_kind_and_price": 0.4},
}


@pytest.mark.parametrize("name", list(PARTLY))
def test_partly_cancel_line_present(name):
    c, key = PARTLY[name]
    out = lines(c)
    assert out.partly_cancel
    assert R.p["partly_cancel"][key] in out.lines


@pytest.mark.parametrize("name", list(NO_PARTLY))
def test_partly_cancel_line_absent(name):
    out = lines(NO_PARTLY[name])
    assert not out.partly_cancel
    assert not any(x.startswith("These pull in different directions") for x in out.lines)


def test_listed_plus_other_equals_d_on_a_rendered_row():
    rng = np.random.default_rng(3)
    for _ in range(200):
        c = dict(
            zip(CFG.group_names, rng.normal(scale=0.5, size=len(CFG.group_names)), strict=True)
        )
        out = lines(c, tenure_known=bool(rng.random() < 0.8))
        assert abs(sum(x[2] for x in out.listed) + out.other - out.d) <= 1e-12


def test_probability_text_boundaries():
    assert probability_text(0.001) == "0.1%"
    assert probability_text(float(np.nextafter(0.001, 0.0))) == "under 0.1%"
    assert probability_text(0.0) == "under 0.1%"
    assert probability_text(0.999) == "99.9%"
    assert probability_text(float(np.nextafter(0.999, 1.0))) == "over 99.9%"
    assert probability_text(0.0423) == "4.2%"


def test_a_tiny_probability_never_reads_zero():
    out = lines({"payment_setup": -3.0}, p=0.0002)
    assert out.lines[-1] == (
        "Chance of lapsing within 7 days: under 0.1% (calibrated for this time window only)"
    )


def test_platt_probability_matches_the_calibrator():
    pl = Platt(seed=1)
    pl.a, pl.b = 0.024395167821727723, 0.9988291492427049
    raw = np.array([-9.0, -6.2, -2.0, 0.0, 3.5])
    want = np.clip(pl.predict(1 / (1 + np.exp(-raw))), 1e-6, 1 - 1e-6)
    assert np.array_equal(platt_probability(raw, pl.a, pl.b), want)


# --- one full rendered explanation per horizon ---------------------------------------------


def test_snapshot_7_days():
    c = {
        "renewal_timing": 0.9,
        "plan_kind_and_price": 0.3,
        "listening_amount": 0.25,
        "songs_finished": 0.1,
        "hidden": 0.05,
        "payment_setup": -0.5,
    }
    facts = {
        "days_until_expiry": 3,
        "days_since_last_payment": 27,
        "last_row_kind": "paid",
        "last_amount_paid": 149,
        "active_days_28": 0,
        "secs_per_day_28": 0.0,
        "last_is_auto_renew": 1,
    }
    out = lines(c, facts=facts, horizon=7, p=0.0423, caveats=("months_used_for_settings",))
    assert out.lines == [
        "Risk is higher than a typical customer in the training months (strong).",
        "Billing timing is pushing risk up (strong). Expires in 3 days, last paid 27 days ago.",
        "Plan and price is pushing risk up (slight). On a paid plan, paid 149 for it.",
        "How much they listen is pushing risk up (slight). No listening in the last 28 days.",
        "How they pay is pulling risk down (moderate). Auto-renew on.",
        "Other factors that did not make the list roughly cancel out.",
        "Chance of lapsing within 7 days: 4.2% (calibrated for this time window only)",
        "This month helped choose the model's settings, though the model was not trained on it.",
    ]


def test_snapshot_30_days():
    c = {
        "listening_amount": 0.45,
        "tenure": 0.3,
        "listening_trend": 0.2,
        "renewal_timing": -0.1,
        "payment_setup": -0.8,
    }
    facts = {
        "active_days_28": 2,
        "secs_per_day_28": 300.0,
        "tenure_days": 213,
        "ratio_secs_7_vs_base": 0.6,
        "ratio_active_7_vs_base": 0.75,
        "last_is_auto_renew": 0,
    }
    out = lines(c, facts=facts, horizon=30, p=0.0187, caveats=("february_2017_gap",))
    assert out.lines == [
        "Risk is close to a typical customer in the training months.",
        "How much they listen is pushing risk up (moderate). "
        "Listened on 2 days of the last 28, about 5 minutes a day.",
        "How long they have been a member is pushing risk up (slight). Member for 7 months.",
        "Their listening trend is pushing risk up (slight). Listening in the last 7 days was "
        "about 60% of the usual (the three weeks before), listening days about 75% of the usual.",
        "How they pay is pulling risk down (strong). Auto-renew off.",
        "Other factors that did not make the list roughly cancel out.",
        "These pull in different directions, so the overall risk is close to typical.",
        "Chance of lapsing within 30 days: 1.9% (calibrated for this time window only)",
        "Renewals on 13 February 2017 may be undercounted in the data.",
    ]
