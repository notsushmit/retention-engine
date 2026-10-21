"""docs/scoring_api.md: equal to a fresh build, registered content present, wording rules kept."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from retention.api import __main__ as cli
from retention.api import docs
from retention.api.config import SERVING_CONFIG, load_serving

DOC_TEXT = docs.DOC.read_text()
SERVING = load_serving()
A3 = next(a for a in SERVING["step4"]["amendments"] if a["id"] == "A3")
NUMBER = re.compile(r"\d[\d,]*\d|\d")
# Numbers allowed in the doc although configs/serving.yaml does not hold them, each with why.
NOT_IN_CONFIG = {
    "199": "the decision-log id D-199 named by the workaround note",
}


def test_the_committed_doc_equals_a_fresh_build():
    assert DOC_TEXT == docs.build()


def test_write_docs_writes_the_doc_and_nothing_else(monkeypatch, tmp_path):
    target = tmp_path / "scoring_api.md"
    monkeypatch.setattr(docs, "DOC", target)
    assert cli.main(["write-docs"]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scoring_api.md"]
    assert target.read_text() == docs.build()


# --- registered content ---------------------------------------------------------------------------


@pytest.mark.parametrize("sentence", SERVING["step5"]["docs"]["must_state"])
def test_each_registered_sentence_is_stated_verbatim(sentence):
    assert f"- {sentence}\n" in DOC_TEXT


def test_there_are_ten_registered_sentences():
    assert len(SERVING["step5"]["docs"]["must_state"]) == 10


@pytest.mark.parametrize("disclaimer", SERVING["wording"]["fixed_disclaimers"])
def test_each_fixed_disclaimer_is_used_verbatim(disclaimer):
    assert f"> {disclaimer}\n" in DOC_TEXT


@pytest.mark.parametrize(
    "text",
    [
        "A5 (2026-10-21, amends contract.outside_population.codes)",
        "reads `unseen_level:hidden_feature`",
        "A6 (2026-10-21, amends contract.never_in_any_body, step5.scan)",
        "3 of 425,520 known IDs contain one never-used word as a whole word",
        "5 of 1,474,743 distinct IDs across the seven tables do (distinct IDs, not table rows)",
        "An identifier exemption, not a relaxation of the never-used rule.",
        "A7 (2026-10-21, amends api.alerts.trend.hidden_dimensions)",
        "top_segments 4, dimension_totals 10, too_small_segments 65, every other field 0",
        "A8 (2026-10-21, amends step5.alerts.trend_knowable_at_2016_11_01)",
        "the registered value 7 was wrong",
        "gives 6 at 2016-11-01; cp7 (week 2016-09-26) is knowable from 2016-11-02",
    ],
)
def test_amendments_a5_to_a8_are_stated(text):
    assert text in DOC_TEXT


def test_the_a3_note_takes_its_count_from_the_registered_text():
    assert A3["counts"]["total_listed"] == 11368
    assert f"listed as a reason in {A3['counts']['total_listed']:,} rows" in DOC_TEXT


@pytest.mark.parametrize(
    "text",
    [
        "together with a placeholder copy of the row",
        "a workaround that relies on NumPy's summation behaviour",
        "chosen after the exact-equality test failed, not a fix of its source",
        "skips in CI, because the pinned file is not in git",
        'the December 2016 withheld code "2016-12 not explained"',
    ],
)
def test_the_decision_log_notes_are_stated(text):
    assert text in DOC_TEXT


@pytest.mark.parametrize("choice", docs.UNREGISTERED)
def test_each_unregistered_choice_is_listed_under_its_heading(choice):
    section = DOC_TEXT.split("## Choices not registered", 1)[1]
    assert f"- {choice}\n" in section


# --- the wording rules ----------------------------------------------------------------------------


def test_the_doc_passes_every_wording_rule():
    assert docs.wording_violations(DOC_TEXT) == []


LIVE = docs.LIVE_EXCEPTION


@pytest.mark.parametrize(
    "text, found",
    [
        (f"{LIVE} {LIVE}", "the live disclaimer is used twice"),
        (f"{LIVE} Scores are live.", "banned word 'live'"),
        ("Scores update in real-time.", "banned word 'real-time'"),
        ("A production service.", "banned word 'production'"),
        ("The two scores are identical.", "banned word 'identical'"),
        ("Lapse by 60 days.", "'60 days'"),
        ("This keeps customers.", "banned word 'keeps'"),
        (
            f"{docs.TENURE_SENTENCE} {docs.TENURE_SENTENCE}",
            "an exact-string exception is used twice",
        ),
        (
            f"{docs.SELF_CHECK_SENTENCE} {docs.SELF_CHECK_SENTENCE}",
            "an exact-string exception is used twice",
        ),
        (f"{docs.TENURE_SENTENCE} Tenure is a filter.", "never-used attribute 'Tenure'"),
        ("Rows by city.", "never-used attribute 'city'"),
        ("The 59-day probability is shown.", "a 59-day probability without 'demonstration'"),
    ],
)
def test_each_rule_catches_its_breach(text, found):
    assert any(v.startswith(found) for v in docs.wording_violations(text)), docs.wording_violations(
        text
    )


def test_each_exception_passes_once():
    once = f"{LIVE} {docs.TENURE_SENTENCE} {docs.SELF_CHECK_SENTENCE}"
    assert docs.wording_violations(once) == []


# --- every number comes from the config -----------------------------------------------------------


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER.findall(text)}


def test_every_number_in_the_doc_is_in_the_config_or_listed_here():
    in_config = numbers(Path(SERVING_CONFIG).read_text())
    missing = numbers(DOC_TEXT) - in_config - set(NOT_IN_CONFIG)
    assert missing == set()
    assert set(NOT_IN_CONFIG) <= numbers(DOC_TEXT)  # every listed exception is still needed


# --- amendments A9 and A10 and the registered notes of steps 7 and 8 -----------------------------


@pytest.mark.parametrize(
    "text",
    [
        "A9 (2026-10-21, amends streaming.pass_marks.feature_parity): feature parity is judged on "
        "61 columns; 7 metadata columns are excluded",
        "The registered wording: not met as written (mis-specified before any run).",
        "A10 (amends step8.http_score_parity): found after a result (D-243, D-244)",
        "every non-float field exact and every float within relative 1e-12",
        "The registered wording: not met as written (996 of 1,000 bodies equal as strings at "
        "224f672, D-243).",
        "outside the sample it is not observed.",
    ],
)
def test_amendments_a9_and_a10_are_stated(text):
    assert text in DOC_TEXT


def test_every_registered_docs_note_of_steps_7_and_8_is_stated_verbatim():
    cfg = load_serving()
    notes = [
        *cfg["step7"]["docs_notes"],
        *cfg["step7_images"]["docs_notes"],
        *cfg["step8"]["docs_notes"],
    ]
    assert len(notes) == 6 and all(f"- {n}" in DOC_TEXT for n in notes)
    assert "memory limit is 1.5GiB, the user's choice; Valkey has no limit" in DOC_TEXT
