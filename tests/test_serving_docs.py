"""The README "Scoring service" section and reports/serving_parity.md (step 9 docs).

Both are inside the registered wording scope (configs/serving.yaml, wording.scope) and
pass every wording rule. Every number in them comes from configs/serving.yaml, from a
run report under the sample's serving folder, or from LISTED, where each figure names
the run it comes from. The run reports are local; without them the number check prints
"numbers NOT checked (local reports absent)" in the test output instead of passing
quietly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from retention.api import docs
from retention.api.config import SERVING_CONFIG, load_serving

WORDING = load_serving()["wording"]
REPORTS = Path("data/processed/kkbox/sample_020/serving")
README = Path("README.md").read_text()
SECTION = README[README.index("## Scoring service") : README.index("## Limitations")]
PARITY_REPORT = Path("reports/serving_parity.md").read_text()
TEXTS = {"README Scoring service": SECTION, "reports/serving_parity.md": PARITY_REPORT}

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
HEXISH = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{4,}\b")
# Figures not in the config or a run report, each with the run it comes from.
LISTED = {
    "672.4": "docker stats after one GET /at-risk?auto_renew=on at eb0e517 (step 8 run, 01:30)",
    "97.38": "sysctl vm.swapusage before and after that same run at eb0e517 (step 8 run)",
    "15": "smoke_2016-11-01.json: readiness hash median 14.94 ms, rounded",
    "6.65": "http_parity_diagnosis_2016-11-01.json: largest relative gap 6.654714871173301e-16",
    "9.83": "ps resident size of Docker's VM process during the step 7d run (23:10)",
    "23": "step4_report.json: its marks, counted (23, every one met)",
    "0.6": "http_parity_2016-11-01.json: customers 1,000 of scored_ids 169,673 (0.589%), rounded",
}


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER.findall(HEXISH.sub(" ", text))}


def report_numbers() -> set[str] | None:
    files = sorted(REPORTS.glob("*.json"))
    if not files:
        return None
    return set().union(*(numbers(json.dumps(json.loads(f.read_text()))) for f in files))


# --- the wording rules --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(TEXTS))
def test_each_text_passes_every_wording_rule(name):
    assert docs.wording_violations(TEXTS[name]) == []


def test_the_fixed_disclaimers_are_in_the_readme_section_verbatim():
    for d in WORDING["fixed_disclaimers"]:
        assert d in SECTION
    assert len(re.findall(r"\blive\b", SECTION)) == 1


@pytest.mark.parametrize("name", list(TEXTS))
def test_the_last_bit_gaps_are_named_a_likely_source_untested(name):
    assert "likely source, untested" in TEXTS[name]


# --- parity: 996 and A10 beside every parity claim (README) -------------------------------------


def paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def parity_problems(text: str) -> list[str]:
    out = []
    for p in paragraphs(text):
        if p.lstrip().startswith("#") and "same scores" in p:
            out.append("'same scores' in a heading")
        if re.search(r"\bparity\b|same scores", p, re.I):
            if "996" not in p or "A10" not in p:
                out.append("a parity claim without 996 and A10 in its paragraph")
            if "same scores as batch" in p and "Step 6" not in p:
                out.append("'same scores as batch' not tied to step 6")
    return out


def test_every_readme_parity_claim_carries_996_and_a10_in_its_paragraph():
    claims = [p for p in paragraphs(SECTION) if re.search(r"\bparity\b", p, re.I)]
    assert claims and parity_problems(SECTION) == []
    p = next(x for x in claims if "Parity over HTTP" in x)
    for text in ("was not met: 996 of 1,000", "written after that result was seen", "Step 6"):
        assert text in p


@pytest.mark.parametrize(
    "planted",
    [
        "Parity: the stack gives the same scores as batch.",
        "Parity over HTTP met under A10.",
        "## The same scores as batch\n\nStep 6 parity, 996 and A10.",
    ],
)
def test_a_parity_claim_without_996_or_a10_is_caught(planted):
    assert parity_problems(planted)


# --- latency and memory (README) -----------------------------------------------------------------


def sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.;])\s+", text)


def threshold_problems(text: str) -> list[str]:
    return [s[:60] for s in sentences(text) if "100 ms" in s and "my unvalidated choice" not in s]


@pytest.mark.parametrize("name", list(TEXTS))
def test_every_100_ms_threshold_is_labelled_my_unvalidated_choice(name):
    assert "100 ms" in TEXTS[name] and threshold_problems(TEXTS[name]) == []
    assert threshold_problems("A threshold of 100 ms was met.")


def test_the_latency_sentence_carries_p99_max_and_the_limit_note():
    s = next(x for x in sentences(SECTION) if "p95 69.429 ms" in x)
    for text in (
        "97.809 ms",
        "108.142 ms",
        "measured before the 1.5 GiB limit",
        WORDING["latency_sentence"],
    ):
        assert text in s


def test_the_memory_figure_carries_swap_and_no_concurrency_and_no_safety_claim():
    p = next(x for x in paragraphs(SECTION) if "672.4 MiB" in x)
    for text in (
        "swap above 0 (97.38 MB)",
        "no concurrent clients",
        "8 concurrent clients was not measured",
    ):
        assert text in p
    assert not re.search(r"\bsafe(ly|ty)?\b", SECTION, re.I)


# --- the parity report ----------------------------------------------------------------------------


def test_the_report_s_a9_history_is_what_a9_s_own_label_says():
    a9 = load_serving()["stream_parity_columns"]
    before = "written while planning the move, before any move code or run"
    assert a9["label"].startswith(before)  # the label: written before any run
    assert f"not met as written ({'mis-specified before any run'})" in PARITY_REPORT
    assert a9["registered_wording"] == "not met as written (mis-specified before any run)"
    assert "A9 was " + before in PARITY_REPORT
    assert "amendment A9, written before the result" in PARITY_REPORT
    assert "A9 was written after" not in PARITY_REPORT


def test_the_readme_states_the_http_sample_s_reach_and_what_it_leaves_out():
    p = next(x for x in paragraphs(SECTION) if "Parity over HTTP" in x)
    for text in (
        "1,000 of the 169,673 scored customers (0.6%)",
        "one seed, scored customers only, at one store date",
        "outside_population and not_active answers are covered by neither",
        "checked in-process only",
    ):
        assert text in p


def test_the_console_readings_are_labelled_as_having_no_saved_report():
    p = next(x for x in paragraphs(SECTION) if x.startswith("Unverified"))
    assert "672.4 MiB, 97.38 MB and 9.83 GB are console readings" in p
    assert "with no report written to disk" in p


def test_the_report_shows_a9_and_a10_as_not_met_as_written_and_996_on_record():
    assert "| Feature parity, as written |" in PARITY_REPORT
    assert "not met as written (mis-specified before any run)" in PARITY_REPORT
    assert "| 996 of 1,000: not met as written |" in PARITY_REPORT
    assert "that result stays on record" in PARITY_REPORT
    assert "Amendment A10 was written after that result was seen." in PARITY_REPORT


# --- every number has a source ------------------------------------------------------------------


def test_every_listed_figure_names_its_source_run_and_is_still_needed():
    for value, source in LISTED.items():
        assert re.search(r"\.json|\brun\b", source), value
    used = set().union(*(numbers(t) for t in TEXTS.values()))
    assert set(LISTED) <= used


@pytest.mark.parametrize("name", list(TEXTS))
def test_every_number_comes_from_the_config_a_run_report_or_the_listed_set(name, capsys):
    found = report_numbers()
    if found is None:
        with capsys.disabled():
            print(f"\n{name}: numbers NOT checked (local reports absent)")
        return
    sources = numbers(Path(SERVING_CONFIG).read_text()) | found | set(LISTED)
    assert numbers(TEXTS[name]) - sources == set()


def test_a_number_with_no_source_is_caught():
    sources = numbers(Path(SERVING_CONFIG).read_text()) | set(LISTED)
    assert numbers("p95 was 123456.789 ms") - sources == {"123456.789"}
