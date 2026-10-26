"""docs/model_card.md: sourced numbers, the four disclaimers, the A16 drift status, the wording."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from retention.api.config import load_serving
from test_dashboard_wording import check

CARD = Path("docs/model_card.md").read_text()
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
HEXISH = re.compile(r"#?\b(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{4,}\b")
OPS = yaml.safe_load(Path("configs/ops.yaml").read_text())


def numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(HEXISH.sub(" ", text))}


SOURCES = set().union(
    *(numbers(p.read_text()) for p in Path("configs").glob("*.yaml")),
    *(numbers(p.read_text()) for p in Path("reports").glob("*.md")),
    *(numbers(p.read_text()) for p in Path("docs").glob("*.md") if p.name != "model_card.md"),
)
LISTED = {
    "2018": "the competition's name, WSDM Cup 2018 (configs/kkbox.yaml)",
    "1.5": "the API container's memory limit, 1.5 GiB (configs/serving.yaml api_memory_limit)",
}


def test_every_number_comes_from_a_report_a_config_or_a_doc():
    assert numbers(CARD) - SOURCES - set(LISTED) == set()


def test_the_four_fixed_disclaimers_are_verbatim():
    for d in load_serving()["wording"]["fixed_disclaimers"]:
        assert d in CARD


def test_the_results_table_is_copied_from_the_comparison_report():
    report = Path("reports/model_comparison.md").read_text()
    for row in re.findall(r"^\| (?:Survival model dt01|LightGBM|Banded)[^\n]*", CARD, re.M):
        for value in re.findall(r"\d\.\d{4} \[[^\]]+\]", row):
            assert value in report, value


def test_the_drift_status_is_the_one_recorded_in_a16():
    a = OPS["amendment_A16"]["not_run"]
    ev = a["real_drift_replays"]["drift_evidence"]
    assert "Drift is demonstrated on synthetic data only, never on KKBox." in CARD
    assert "dataset-level alert rule was NOT MET (shifted share 0.2979, below 0.5)" in CARD
    assert "margin rule was met (0.2553 against 0.20)" in CARD
    assert "0.2979" in ev and "0.2553" in ev
    assert "real drift replays on the 20% sample are NOT RUN" in CARD
    assert "retraining flow and the one real retraining run are NOT DONE" in CARD
    assert "not tested" in CARD


def test_the_card_passes_every_wording_rule_and_claims_no_hosting_or_demo():
    assert check(CARD) == []
    assert not re.search(r"\bdemo\b|\bhosted at\b|\blive link\b|\bone command\b", CARD, re.I)
    assert "nothing in this repository is hosted" in CARD  # a negative statement only
