"""The plan, observed and report steps end to end on synthetic customers, and the wording rules.

The Phase 6 docs (docs/uplift_validation.md, docs/causal_assumptions.md,
docs/counterfactuals.md) are exempt from the external-name ban: they keep their
own public-data figures. Every Phase 7 text is checked.
"""

import json
import re
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import retention.optimize.__main__ as cli
from retention.optimize.config import load_actions_config
from retention.optimize.report import (
    REPORT,
    load_logs,
    render,
    solver_lines,
    wording_violations,
)

CFG = load_actions_config()
README_SENTENCE = (
    "KKBox backtest figures are model estimates under assumed effects, not observed outcomes."
)


def synthetic_customers(n: int, snapshot: date, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = rng.uniform(0, 60, n)
    return pd.DataFrame(
        {
            "msno": [f"c{seed}_{i:04d}" for i in range(n)],
            "snapshot_date": [snapshot] * n,
            "auto_renew": np.where(rng.random(n) < 0.3, "off", "on"),
            "days_until_paid_expiry": pd.array(rng.integers(0, 30, n), dtype="Int32"),
            "dt01_lapse_30": rng.random(n) * 0.3,
            "days_30_90": days,
            "days_30_365": days * 4,
            "p30": rng.beta(0.6, 4, n),
            "price_per_day": rng.choice([0.0, 99 / 30, 149 / 30], n, p=[0.05, 0.45, 0.5]),
            "has_price": True,
        }
    )


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("optimize")
    customers = pd.concat(
        [
            synthetic_customers(400, date(2016, 11, 1), 1),
            synthetic_customers(300, date(2016, 10, 1), 2),
        ],
        ignore_index=True,
    )
    customers.to_parquet(out / "value.parquet")
    (out / "dt01_refit.json").write_text(json.dumps({"passed": True, "reproduction": [{}] * 3}))
    disagree = {"top_decile": 40, "disagree": 7, "share": 0.175}
    per = {
        t: {"customers": 400, "auto_renew_off": 120, "auto_renew_on": 280, "no_usable_price": 20,
            "missing_deadline": 0, "model_disagreement": disagree}
        for t in ("2016-11-01", "2016-10-01")
    }  # fmt: skip
    (out / "recount.json").write_text(
        json.dumps(
            {
                "per_snapshot": per,
                "price_table": {
                    "top": [
                        {"plan_days": 30, "list_price": 149, "paid": 149, "n": 10, "share": 0.5}
                    ]
                },
            }  # fmt: skip
        )
    )
    (out / "value_report.json").write_text(json.dumps({"per_snapshot": per}))

    cfg = replace(CFG, headline_budget_level=1000.0, random_draws=20)
    mp = pytest.MonkeyPatch()
    mp.setattr(cli, "load_actions_config", lambda: cfg)
    mp.setattr(cli, "out_dir", lambda fraction: out)
    mp.setattr(cli, "require_clean", lambda repo: None)
    mp.setattr(cli, "head_commit", lambda repo: "test000")
    mp.setattr(cli, "load_protocol", lambda: None)
    # other tests in this process may have loaded the label code; the separation itself
    # is checked in a fresh interpreter (test_value_step_never_loads_label_building_code)
    mp.setattr(cli, "assert_no_label_building_code", lambda: None)

    def fake_read_once(path, snapshot, protocol, out_dir, head):
        c = customers[customers.snapshot_date == snapshot]
        rng = np.random.default_rng(5)
        y = pd.array((rng.random(len(c)) < c.p30.to_numpy()).astype(int), dtype="Int8")
        return pd.DataFrame({"msno": c.msno.to_numpy(), "churned_30": y})

    mp.setattr(cli, "read_once", fake_read_once)
    args = type("A", (), {"fraction": None})()
    assert cli.plan(args) == 0
    assert cli.observed_step(args) == 0
    mp.undo()
    return out, cfg


def test_plan_writes_every_registered_cell(run_dir):
    out, cfg = run_dir
    rep = json.loads((out / "plan_report.json").read_text())
    assert rep["criterion_4"] == {
        "primary_solves": 13 * 8,
        "secondary_solves": 8,
        "rows_carry_scenario_fields": True,
    }
    assert rep["breaches"] == [] and rep["solver_summary"]["breaches"] == 0
    s = pd.read_parquet(out / "summary.parquet")
    assert set(s.effect_basis) == {"assumed"}
    assert set(s.strategy) == {
        "optimizer", "highest_risk", "highest_assumed_uplift_x_value", "fixed_action_reminder",
        "no_action", "random",
    }  # fmt: skip
    opt = s[s.strategy == "optimizer"]
    assert (opt.expected_cost <= opt.budget.fillna(np.inf) + 1e-9).all()
    low = s[(s.scenario == "low_effect") & (s.strategy == "optimizer")]
    assert (low.contacted == 0).all()  # registered expectation: zero effect, empty plan
    st = pd.read_parquet(out / "stability.parquet")
    assert len(st) == 12 * 8


def test_observed_shares_carry_the_label(run_dir):
    out, cfg = run_dir
    obs = json.loads((out / "observed_churn.json").read_text())
    assert obs["label"] == "says nothing about whether any action works"
    assert set(obs["shares"]) == {"2016-11-01", "2016-10-01"}
    assert all(v["label"] == obs["label"] for by in obs["shares"].values() for v in by.values())


def test_rendered_report_keeps_every_wording_rule(run_dir):
    out, cfg = run_dir
    text = render(cfg, load_logs(out))
    assert wording_violations(text, cfg) == []
    tables = sum(1 for ln in text.splitlines() if ln.startswith("| Strategy |"))
    assert text.count(f"*{cfg.caption}.*") == tables >= 2
    assert README_SENTENCE in text
    for line in text.splitlines():
        if "expected retained value is" in line:
            assert line.startswith("Under the assumed relative base effect")


@pytest.mark.parametrize(
    ("bad", "found"),
    [
        ("The discount saves customers.", "banned word 'saves'"),
        ("This plan will retain more.", "banned word 'will retain'"),
        ("It prevents churn.", "banned word 'prevents'"),
        ("The reminder keeps people.", "banned word 'keeps'"),
        ("The reminder works.", "banned word 'works'"),
        ("Customers retained: 5.", "'retained' outside the registered quantity names"),
        ("The uplift is large.", "'uplift' without 'assumed'"),
        ("The net value is 5 price units.", "effect sentence without the prefix"),
        ("As on Criteo data.", "external dataset name 'Criteo'"),
    ],
)
def test_wording_scan_catches(run_dir, bad, found):
    out, cfg = run_dir
    text = render(cfg, load_logs(out)) + "\n" + bad + "\n"
    assert any(v.startswith(found) for v in wording_violations(text, cfg))


def test_works_is_allowed_only_in_the_exact_label():
    assert wording_violations("*says nothing about whether any action works.*", CFG) == []
    assert wording_violations("says nothing about whether the action works", CFG) != []


def test_missing_caption_is_caught(run_dir):
    out, cfg = run_dir
    text = render(cfg, load_logs(out)).replace(f"*{cfg.caption}.*", "*a table.*", 1)
    assert "comparison table without the registered caption" in wording_violations(text, cfg)


def readme_section() -> str:
    text = Path("README.md").read_text()
    m = re.search(r"^## Budget optimizer\n(.*?)(?=^## |\Z)", text, flags=re.S | re.M)
    assert m, "README has no Budget optimizer section"
    return m.group(0)


def test_readme_states_the_sentence_and_keeps_the_rules():
    section = readme_section()
    assert README_SENTENCE in section
    assert wording_violations(section, CFG) == []


def test_no_external_dataset_name_in_phase_7_files():
    raw = yaml.safe_load(Path("configs/actions.yaml").read_text())
    raw["wording_rules"].pop("external_figures_ban")
    texts = {"configs/actions.yaml": yaml.safe_dump(raw), "README section": readme_section()}
    for f in Path("src/retention/optimize").glob("*.py"):
        texts[str(f)] = f.read_text()
    if REPORT.exists():
        texts[str(REPORT)] = REPORT.read_text()
    for where, text in texts.items():
        for name in CFG.external_names:
            assert name.lower() not in text.lower(), (where, name)
    # the Phase 6 docs are exempt, as registered
    assert set(CFG.external_exempt) == {
        "docs/uplift_validation.md", "docs/causal_assumptions.md", "docs/counterfactuals.md",
    }  # fmt: skip


def test_committed_report_equals_the_one_rebuilt_from_the_logs():
    out = cli.out_dir(None)
    if not (REPORT.exists() and (out / "plan_report.json").exists()):
        pytest.skip("no logged plan run here")
    assert render(CFG, load_logs(out)) == REPORT.read_text()


def summary_with(unproven: int) -> dict:
    return {"cells": 112, "proven": 112 - unproven, "unproven": unproven,
            "unproven_share": unproven / 112, "time_limit_seconds": 120.0,
            "max_highs_minus_greedy": 0.0, "breaches": 0, "repairs": 0}  # fmt: skip


def test_unproven_count_is_always_reported():
    text = "\n".join(solver_lines(CFG, summary_with(3)))
    assert "not proven: 3" in text and "within 120 s" in text
    assert "The exact check is limited" not in text  # 3 of 112 is under 10%
    assert wording_violations(text, CFG) == []


def test_above_ten_percent_the_limit_is_stated_plainly():
    text = "\n".join(solver_lines(CFG, summary_with(12)))  # 10.7%
    assert "The exact check is limited: in 12 of 112 cells (10.7%)" in text
    assert wording_violations(text, CFG) == []
    assert "The exact check is limited" not in "\n".join(solver_lines(CFG, summary_with(11)))


def test_plan_report_counts_unproven_cells(run_dir):
    out, cfg = run_dir
    s = json.loads((out / "plan_report.json").read_text())["solver_summary"]
    assert s["cells"] == 112 and s["proven"] + s["unproven"] == 112
    assert s["time_limit_seconds"] == 120.0
    checks = json.loads((out / "plan_report.json").read_text())["solver_checks"]
    assert all("seconds" in c and c["exact_check"] in ("proven", "not proven") for c in checks)
