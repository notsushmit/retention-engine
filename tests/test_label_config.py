from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest
import yaml

from retention.labels.config import load_label_config, parse_label_config

RAW = yaml.safe_load(Path("configs/kkbox.yaml").read_text())["labels"]


def raw(**changes):
    out = deepcopy(RAW)
    out.update(changes)
    return out


def test_project_config_loads_with_default_cutoff():
    cfg = load_label_config()
    assert cfg.as_of == date(2017, 3, 31)
    assert cfg.last_labelable_expiry == date(2017, 3, 1)
    assert cfg.snapshots[0] == date(2015, 3, 1) and cfg.snapshots[-1] == date(2017, 2, 1)
    assert cfg.partial_snapshots == (date(2017, 1, 1), date(2017, 2, 1))


def test_unknown_label_key_fails():
    with pytest.raises(ValueError, match="unknown key.*renewal_windw_days"):
        parse_label_config(raw(renewal_windw_days=30))


def test_unknown_snapshot_key_fails():
    snaps = dict(RAW["snapshots"], frequency="MS")
    with pytest.raises(ValueError, match="unknown key.*frequency"):
        parse_label_config(raw(snapshots=snaps))


def test_missing_key_fails():
    bad = raw()
    del bad["followup_cap_days"]
    with pytest.raises(ValueError, match="missing key.*followup_cap_days"):
        parse_label_config(bad)


def test_partial_snapshot_rejected_unless_allowed():
    snaps = {"dates": ["2016-12-01", "2017-01-01"]}
    with pytest.raises(ValueError, match="allow_partial"):
        parse_label_config(raw(snapshots=snaps))
    cfg = parse_label_config(raw(snapshots={**snaps, "allow_partial": True}))
    assert cfg.partial_snapshots == (date(2017, 1, 1),)


def test_last_fully_observable_snapshot_for_90_day_followup():
    cfg = parse_label_config(raw(snapshots={"dates": ["2016-12-01"]}))
    assert cfg.fully_observable(date(2016, 12, 1))  # 2016-12-01 + 90 days = 2017-03-01
    assert not cfg.fully_observable(date(2016, 12, 2))
    assert cfg.fully_observable(date(2017, 1, 1), days=59)  # 2017-03-01
    assert not cfg.fully_observable(date(2017, 1, 1), days=60)


def test_as_of_override_moves_cutoff_and_rejects_late_snapshots():
    cfg = parse_label_config(raw(snapshots={"dates": ["2016-06-01"]}), as_of=date(2016, 10, 1))
    assert cfg.last_labelable_expiry == date(2016, 9, 1)
    with pytest.raises(ValueError, match="after as_of"):
        parse_label_config(
            raw(snapshots={"dates": ["2016-11-01"], "allow_partial": True}), as_of=date(2016, 10, 1)
        )


def test_weekly_range_and_explicit_dates():
    cfg = parse_label_config(
        raw(snapshots={"start": "2016-07-04", "end": "2016-07-25", "freq": "W-MON"})
    )
    assert cfg.snapshots == (
        date(2016, 7, 4),
        date(2016, 7, 11),
        date(2016, 7, 18),
        date(2016, 7, 25),
    )
    with pytest.raises(ValueError, match="either"):
        parse_label_config(raw(snapshots={"dates": ["2016-07-01"], "start": "2016-07-01"}))
    with pytest.raises(ValueError, match="freq"):
        parse_label_config(
            raw(snapshots={"start": "2016-07-01", "end": "2016-08-01", "freq": "not-a-freq"})
        )


def test_horizon_beyond_cap_rejected():
    with pytest.raises(ValueError, match="horizons"):
        parse_label_config(raw(horizons=[7, 120]))
