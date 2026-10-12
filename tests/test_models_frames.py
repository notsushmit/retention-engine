import json
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retention.data.sample import bucket
from retention.models.data import META_KEY, cache_path
from retention.models.frames import fit_rows, score_rows
from retention.models.protocol import Slice, load_protocol

P = load_protocol()
AUG, OCT = date(2016, 8, 1), date(2016, 10, 1)
IDS = [f"m{i}" for i in range(40)]


def labels_frame(snapshot):
    n = len(IDS)
    return pd.DataFrame(
        {
            "msno": IDS,
            "snapshot_date": [snapshot] * n,
            "in_exp30": [i % 4 != 0 for i in range(n)],  # every 4th customer is not in P2
            "segment": ["trial_only" if i % 10 == 1 else "paid" for i in range(n)],
            "churned_7": [i % 3 == 0 for i in range(n)],
            "event_observed": [1] * n,
            "days_to_event": list(range(n)),
        }
    ).astype({"churned_7": "int8"})


def write_labels(path, df, as_of=None):
    table = pa.Table.from_pandas(df, preserve_index=False)
    if as_of:
        meta = {"as_of": as_of.isoformat(), "snapshots": [AUG.isoformat()]}
        table = table.replace_schema_metadata({META_KEY: json.dumps(meta).encode()})
    pq.write_table(table, path)


@pytest.fixture
def dirs(tmp_path):
    processed, cache = tmp_path / "processed", tmp_path / "cache"
    processed.mkdir()
    cache.mkdir()
    feats = pd.DataFrame(
        [(m, t, i) for t in (AUG, OCT) for i, m in enumerate(IDS)],
        columns=["msno", "snapshot_date", "days_until_expiry"],
    )
    feats.to_parquet(processed / "features.parquet")
    write_labels(processed / "labels.parquet", pd.concat([labels_frame(AUG), labels_frame(OCT)]))
    return processed, cache


def expected_p2_main(ids):
    return [m for i, m in enumerate(IDS) if i % 4 != 0 and i % 10 != 1 and m in ids]


def test_fit_rows_keep_p2_main_and_carry_the_target(dirs):
    processed, cache = dirs
    write_labels(cache_path(cache, OCT, (AUG,)), labels_frame(AUG), as_of=OCT)
    df = fit_rows(P, processed, cache, OCT, (AUG,), ["days_until_expiry"], 7)
    assert df.msno.tolist() == expected_p2_main(IDS)
    assert (df.y == df.churned_7).all()


def test_fit_rows_refuse_a_build_stamped_later_than_the_fold_date(dirs):
    processed, cache = dirs
    write_labels(cache_path(cache, OCT, (AUG,)), labels_frame(AUG), as_of=date(2016, 12, 1))
    with pytest.raises(ValueError, match="later than the split date"):
        fit_rows(P, processed, cache, OCT, (AUG,), ["days_until_expiry"], 7)


def test_fit_rows_never_build_labels(dirs):
    processed, cache = dirs
    with pytest.raises(FileNotFoundError, match="build-labels"):
        fit_rows(P, processed, cache, OCT, (AUG,), ["days_until_expiry"], 7)


def test_score_rows_of_a_half_keep_only_that_half(dirs):
    processed, _ = dirs
    df = score_rows(P, processed, Slice(OCT, "selection"), ["days_until_expiry"], 7)
    selection = [m for m in IDS if bucket(m) % 2 == 0]
    assert df.msno.tolist() == expected_p2_main(selection)


def test_the_59_day_target_is_derived_from_the_lapse_day(dirs):
    processed, _ = dirs
    df = score_rows(P, processed, Slice(OCT), ["days_until_expiry"], 59)
    assert (df.y == (df.days_to_event <= 59).astype(int)).all()
