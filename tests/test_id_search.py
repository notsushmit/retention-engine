"""The shared ID search finds an ID anywhere but inside a hash (D-265)."""

import pytest

from id_search import assert_no_ids, blank_hashes, ids_in

HASH = "0" * 20 + "c022" + "b" * 40  # 64 hex characters holding the ID by chance


def test_an_id_inside_a_hash_is_not_found():
    assert blank_hashes(f'{{"config_sha256":"{HASH}"}}') == '{"config_sha256":" "}'
    assert ids_in(f'{{"config_sha256":"{HASH}"}}', ["c022"]) == []
    assert_no_ids(f'{{"h":"{HASH}"}}', ["c022"])


@pytest.mark.parametrize(
    "text",
    [
        '{"x":"c022"}',  # as a value
        '{"c022":1}',  # as a key
        '{"x":"id-c022-q"}',  # inside a longer word
        '{"x":"c022ffffffff"}',  # followed by a short hex run (under 32)
        "plain c022 text",
        f'{{"h":"{HASH}","x":"c022"}}',  # a hash and an ID in the same body
        "a" * 20 + "c022",  # 24 hex characters in a row: shorter than a hash run
    ],
)
def test_an_id_anywhere_else_is_found(text):
    assert ids_in(text, ["zzz", "c022"]) == ["c022"]
    with pytest.raises(AssertionError, match="customer IDs found"):
        assert_no_ids(text, ["c022"], where="planted")


def test_the_failure_message_names_where_and_what():
    with pytest.raises(AssertionError, match=r"/model: customer IDs found: \['c022'\]"):
        assert_no_ids('{"x":"c022"}', ["c022"], where="/model")
