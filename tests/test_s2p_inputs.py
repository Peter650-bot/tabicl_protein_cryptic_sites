"""Their input rebuild: label quirks, entry selection, and the author-key join."""
from __future__ import annotations

import pytest

from cryptic_tabicl import config as C
from cryptic_tabicl import s2p_inputs as S


def test_three_to_one_matches_their_table():
    assert S.three_to_one("MSE") == "M"
    assert S.three_to_one("SEP") == "S"
    assert S.three_to_one("ALA") == "A"
    assert S.three_to_one("HOH") == "X"          # unknown -> X, never an error


def test_label_positions_plain_numbers():
    assert S.label_positions([10, 11, 12, 13], {"11", "13"}) == [1, 3]


def test_label_positions_repeat_counting_quirk():
    # A run 60, 60, 60 is tried as 60/60A, 60/60B, 60/60C -- their loop counts
    # repeats instead of reading insertion codes.
    assert S.label_positions([59, 60, 60, 60, 61], {"60B"}) == [2]
    # "60" matches EVERY repeat, because the bare number is always tried too.
    assert S.label_positions([59, 60, 60, 61], {"60"}) == [1, 2]


def test_label_positions_counter_resets_after_run():
    # After a run ends, the counter resets only when the next number differs.
    assert S.label_positions([5, 5, 6, 7], {"6A", "7A"}) == [2, 3]


def test_their_entries_skip_multichain_and_pool_holos():
    entries = {
        "1abc": [{"apo_chain": "A", "apo_pocket_selection": ["A_1", "A_2"]},
                 {"apo_chain": "A", "apo_pocket_selection": ["A_2", "A_9"]}],
        "2xyz": [{"apo_chain": "A-B", "apo_pocket_selection": ["A_1", "B_4"]}],
    }
    out = S.their_test_entries(entries)
    assert set(out) == {"1abc"}
    assert out["1abc"] == ("A", {"1", "2", "9"})


def test_their_line_format():
    assert S.their_line("1abc", "A", "MKV", [0, 2]) == "1abc;A;UNKNOWN;M0 V2;MKV"


def test_key_of_uses_empty_icode():
    assert S.key_of("A", (12, "", "ALA")) == ("A", 12, "")
    assert S.key_of("A", (12, " ", "ALA")) == ("A", 12, "")
    assert S.key_of("A", (159, "A", "LYS")) == ("A", 159, "A")


biotite = pytest.importorskip("biotite")


@pytest.mark.skipif(biotite.__version__ != "0.41.2", reason="their pin is biotite 0.41.2")
def test_chain_residues_carries_insertion_codes():
    path = C.STRUCTURE_CACHE / "1dpj.cif.gz"
    if not path.exists():
        pytest.skip("structure cache not staged")
    res = S.chain_residues(path, "A")
    keys = [S.key_of("A", r) for r in res]
    assert ("A", 159, "A") in keys and ("A", 159, "B") in keys
    assert not S.find_duplicate_keys("A", res)
    assert len(S.sequence_of(res)) == len(res)
