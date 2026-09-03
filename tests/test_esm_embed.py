"""Index bookkeeping for ESM embeddings.

The GPU call is trivial; the failure mode is alignment. If row i of the
embedding matrix is not residue i, the result is a plausible feature matrix
that performs slightly worse -- indistinguishable from "embeddings did not
help". These tests pin the two places that can shift: chunk reassembly and
per-chain grouping.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl.esm_embed import (  # noqa: E402
    MAX_WINDOW, chain_sequences, chunk_spans,
)


@dataclass
class FakeRec:
    chain: str
    seq_id: int
    resname: str
    icode: str = ""


# ------------------------------------------------------------------ chunking
@pytest.mark.parametrize("n", [1, 2, 300, 1021, 1022, 1023, 1486, 2500, 5000])
def test_keep_intervals_tile_the_chain_exactly_once(n):
    spans = chunk_spans(n)
    covered = []
    for _s, _e, ks, ke in spans:
        covered.extend(range(ks, ke))
    assert covered == list(range(n)), f"n={n} does not tile"


@pytest.mark.parametrize("n", [1023, 1486, 2500, 5000])
def test_no_window_exceeds_the_model_context(n):
    for s, e, _ks, _ke in chunk_spans(n):
        assert e - s <= MAX_WINDOW


@pytest.mark.parametrize("n", [1023, 1486, 2500])
def test_kept_slice_lies_inside_its_own_window(n):
    """`keep` is sliced out of `emb` using `keep_start - start`; if keep ever
    fell outside [start, end) that slice would silently take the wrong rows."""
    for s, e, ks, ke in chunk_spans(n):
        assert s <= ks < ke <= e


def test_short_chain_is_a_single_untrimmed_window():
    assert chunk_spans(300) == [(0, 300, 0, 300)]


def test_reassembly_reproduces_the_original_sequence():
    """End-to-end simulation of the concatenate step in `embed_shard`."""
    n = 1486
    truth = np.arange(n)
    pieces = [truth[s:e][ks - s: ke - s] for s, e, ks, ke in chunk_spans(n)]
    assert np.array_equal(np.concatenate(pieces), truth)


def test_empty_chain_yields_no_spans():
    assert chunk_spans(0) == []


# ------------------------------------------------------------------ grouping
def test_sequence_and_keys_stay_parallel():
    recs = [FakeRec("A", 10, "ALA"), FakeRec("A", 11, "GLY"),
            FakeRec("B", 5, "TRP")]
    groups = chain_sequences(recs)
    assert [g["chain"] for g in groups] == ["A", "B"]
    assert groups[0]["seq"] == "AG"
    assert len(groups[0]["seq"]) == len(groups[0]["keys"])
    assert groups[1]["seq"] == "W"


def test_res_ix_indexes_the_original_record_list():
    """`res_ix` is the join key that survives insertion codes, so it must be a
    position in the flat record list, not a position within its own chain."""
    recs = [FakeRec("A", 1, "ALA"), FakeRec("B", 1, "GLY"), FakeRec("B", 2, "SER")]
    groups = {g["chain"]: g for g in chain_sequences(recs)}
    assert [k[2] for k in groups["A"]["keys"]] == [0]
    assert [k[2] for k in groups["B"]["keys"]] == [1, 2]


def test_insertion_codes_are_carried_not_collapsed():
    """52 and 52A are different residues; the key must distinguish them."""
    recs = [FakeRec("A", 52, "ALA", ""), FakeRec("A", 52, "GLY", "A")]
    keys = chain_sequences(recs)[0]["keys"]
    assert keys == [(52, "", 0), (52, "A", 1)]
    assert len(set(keys)) == 2


def test_interleaved_chains_do_not_merge_out_of_order():
    """gemmi emits residues chain by chain, but grouping must not depend on it."""
    recs = [FakeRec("A", 1, "ALA"), FakeRec("B", 1, "GLY"), FakeRec("A", 2, "SER")]
    groups = {g["chain"]: g for g in chain_sequences(recs)}
    assert groups["A"]["seq"] == "AS"
    assert [k[2] for k in groups["A"]["keys"]] == [0, 2]


def test_unknown_residue_becomes_X_not_dropped():
    """Dropping would shift every downstream index by one."""
    recs = [FakeRec("A", 1, "ALA"), FakeRec("A", 2, "MSE"), FakeRec("A", 3, "GLY")]
    g = chain_sequences(recs)[0]
    assert g["seq"] == "AXG"
    assert len(g["keys"]) == 3
