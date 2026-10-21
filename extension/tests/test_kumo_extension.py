"""Local tests for the Kumo Tabular extension. No GPU and no `sdm` needed.

What they pin: the Kumo harness selects exactly fe1's rows (same `select`, same
`build_context` draw) and hands the model exactly fe1's 117 columns, with only
`resname` and `ss` typed categorical.

Run from the repo root: pytest extension/tests -q
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cryptic_tabicl import config as C, feature_sets as FS, sampling  # noqa: E402
from cryptic_tabicl.model import feature_matrix  # noqa: E402
from extension.kumo_model import to_stypes  # noqa: E402

PAM_RUN = C.BUILD_DIR / "ftemb_stack" / "fe1"
TABLE = C.BUILD_DIR / "residues_ftemb.parquet"


def load(path: Path, name: str):
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit as e:                     # the staged table is not on this machine
        pytest.skip(str(e))
    return mod


@pytest.fixture(scope="module")
def stack():
    return load(ROOT / "extension" / "kumo_stack.py", "kumo_stack")


@pytest.fixture(scope="module")
def table():
    if not TABLE.exists() or not PAM_RUN.exists():
        pytest.skip("build/residues_ftemb.parquet or build/ftemb_stack/fe1 missing")
    return pd.read_parquet(TABLE)


def jobs():
    return [(k, q) for k in range(4) for q in [f"sub{s}" for s in range(4)] + ["test"]]


def test_select_is_a_verbatim_copy_of_fe1s(stack):
    fe1 = load(ROOT / "deploy" / "ftemb_stack.py", "ftemb_stack")
    assert inspect.getsource(stack.select) == inspect.getsource(fe1.select)
    assert stack.SEED == fe1.SEED and stack.BUDGET == fe1.BUDGET
    assert stack.TABLE_NAME == fe1.TABLE_NAME


def test_job_tags_unique_and_file_names_match_fe1(stack):
    all_jobs = [{"budget": 1, "seed": 0, **j, "run_id": "x", "table": "t"}
                for s in stack.STAGES.values() for j in s]
    tags = [stack.job_tag(j) for j in all_jobs]
    assert len(set(tags)) == len(tags)
    names = {stack.result_name(j) for j in stack.STAGES["cv"] + stack.STAGES["test"]}
    assert names == {f"ft117_k{k}_{q}.json" for k, q in jobs()}
    assert stack.result_name(stack.STAGES["harness"][0]) == "ft117_k0_test_tabiclv2.json"
    assert {stack.result_name(j) for j in stack.STAGES["gate"]} == \
        {"ft117_k0_sub0_rep0.json", "ft117_k0_sub0_rep1.json"}


def test_to_stypes_types_only_resname_and_ss_categorical():
    X = feature_matrix(pd.DataFrame({"resname": ["ALA", "GLY"], "ss": ["H", "E"],
                                     "sasa": [1.0, None], "chain_len": [10, 10],
                                     "apo_id": ["a", "a"], "is_cryptic": [0, 1]}))
    assert to_stypes(X) == {"resname": "categorical", "ss": "categorical",
                            "sasa": "numerical", "chain_len": "numerical"}
    with pytest.raises(TypeError):
        to_stypes(pd.DataFrame({"odd": ["x", "y"]}))


@pytest.mark.parametrize("k,query", [(0, "test"), (0, "sub0")])
def test_model_sees_fe1s_117_columns_in_order(stack, table, k, query):
    ctx_src, target, t = stack.select(table, {"k": k, "query": query})
    cols = FS.resolve("ft117", t)
    keep = [c for c in t.columns if c in set(cols) | set(C.NON_FEATURE_COLS)]
    X = feature_matrix(ctx_src[keep].head(100))
    ref = json.loads((PAM_RUN / f"ft117_k{k}_{query}.json").read_text())
    assert list(X.columns) == ref["features"] and len(X.columns) == 117
    st = to_stypes(X)
    assert [c for c, v in st.items() if v == "categorical"] == ["resname", "ss"]
    assert list(st) == ref["features"]


def test_contexts_and_queries_are_fe1s(stack, table):
    for k, query in jobs():
        ctx_src, target, _ = stack.select(table, {"k": k, "query": query})
        ctx, _ = sampling.build_context(ctx_src, budget=stack.BUDGET, seed=stack.SEED)
        ref = json.loads((PAM_RUN / f"ft117_k{k}_{query}.json").read_text())
        assert len(ctx) == ref["n_context"], (k, query)
        assert ctx[C.GROUP_COL].nunique() == ref["n_context_proteins"], (k, query)
        assert target[C.GROUP_COL].tolist() == ref["apo_id"], (k, query)
        assert [int(s) for s in target["seq_id"]] == ref["seq_id"], (k, query)


def test_verdict_is_two_sided():
    sys.path.insert(0, str(ROOT / "scripts"))
    from extension.kumo_report import verdict
    assert verdict([0.001, 0.02]) == "better"
    assert verdict([-0.02, -0.001]) == "worse"
    assert verdict([-0.01, 0.01]) == "tie"
