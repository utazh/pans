"""CPU-only regression tests; no GPU/performance claims are made here."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import random
import sys
from pathlib import Path

import pytest

from contiguous_fuxian.promixed import select_promixed_gqa_blocks as select


@pytest.fixture(scope="module")
def archived_select():
    path = Path(__file__).parent / "fixtures" / "promixed_snapshot.py"
    data = path.read_bytes()
    assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == (
        "a0dc97855b7aecae94ff995cd36699ffd037b53c"
    )
    spec = importlib.util.spec_from_file_location("prism_archived_selection", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.select_promixed_gqa_blocks


def test_zero_coverage_is_actually_off(archived_select):
    rows = [[10.0, 0.0, 9.0], [0.0, 10.0, 9.0]]
    kwargs = dict(keep_blocks=1, coverage_fraction=0.0,
                  utility_max_weight=0.0, utility_mean_weight=1.0,
                  utility_vote_weight=0.0)
    assert archived_select(rows, **kwargs).selected_blocks == (0,)
    assert select(rows, **kwargs).selected_blocks == (2,)


def test_utility_weights_work_when_zero_coverage_and_budget_below_group_count():
    rows = [[10.0, 0.0, 9.0], [0.0, 10.0, 9.0]]
    common = dict(keep_blocks=1, coverage_fraction=0.0, utility_vote_weight=0.0)
    assert select(rows, utility_max_weight=1.0, utility_mean_weight=0.0,
                  **common).selected_blocks == (0,)
    assert select(rows, utility_max_weight=0.0, utility_mean_weight=1.0,
                  **common).selected_blocks == (2,)


def test_adaptive_zero_coverage_also_means_no_reservation():
    decision = select([[10, 0, 9], [0, 10, 9]], keep_blocks=1,
                      coverage_fraction=1.0, adaptive_coverage=True,
                      p4_threshold=0.90, p2_threshold=0.95, p1_threshold=0.99,
                      utility_max_weight=0.0, utility_mean_weight=1.0,
                      utility_vote_weight=0.0)
    assert decision.effective_coverage_fraction == 0.0
    assert decision.selected_blocks == (2,)


@pytest.mark.parametrize("period", [1, 2, 4, 8])
def test_fixed_period_is_not_changed_by_uncertainty(period):
    agreed = [[9, 8, 0, 0]] * 4
    disagreed = [[9, 8, 7.999, 7.999], [7.999, 7.999, 9, 8]]
    for rows in (agreed, disagreed):
        decision = select(rows, keep_blocks=2, reuse_mode="fixed",
                          max_period=period, sensitivity_risk=1.0)
        assert decision.period == period


def test_fixed_mode_does_not_change_selection_at_same_layer_budget():
    rng = random.Random(617)
    for _ in range(100):
        rows = [[rng.random() for _ in range(24)] for _ in range(4)]
        adaptive = select(rows, keep_blocks=6)
        fixed = select(rows, keep_blocks=6, reuse_mode="fixed", max_period=4)
        assert fixed.selected_blocks == adaptive.selected_blocks
        assert fixed.priority_blocks == adaptive.priority_blocks
        assert fixed.agreement == adaptive.agreement


def test_legacy_positive_coverage_default_is_preserved(archived_select):
    rng = random.Random(614)
    for _ in range(500):
        groups = rng.randint(1, 8)
        blocks = rng.randint(2, 35)
        rows = [[rng.random() for _ in range(blocks)] for _ in range(groups)]
        kwargs = dict(keep_blocks=rng.randint(1, blocks),
                      coverage_fraction=rng.choice([0.01, 0.5, 1.0]),
                      max_period=rng.choice([1, 2, 4, 8]),
                      sensitivity_risk=rng.random())
        assert dataclasses.asdict(select(rows, **kwargs)) == dataclasses.asdict(
            archived_select(rows, **kwargs)
        )


@pytest.mark.parametrize("budget", [True, 1.5, "1", None])
def test_noninteger_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="integer"):
        select([[1, 2]], keep_blocks=budget)


@pytest.mark.parametrize("budget", [0, -1, 3])
def test_out_of_range_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="must be in"):
        select([[1, 2]], keep_blocks=budget)


@pytest.mark.parametrize("period", [True, 1.5, 0, 3, 16])
def test_invalid_fixed_period_is_rejected(period):
    with pytest.raises(ValueError):
        select([[1, 2]], keep_blocks=1, reuse_mode="fixed", max_period=period)


def test_unknown_reuse_mode_is_rejected():
    with pytest.raises(ValueError, match="reuse mode"):
        select([[1, 2]], keep_blocks=1, reuse_mode="unrecognized")


def test_all_zero_scores_have_deterministic_budget():
    decision = select([[0] * 8] * 4, keep_blocks=3, coverage_fraction=0.0,
                      reuse_mode="fixed", max_period=1)
    assert decision.selected_blocks == (0, 1, 2)
    assert decision.period == 1


def test_numpy_integer_budget_is_accepted():
    np = pytest.importorskip("numpy")
    assert len(select([[1, 2]], keep_blocks=np.int64(1)).selected_blocks) == 1
