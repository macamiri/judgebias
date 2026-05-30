"""Correctness tests: inject a KNOWN bias into a deterministic mock judge and
assert the estimator recovers it. No network, no API — fully offline."""
import numpy as np

import judgebias as jb


def test_position_bias_detects_always_first_judge():
    # judge that always picks whatever is shown FIRST -> flips on every swap -> ~1.0
    items = [(i, f"A{i}", f"B{i}") for i in range(200)]
    df = jb.swap_and_judge(items, lambda first, second: "first")
    res = jb.position_bias(df["inconsistent"])
    assert res.effect > 0.95
    assert res.present  # 95% CI excludes 0


def test_position_bias_clean_for_content_judge():
    # judge that picks by stable content (lexicographically larger) -> order-consistent
    items = [(i, f"A{i}", f"B{i}") for i in range(200)]
    judge = lambda first, second: "first" if first > second else "second"
    df = jb.swap_and_judge(items, judge)
    res = jb.position_bias(df["inconsistent"])
    assert res.effect < 0.05
    assert not res.present


def test_length_bias_detects_longer_preference():
    rng = np.random.default_rng(0)
    n = 300
    len_a = rng.integers(10, 100, n)
    len_b = rng.integers(10, 100, n)
    choice = np.where(len_a > len_b, "a", "b")  # judge always picks the longer
    res = jb.length_bias(len_a, len_b, choice)
    assert res.effect > 0.95
    assert res.present


def test_length_bias_clean_for_content_independent_choice():
    rng = np.random.default_rng(1)
    n = 600
    len_a = rng.integers(10, 100, n)
    len_b = rng.integers(10, 100, n)
    # choice independent of length
    choice = np.where(rng.random(n) < 0.5, "a", "b")
    res = jb.length_bias(len_a, len_b, choice)
    assert abs(res.effect - 0.5) < 0.1
    assert not res.present  # CI brackets 0.50


def test_self_preference_sign_and_ci():
    rng = np.random.default_rng(2)
    n = 400
    judge_own = (rng.random(n) < 0.80).astype(int)   # judge favors own 80%
    human_own = (rng.random(n) < 0.50).astype(int)   # humans 50%
    res = jb.self_preference(judge_own, human_own)
    assert res.effect > 0.15
    assert res.present
    # symmetric (no self-preference) case
    res2 = jb.self_preference((rng.random(n) < 0.5).astype(int), (rng.random(n) < 0.5).astype(int))
    assert not res2.present


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    x = rng.random(500)  # non-degenerate sample, mean ~0.5
    lo, hi = jb.bootstrap_mean_ci(x, n_boot=1000, seed=42)
    assert lo < hi                      # non-degenerate interval
    assert lo <= x.mean() <= hi         # CI brackets the sample mean


def test_report_renders_and_flags():
    items = [(i, "AAAAA", "BBBBB") for i in range(50)]
    df = jb.swap_and_judge(items, lambda a, b: "first")
    rep = jb.BiasReport().add(jb.position_bias(df["inconsistent"]))
    txt = str(rep)
    assert "position bias" in txt
    assert "summary:" in txt


def test_swap_and_judge_rejects_bad_verdict():
    import pytest
    with pytest.raises(ValueError):
        jb.swap_and_judge([(0, "x", "y")], lambda a, b: "left")
