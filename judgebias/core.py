"""judgebias.core — judge-agnostic bias diagnostics for LLM-as-judge pipelines.

You bring your own judge and your own judgments; judgebias returns, per bias, a
measured **effect size with a 95% bootstrap confidence interval** and a concrete
**correction**. There is no model to train and no benchmark to beat — just
statistics over the decisions your judge already made (or makes, via
``swap_and_judge``).

Effects are reported on an interpretable scale with an explicit *no-bias
baseline*; a result is flagged as a bias only when its 95% CI excludes that
baseline.

Confidence intervals use a nonparametric bootstrap (``n_boot=2000``, ``seed=42``).
We use bootstrap rather than a closed-form/exact test because it applies
*uniformly* to all three effect types — including the paired judge-vs-human
delta in ``self_preference`` — and runs in well under a second on these sample
sizes. See README "Architecture / how the numbers are computed".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

Verdict = str  # judge_fn output: "first" | "second" | "tie"
Choice = str   # normalized identity-space choice: "a" | "b" | "tie"


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class BiasResult:
    """One bias measured on a set of judgments."""
    name: str
    effect: float          # point estimate, on the `unit` scale
    ci_low: float
    ci_high: float
    baseline: float        # the value that means "no bias"
    n: int                 # number of judgments the estimate is based on
    unit: str              # human-readable description of the effect scale
    correction: str        # concrete recommendation to mitigate the bias
    detail: str = ""

    @property
    def present(self) -> bool:
        """True when the 95% CI excludes the no-bias baseline."""
        if math.isnan(self.ci_low) or math.isnan(self.ci_high):
            return False
        return self.ci_low > self.baseline or self.ci_high < self.baseline

    def line(self) -> str:
        flag = "BIAS" if self.present else " ok "
        return (
            f"[{flag}] {self.name:<16} {self.unit}\n"
            f"        effect = {self.effect:+.3f}   95% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}]"
            f"   (no-bias = {self.baseline:+.2f}, n = {self.n})\n"
            f"        fix: {self.correction}"
        )


@dataclass
class BiasReport:
    """A collection of BiasResults with a readable summary."""
    results: List[BiasResult] = field(default_factory=list)
    title: str = "judgebias report"

    def add(self, result: BiasResult | None) -> "BiasReport":
        if result is not None:
            self.results.append(result)
        return self

    def to_text(self) -> str:
        lines = [self.title, "=" * len(self.title)]
        for r in self.results:
            lines.append(r.line())
            lines.append("")
        flagged = [r.name for r in self.results if r.present]
        summary = f"summary: {len(flagged)}/{len(self.results)} biases flagged at 95% CI"
        summary += f" -> {', '.join(flagged)}" if flagged else " (none cleared the CI test)"
        lines.append(summary)
        return "\n".join(lines)

    __str__ = to_text


# --------------------------------------------------------------------------- #
# Bootstrap helpers
# --------------------------------------------------------------------------- #
def bootstrap_mean_ci(samples: Sequence[float], n_boot: int = 2000,
                      alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 1-D sample."""
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def bootstrap_paired_delta_ci(x: Sequence[float], y: Sequence[float], n_boot: int = 2000,
                              alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap CI for mean(x) - mean(y), resampling paired indices."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or x.size != y.size:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    deltas = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


# --------------------------------------------------------------------------- #
# Bias estimators
# --------------------------------------------------------------------------- #
def length_bias(len_a, len_b, choice, n_boot: int = 2000, seed: int = 42) -> BiasResult:
    """Verbosity/length bias: how often the judge picks the *longer* response.

    Parameters are aligned per judgment. ``choice`` is in {"a","b","tie"}; ties
    and equal-length pairs are excluded. Effect = P(judge picks longer); 0.50 = no
    length preference, > 0.50 = favors longer answers.
    """
    la = np.asarray(len_a, dtype=float)
    lb = np.asarray(len_b, dtype=float)
    ch = np.asarray(choice, dtype=object)
    keep = (ch != "tie") & (la != lb)
    la, lb, ch = la[keep], lb[keep], ch[keep]
    longer_is_a = la > lb
    chose_a = ch == "a"
    longer_won = np.where(longer_is_a, chose_a, ~chose_a).astype(float)
    n = int(longer_won.size)
    effect = float(longer_won.mean()) if n else float("nan")
    lo, hi = bootstrap_mean_ci(longer_won, n_boot=n_boot, seed=seed)
    return BiasResult(
        name="length bias", effect=effect, ci_low=lo, ci_high=hi, baseline=0.5, n=n,
        unit="P(judge picks the longer response)",
        correction=("Length-control your win-rate: regress preference on response length and "
                    "report the length-adjusted estimate (cf. length-controlled AlpacaEval, arXiv:2404.04475)."),
        detail="0.50 = no length preference; >0.50 favors longer answers.",
    )


def position_bias(inconsistent, n_boot: int = 2000, seed: int = 42) -> BiasResult:
    """Position/order bias: how often the verdict changes when A/B are swapped.

    ``inconsistent`` is a boolean per pair (True when the judge flipped under
    order swap). Produce it with ``swap_and_judge`` for your own judge, or from a
    dataset that already ran both orders. Effect = order-inconsistency rate; 0 = no
    position bias.
    """
    arr = np.asarray(inconsistent, dtype=float)
    n = int(arr.size)
    effect = float(arr.mean()) if n else float("nan")
    lo, hi = bootstrap_mean_ci(arr, n_boot=n_boot, seed=seed)
    return BiasResult(
        name="position bias", effect=effect, ci_low=lo, ci_high=hi, baseline=0.0, n=n,
        unit="order-inconsistency rate",
        correction=("Evaluate every pair in BOTH A/B orders and average; treat order-flips as ties "
                    "(the MT-Bench convention, arXiv:2306.05685)."),
        detail="0.00 = verdict never changes when A/B are swapped; higher = more order-driven flips.",
    )


def swap_and_judge(items: Iterable[Tuple], judge_fn: Callable[[str, str], Verdict]) -> pd.DataFrame:
    """Run your judge on each pair in BOTH orders to expose position bias.

    ``items``: iterable of ``(item_id, response_a, response_b)``.
    ``judge_fn(first, second) -> "first" | "second" | "tie"`` — your judge, seeing
    the two responses in the given presentation order.

    Returns a DataFrame[item_id, pick_ab, pick_ba, inconsistent] (identity-space)
    you can hand straight to ``position_bias``.
    """
    map_ab = {"first": "a", "second": "b", "tie": "tie"}
    map_ba = {"first": "b", "second": "a", "tie": "tie"}
    rows = []
    for item_id, a, b in items:
        v_ab, v_ba = judge_fn(a, b), judge_fn(b, a)
        if v_ab not in map_ab or v_ba not in map_ab:
            raise ValueError(f"judge_fn must return 'first'|'second'|'tie'; got {v_ab!r}, {v_ba!r}")
        pick_ab, pick_ba = map_ab[v_ab], map_ba[v_ba]
        inconsistent = (pick_ab != pick_ba) and ("tie" not in (pick_ab, pick_ba))
        rows.append((item_id, pick_ab, pick_ba, inconsistent))
    return pd.DataFrame(rows, columns=["item_id", "pick_ab", "pick_ba", "inconsistent"])


def self_preference(judge_chose_own, baseline_chose_own, n_boot: int = 2000, seed: int = 42) -> BiasResult:
    """Self-preference: does the judge favor its own model beyond a human baseline?

    Both inputs are aligned per pair (1 if the judge's own model was chosen, else
    0), restricted to pairs where the judge's model is a competitor. Effect =
    judge win-rate-for-own minus baseline win-rate-for-own; 0 = no self-preference.
    """
    j = np.asarray(judge_chose_own, dtype=float)
    b = np.asarray(baseline_chose_own, dtype=float)
    if j.shape != b.shape:
        raise ValueError("judge and baseline arrays must be aligned per pair (same length/order).")
    n = int(j.size)
    effect = float(j.mean() - b.mean()) if n else float("nan")
    lo, hi = bootstrap_paired_delta_ci(j, b, n_boot=n_boot, seed=seed)
    return BiasResult(
        name="self-preference", effect=effect, ci_low=lo, ci_high=hi, baseline=0.0, n=n,
        unit="judge win-rate for own model minus human baseline",
        correction=("Don't let a model grade its own family: use a different judge for self-comparisons, "
                    "or subtract this judge-vs-human gap as a debias offset."),
        detail="0.00 = judge favors its own model no more than humans do; >0 = self-preference.",
    )


def diagnose(df: pd.DataFrame, *, choice_col: str = "choice",
             len_a_col: str | None = None, len_b_col: str | None = None,
             inconsistent_col: str | None = None,
             n_boot: int = 2000, seed: int = 42, title: str = "judgebias report") -> BiasReport:
    """Convenience: run every bias the given judgments frame supports.

    Provide the columns you have: lengths -> length bias; an inconsistency flag ->
    position bias. (self_preference needs a separate human-baseline frame; call it
    directly.)
    """
    rep = BiasReport(title=title)
    if len_a_col and len_b_col:
        rep.add(length_bias(df[len_a_col], df[len_b_col], df[choice_col], n_boot=n_boot, seed=seed))
    if inconsistent_col:
        rep.add(position_bias(df[inconsistent_col], n_boot=n_boot, seed=seed))
    return rep
