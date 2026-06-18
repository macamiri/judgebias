"""judgebias — point it at your LLM judge + your judgments, get per-bias effect
sizes with 95% CIs and concrete corrections.

Quickstart (bring your own judge)::

    import judgebias as jb

    # 1) position bias: run YOUR judge in both A/B orders
    pairs = [(i, resp_a[i], resp_b[i]) for i in range(len(resp_a))]
    swapped = jb.swap_and_judge(pairs, my_judge)          # my_judge(first, second) -> "first"|"second"|"tie"
    print(jb.position_bias(swapped["inconsistent"]))

    # 2) length bias: over judgments you already have
    print(jb.length_bias(len_a, len_b, choice))           # choice in {"a","b","tie"}

    # 3) self-preference: your judge vs a human baseline on the same pairs
    print(jb.self_preference(judge_chose_own, human_chose_own))

See ``examples/mtbench_judge_bias.py`` for a full run on a real GPT-4 judge.
"""
from .core import (
    BiasResult,
    BiasReport,
    bootstrap_mean_ci,
    bootstrap_paired_delta_ci,
    length_bias,
    position_bias,
    swap_and_judge,
    self_preference,
    diagnose,
)

__version__ = "0.1.1"
__all__ = [
    "BiasResult",
    "BiasReport",
    "bootstrap_mean_ci",
    "bootstrap_paired_delta_ci",
    "length_bias",
    "position_bias",
    "swap_and_judge",
    "self_preference",
    "diagnose",
]
