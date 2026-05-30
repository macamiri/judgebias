"""Real-data demo: measure GPT-4's judge biases on MT-Bench.

Dataset: lmsys/mt_bench_human_judgments (CC-BY-4.0) — it ships a real GPT-4
judge's pairwise verdicts (`gpt4_pair`, 2400) plus human verdicts (`human`,
3355) over 6 models, with model identities. That lets us measure all three
biases on a REAL LLM judge, with a human baseline, fully offline.

Run:
    pip install judgebias[examples]
    python examples/mtbench_judge_bias.py
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
from datasets import load_dataset

import judgebias as jb

OWN_MODEL = "gpt-4"  # the judge is GPT-4; is it among the competitors? (yes, in MT-Bench)


def assistant_chars(conversation):
    """Total characters the model produced (assistant turns only)."""
    return sum(len(m["content"]) for m in conversation if m.get("role") == "assistant")


def normalize(df):
    df = df.copy()
    df["len_a"] = df["conversation_a"].map(assistant_chars)
    df["len_b"] = df["conversation_b"].map(assistant_chars)
    df["choice"] = df["winner"].map(
        lambda w: "a" if w == "model_a" else "b" if w == "model_b" else "tie"
    )
    # MT-Bench already ran both A/B orders; "tie (inconsistent)" == the verdict flipped.
    df["inconsistent"] = df["winner"] == "tie (inconsistent)"
    df["pair_key"] = list(
        zip(df["question_id"], df["turn"],
            [frozenset((a, b)) for a, b in zip(df["model_a"], df["model_b"])])
    )
    return df


def own_win_rate_by_pair(df):
    """For pairs where OWN_MODEL competes and the verdict is decisive: fraction
    of judgments (per pair) that chose OWN_MODEL."""
    competes = (df["model_a"] == OWN_MODEL) | (df["model_b"] == OWN_MODEL)
    d = df[competes & (df["choice"] != "tie")].copy()
    own_is_a = d["model_a"] == OWN_MODEL
    d["chose_own"] = np.where(own_is_a, d["choice"] == "a", d["choice"] == "b").astype(int)
    return d.groupby("pair_key")["chose_own"].mean()


def main():
    ds = load_dataset("lmsys/mt_bench_human_judgments")
    gpt4 = normalize(ds["gpt4_pair"].to_pandas())
    human = normalize(ds["human"].to_pandas())

    # (1) position bias — GPT-4's order-inconsistency rate
    pos = jb.position_bias(gpt4["inconsistent"])

    # (2) length bias — GPT-4 judge, with a human baseline for context
    length_gpt4 = jb.length_bias(gpt4["len_a"], gpt4["len_b"], gpt4["choice"])
    length_human = jb.length_bias(human["len_a"], human["len_b"], human["choice"])

    # (3) self-preference — GPT-4 judge vs human baseline on the SAME pairs where gpt-4 competes
    g_own = own_win_rate_by_pair(gpt4)
    h_own = own_win_rate_by_pair(human)
    common = g_own.index.intersection(h_own.index)
    selfpref = jb.self_preference(g_own.loc[common].values, h_own.loc[common].values)

    report = jb.BiasReport(title="judgebias - GPT-4 judge on MT-Bench (lmsys/mt_bench_human_judgments)")
    report.add(pos).add(length_gpt4).add(selfpref)
    print(report)
    print()
    print("context / baselines:")
    print(f"  length bias, HUMAN judges:  effect = {length_human.effect:+.3f}  "
          f"95% CI [{length_human.ci_low:+.3f}, {length_human.ci_high:+.3f}]  n = {length_human.n}")
    print(f"  self-preference compared on n = {len(common)} pairs where gpt-4 is a competitor "
          f"(judged by both GPT-4 and humans)")


if __name__ == "__main__":
    main()
