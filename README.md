# judgebias

**Point `judgebias` at your LLM judge and your judgments — it returns, per bias, a measured effect size with a 95% confidence interval and a concrete correction.** No model to train, no benchmark to beat: just statistics over the decisions your judge already made.

LLM-as-judge is now standard for model selection, prompt A/B tests, and release gating. But judges have systematic habits — they favor the response shown **first**, the **longer** one, and answers from their **own** model family — and those habits silently bend the win-rates teams ship decisions on. `judgebias` measures how much each habit is bending *your* numbers.

```python
import judgebias as jb

# your judge: judge_fn(first, second) -> "first" | "second" | "tie"
swapped = jb.swap_and_judge(pairs, my_judge)     # runs each pair in both A/B orders
print(jb.position_bias(swapped["inconsistent"])) # -> effect + 95% CI + the fix
```

---

## Who this is for

Teams increasingly use one AI model to grade another model's answers — it is fast and cheap. But the grader has hidden habits: it tends to favor whichever answer it sees first, the longer answer, and answers written in its own style. These habits quietly bend the scores teams rely on to pick a model, compare prompts, or sign off a release — so the apparent winner may have just been longer, or first, not better. `judgebias` is for the engineers running those automated comparisons. Today they either trust the grader blindly or read research papers and adjust by hand. Point it at your own grader and your own results: it measures how strongly each habit is skewing your scores and gives a concrete, specific way to correct each one — no paper required.

---

## What it measures (v0.1)

| Bias | Question it answers | Effect (no-bias baseline) |
|---|---|---|
| **position** | Does the verdict change when you swap A/B? | order-inconsistency rate (0.0) |
| **length** | Does the judge prefer the longer response? | P(picks longer) (0.5) |
| **self-preference** | Does the judge favor its own model beyond a human baseline? | judge − human win-rate for own model (0.0) |

Each result is flagged `BIAS` only when its 95% CI excludes the no-bias baseline.

---

## Real example: GPT-4 grading MT-Bench

This is the **actual, unedited output** of [`examples/mtbench_judge_bias.py`](examples/mtbench_judge_bias.py), run on [`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) — a dataset that ships a real GPT-4 judge's 2,400 pairwise verdicts plus 3,355 human verdicts over 6 models, so all three biases can be measured on a real judge with a human baseline, fully offline:

```
judgebias - GPT-4 judge on MT-Bench (lmsys/mt_bench_human_judgments)
====================================================================
[BIAS] position bias    order-inconsistency rate
        effect = +0.158   95% CI [+0.144, +0.173]   (no-bias = +0.00, n = 2400)
        fix: Evaluate every pair in BOTH A/B orders and average; treat order-flips as ties (the MT-Bench convention, arXiv:2306.05685).

[BIAS] length bias      P(judge picks the longer response)
        effect = +0.735   95% CI [+0.715, +0.756]   (no-bias = +0.50, n = 1792)
        fix: Length-control your win-rate: regress preference on response length and report the length-adjusted estimate (cf. length-controlled AlpacaEval, arXiv:2404.04475).

[BIAS] self-preference  judge win-rate for own model minus human baseline
        effect = +0.119   95% CI [+0.083, +0.154]   (no-bias = +0.00, n = 430)
        fix: Don't let a model grade its own family: use a different judge for self-comparisons, or subtract this judge-vs-human gap as a debias offset.

summary: 3/3 biases flagged at 95% CI -> position bias, length bias, self-preference

context / baselines:
  length bias, HUMAN judges:  effect = +0.681  95% CI [+0.664, +0.699]  n = 2562
  self-preference compared on n = 430 pairs where gpt-4 is a competitor (judged by both GPT-4 and humans)
```

**Read that as:** on MT-Bench, GPT-4-as-judge flips its verdict on **15.8%** of pairs when you swap the order; picks the **longer** answer **73.5%** of the time (humans: 68.1% — the judge is *more* length-biased than people); and prefers its **own** model **+11.9 points** more often than humans do. Every number has a CI and a fix.

*On self-preference specifically:* on the 430 MT-Bench pairs where `gpt-4` is one of the two answers, GPT-4-**as-judge** chose gpt-4's answer **11.9 percentage points** more often than the **human raters** did on those *same* pairs (95% CI [+8.3, +15.4]). Because the human rate is the baseline, that gap is the judge's *excess* preference for its own model — over and above the quality humans already credited it.

---

## Install

```bash
pip install judgebias
pip install "judgebias[examples]"  # for demo scripts
```

---

## Quickstart — bring your own judge

`judgebias` is judge-agnostic: it never calls a model itself. You supply your judge as a function (GPT-4o, Claude, Gemini, a fine-tune — anything), or you supply judgments you already have.

```python
import judgebias as jb

# --- position bias: judgebias drives YOUR judge in both orders ---
# my_judge(first, second) -> "first" | "second" | "tie"
pairs = [(qid, resp_a, resp_b), ...]
swapped = jb.swap_and_judge(pairs, my_judge)
print(jb.position_bias(swapped["inconsistent"]))

# --- length bias: over judgments you already have ---
# choice in {"a", "b", "tie"}; len_a/len_b are response lengths (chars, tokens — your call)
print(jb.length_bias(len_a, len_b, choice))

# --- self-preference: your judge vs a human (or reference) baseline on the SAME pairs ---
# 1 if the judge's own model was chosen on that pair, else 0
print(jb.self_preference(judge_chose_own, human_chose_own))

# --- or let judgebias run everything a judgments frame supports ---
report = jb.diagnose(df, choice_col="choice",
                     len_a_col="len_a", len_b_col="len_b",
                     inconsistent_col="inconsistent")
print(report)
```

---

## How the numbers are computed (architecture)

- **Confidence intervals are a nonparametric percentile bootstrap** (`n_boot=2000`, `seed=42`). We use bootstrap rather than a closed-form/exact test because it applies *uniformly* to all three effect types — including the **paired** judge-vs-human delta in `self_preference`, where a textbook proportion interval doesn't fit — and it runs in well under a second at these sample sizes. Everything is seeded, so reported numbers are reproducible.
- **Effects are reported on interpretable scales with an explicit no-bias baseline**, and a result is only flagged `BIAS` when its CI excludes that baseline — so the tool never cries wolf on a noisy estimate.
- **The core is pure `numpy`/`pandas`.** No deep-learning dependency; the bias math is statistics, not ML.
- **Position bias** is the order-inconsistency rate. For your own judge, `swap_and_judge` generates it by running both orders. In the MT-Bench demo it is read from the dataset's `tie (inconsistent)` label — MT-Bench already ran both orders and recorded the flips.

---

## Why this exists (vs the alternatives)

| Tool | What it does | Why it isn't this |
|---|---|---|
| **length-controlled AlpacaEval** (arXiv:2404.04475) | Corrects **length** bias | One bias, inside *its own* benchmark — not your judge, not your data |
| **BiasScope** (ICLR 2026) | *Discovers* novel biases by attacking a judge on benchmark sets | Research code (not pip-installable); benchmark-driven discovery, not a point-at-your-own-judgments diagnostic |
| **CALM** (arXiv:2410.02736) | A 12-bias taxonomy + quantification study | Paper + dataset; no installable package |
| eval harnesses (lm-eval-harness, inspect-ai, Braintrust) | Run judge-based scoring | Treat the judge as a trusted oracle — no per-bias diagnostic |

**Why a bring-your-own-judge *workflow* is the contribution.** The individual estimators are not new: MT-Bench (arXiv:2306.05685) measured order sensitivity, length-controlled AlpacaEval (arXiv:2404.04475) corrects verbosity, and CALM (arXiv:2410.02736) catalogued twelve judge biases. What did *not* exist is an installed, judge-agnostic tool you point at **your own judge and your own judgments** to get per-bias effect sizes, confidence intervals, and concrete corrections in a single workflow. CALM ships as a paper plus a dataset; AlpacaEval corrects one bias inside its own benchmark; BiasScope is research code that auto-discovers biases on fixed benchmarks. `judgebias` packages the measurement-to-correction loop for the judge you actually run — the product is the workflow, not the paper.

---

## When NOT to use this

- **You don't control the judge.** Fixed leaderboards (Chatbot Arena, MMLU) — there is no judge of yours to debias.
- **Your judgment set is small (< ~400 pairs).** The bootstrap CIs get too wide to act on — collect more judgments first. (For reference, the self-preference example here runs on n = 430 and its CI is already ±0.035.)
- **You need causal debiasing, not measurement.** `judgebias` *diagnoses* biases and recommends corrections; it does not re-train or causally intervene on your judge.
- **Single-model, pointwise** evaluation with no pairwise comparison — position and self-preference don't apply (length still can).
- You need **conformal prediction intervals** for scores (use MAPIE/crepes), or you want to **discover unknown** biases rather than measure known ones (use BiasScope).

## Limitations (honest)

- **Length and self-preference are correlational.** A judge picking the longer answer 73.5% of the time may partly reflect that longer answers are genuinely better; the recommended fix (length-controlled win-rate) is the right next step, and self-preference is reported *relative to a human baseline* precisely to net out genuine quality.
- **The MT-Bench demo is one judge (GPT-4) over six 2023-era models.** It demonstrates the tool on a real judge; it is not a claim about every judge. Point the tool at *your* setup for numbers that apply to you.
- **`swap_and_judge` calls your judge twice per pair** (both orders) — budget for 2× judge calls when measuring position bias on your own judge.
- v0.1 covers position, length, and self-preference. Formatting and sycophancy biases are planned (they need a controlled-perturbation harness).

## Reproduce the demo

```bash
pip install -e ".[examples]"
python examples/mtbench_judge_bias.py     # prints the block shown above; seeded, deterministic
pytest -q                                  # 8 correctness tests (inject a known bias, assert recovery)
```

**Second example — your own judge on Chatbot Arena.** [`examples/chatbot_arena_demo.py`](examples/chatbot_arena_demo.py) runs the same diagnostics on `lmsys/chatbot_arena_conversations` with a judge you supply; it needs HuggingFace access (accept the dataset terms + a token) and an LLM judge API key, so it is not run in CI.

---

## Notes

Built with assistance from AI coding agents; the architecture, methodology, statistics, and analysis are the author's.

Grounding: judgebias's three measurements trace to published work — position bias and the swap-and-average correction to [1], length-controlled win-rates to [2], and the bias taxonomy to [3] (see [References](#references)). Demo data: `lmsys/mt_bench_human_judgments` (CC-BY-4.0).

## References

[1] Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023 Datasets and Benchmarks Track. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)

[2] Dubois et al. (2024). *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators*. COLM 2024. [arXiv:2404.04475](https://arxiv.org/abs/2404.04475)

[3] Ye et al. (2024). *Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge*. arXiv preprint. [arXiv:2410.02736](https://arxiv.org/abs/2410.02736)

[4] Lai et al. (2026). *BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation*. ICLR 2026. [arXiv:2602.09383](https://arxiv.org/abs/2602.09383)

## License

MIT — see [LICENSE](LICENSE).
