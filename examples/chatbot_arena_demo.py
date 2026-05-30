"""Second example: measure YOUR judge's biases on Chatbot Arena prompts.

Unlike the MT-Bench demo (which uses a real GPT-4 judge already baked into the
data), this one needs two things you provide:

  1. HuggingFace access to ``lmsys/chatbot_arena_conversations`` — a GATED dataset.
     Accept the terms on the dataset page, then authenticate:
         huggingface-cli login          # or: export HF_TOKEN=...
  2. An LLM judge. Set OPENAI_API_KEY to use the built-in GPT-4o judge (needs
     ``pip install openai``), or call ``main(judge_fn=...)`` with your own
     ``judge_fn(first, second) -> "first" | "second" | "tie"``.

Chatbot Arena has no systematic A/B swaps, so ``judgebias.swap_and_judge`` drives
your judge over BOTH orders to measure position bias directly.

    python examples/chatbot_arena_demo.py
"""
import os

import judgebias as jb

N_PAIRS = 200  # swap_and_judge calls your judge twice per pair — keep the budget modest


def openai_judge(first, second):
    """Minimal GPT-4o pairwise judge (requires OPENAI_API_KEY + ``pip install openai``)."""
    from openai import OpenAI

    prompt = (
        "Compare two assistant responses. Reply with exactly one token: "
        "A if the first is better, B if the second is better, or TIE.\n\n"
        f"[First]\n{first}\n\n[Second]\n{second}\n\nVerdict:"
    )
    out = (
        OpenAI()
        .chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        .choices[0].message.content.strip().upper()
    )
    return {"A": "first", "B": "second"}.get(out[:1], "tie")


def _assistant_text(conversation):
    return " ".join(m["content"] for m in conversation if m.get("role") == "assistant")


def main(judge_fn=None):
    judge_fn = judge_fn or openai_judge
    if judge_fn is openai_judge and "OPENAI_API_KEY" not in os.environ:
        print("Set OPENAI_API_KEY to use the built-in GPT-4o judge, or call "
              "main(judge_fn=your_judge). See the module docstring.")
        return

    try:
        from datasets import load_dataset
        ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    except Exception as e:  # gated dataset / not authenticated / offline
        print("Could not load the gated dataset lmsys/chatbot_arena_conversations.")
        print("Accept its terms on the HuggingFace dataset page, then authenticate:")
        print("  huggingface-cli login          # or: export HF_TOKEN=...")
        print(f"(underlying error: {type(e).__name__})")
        return

    pairs, len_a, len_b = [], [], []
    for row in ds:
        a, b = _assistant_text(row["conversation_a"]), _assistant_text(row["conversation_b"])
        if a and b:
            pairs.append((row.get("question_id", len(pairs)), a, b))
            len_a.append(len(a))
            len_b.append(len(b))
        if len(pairs) >= N_PAIRS:
            break

    swapped = jb.swap_and_judge(pairs, judge_fn)
    report = jb.BiasReport(title=f"judgebias - your judge on Chatbot Arena (n={len(pairs)} pairs)")
    report.add(jb.position_bias(swapped["inconsistent"]))
    report.add(jb.length_bias(len_a, len_b, swapped["pick_ab"].tolist()))
    print(report)


if __name__ == "__main__":
    main()
