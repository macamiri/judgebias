"""Real-data demo: measure GPT-4's judge biases on MT-Bench.

The demo now lives *inside* the installed package (``judgebias/demo.py``) so that
pip-only users can run it without cloning::

    pip install "judgebias[examples]"
    python -m judgebias.demo          # or the console script: judgebias-demo

This script is the from-a-clone equivalent and produces identical output:

    pip install -e ".[examples]"
    python examples/mtbench_judge_bias.py
"""
from judgebias.demo import main

if __name__ == "__main__":
    main()
