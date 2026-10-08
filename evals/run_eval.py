"""Run the baseline-vs-agent comparison with the scripted MOCK provider (works offline).

    python evals/run_eval.py                  # prints the markdown report
    python evals/run_eval.py --out docs/eval  # also writes results.md / results.json

The numbers describe the pipeline, not a real model: see README "What the mock results do and do
not prove". For a real model use ``evals/run_real_provider.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pma.cli import main

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    argv = ["eval", "--provider", "mock", "--csv", str(here / "products.csv"), *sys.argv[1:]]
    raise SystemExit(main(argv))
