"""Evaluate baseline vs agent with a REAL LLM provider (never run in the build workspace).

Configure the provider through environment variables (see .env.example), for example a local
Ollama server::

    export LLM_PROVIDER=openai_compatible
    export OPENAI_BASE_URL=http://localhost:11434/v1
    export OPENAI_MODEL=llama3.1
    python evals/run_real_provider.py --limit 5        # smoke test first
    python evals/run_real_provider.py --yes            # full 30 products

or a hosted free tier (set ``OPENAI_BASE_URL``, ``OPENAI_API_KEY``, ``OPENAI_MODEL``), or
``LLM_PROVIDER=anthropic`` with ``ANTHROPIC_API_KEY``.

Results are written to ``evals/results/real_<provider>_<timestamp>.{md,json}``. The ``mock_script``
column of the CSV is ignored: the model's own mistakes are what get measured. Cost control:
``MAX_TOTAL_TOKENS`` / ``MAX_COST_USD`` / ``MAX_LLM_CALLS`` apply per product run.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from pma.config import Settings
from pma.dataset import load_cases
from pma.errors import ConfigError
from pma.evaluation import render_markdown, run_comparison
from pma.logging_setup import configure_logging
from pma.providers import build_provider

HERE = Path(__file__).resolve().parent
# rough upper bound of LLM calls per product: baseline (1-3) + agent (4 plan + 2-3 per iteration)
CALLS_PER_PRODUCT = {"baseline": 3, "agent": 4 + 3 * 3}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", default=str(HERE / "products.csv"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N products")
    ap.add_argument("--approaches", default="baseline,agent")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args(argv)

    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if settings.llm_provider == "mock":
        print(
            "Refusing to run: LLM_PROVIDER is 'mock'. This script exists to measure a REAL model.\n"
            "Set LLM_PROVIDER=openai_compatible (or anthropic) and the matching variables.",
            file=sys.stderr,
        )
        return 2

    cases = load_cases(Path(args.csv))
    if args.limit:
        cases = cases[: args.limit]
    approaches = tuple(a.strip() for a in args.approaches.split(",") if a.strip())
    upper = sum(CALLS_PER_PRODUCT[a] for a in approaches) * len(cases)
    model = (
        settings.openai_model
        if settings.llm_provider == "openai_compatible"
        else settings.anthropic_model
    )
    print(f"Provider {settings.llm_provider} / model {model}: {len(cases)} products x {approaches}")
    print(
        f"Up to ~{upper} LLM calls (upper bound). Per-run budgets: "
        f"tokens={settings.max_total_tokens}, cost=${settings.max_cost_usd}, "
        f"calls={settings.max_llm_calls}"
    )
    if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        return 1

    configure_logging("WARNING", "text")
    settings = replace(settings, trace_dir=Path(tempfile.mkdtemp(prefix="pma-eval-")))
    provider = build_provider(settings)
    report = run_comparison(
        cases, provider, settings, approaches=approaches, save_traces=True,
        progress=lambda m: print(m, file=sys.stderr),
    )  # fmt: skip
    report.notes.append(
        f"Real-provider run on {datetime.now(UTC):%Y-%m-%d}; n={len(cases)}, single run, "
        "temperature as configured in the provider: expect run-to-run variance."
    )
    markdown = render_markdown(
        report, f"Evaluation results - REAL provider ({settings.llm_provider})"
    )
    print(markdown)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    stem = f"real_{settings.llm_provider}_{stamp}"
    (out / f"{stem}.md").write_text(markdown, encoding="utf-8")
    (out / f"{stem}.json").write_text(report.to_json(), encoding="utf-8")
    print(f"wrote {out / (stem + '.md')} and .json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
