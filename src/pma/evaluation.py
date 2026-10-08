"""Evaluation harness: single-prompt baseline vs. the agent loop on the CSV products.

With the mock provider the numbers describe the *pipeline* (see the README section "What the mock
results do and do not prove"). With a real provider the same code measures the real model, which is
what ``evals/run_real_provider.py`` is for.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from pma.agent import MarketingAgent
from pma.baseline import BaselineGenerator
from pma.config import Settings
from pma.dataset import EvalCase
from pma.providers.base import LLMProvider
from pma.rag.retriever import build_retriever
from pma.schemas import AgentResult

SIGNALS = {
    "injection": lambda r: r.security.injection_detected,
    "conflict": lambda r: any("conflicting values" in w for w in r.report.data_warnings),
    "missing_specs": lambda r: any("no specs" in w for w in r.report.data_warnings),
    "missing_price": lambda r: any("no price" in w for w in r.report.data_warnings),
}


@dataclass
class CaseOutcome:
    id: str
    case_type: str
    status: str
    passed: bool
    iterations: int
    llm_calls: int
    total_tokens: int
    tokens_estimated: bool
    latency_ms: float
    failed_checks: list[str] = field(default_factory=list)
    passed_first_iteration: bool = False
    signals: dict[str, bool] = field(default_factory=dict)


@dataclass
class ApproachSummary:
    name: str
    n: int
    passed: int
    pass_rate: float
    avg_iterations: float
    avg_llm_calls: float
    avg_total_tokens: float
    tokens_estimated: bool
    avg_latency_ms: float
    p95_latency_ms: float
    statuses: dict[str, int]
    failure_reasons: dict[str, int]
    pass_by_case_type: dict[str, str]
    first_iteration_pass_rate: float
    outcomes: list[CaseOutcome]


def _outcome(case: EvalCase, result: AgentResult) -> CaseOutcome:
    last = result.evaluations[-1] if result.evaluations else None
    failed: list[str] = []
    if last is not None and not result.passed:
        failed = [c.name for c in last.failed_checks]
        if last.judge is not None and not last.judge.passed:
            failed.append("llm_judge")
    if result.status in ("error", "budget_exceeded"):
        failed.append(result.status)
    first = result.evaluations[0].passed if result.evaluations else False
    return CaseOutcome(
        id=case.product.id or "",
        case_type=case.case_type,
        status=result.status,
        passed=result.passed,
        iterations=result.iterations,
        llm_calls=result.usage.llm_calls,
        total_tokens=result.usage.total_tokens,
        tokens_estimated=result.usage.tokens_estimated,
        latency_ms=result.latency_ms,
        failed_checks=failed,
        passed_first_iteration=first,
        signals={s: SIGNALS[s](result) for s in case.expect if s in SIGNALS},
    )


def summarize(name: str, outcomes: list[CaseOutcome]) -> ApproachSummary:
    n = len(outcomes)
    latencies = sorted(o.latency_ms for o in outcomes)
    reasons: Counter[str] = Counter()
    for o in outcomes:
        reasons.update(o.failed_checks)
    by_type: dict[str, list[bool]] = {}
    for o in outcomes:
        by_type.setdefault(o.case_type, []).append(o.passed)
    p95_index = max(0, min(n - 1, int(round(0.95 * n)) - 1))
    return ApproachSummary(
        name=name,
        n=n,
        passed=sum(o.passed for o in outcomes),
        pass_rate=sum(o.passed for o in outcomes) / n if n else 0.0,
        avg_iterations=statistics.fmean(o.iterations for o in outcomes) if n else 0.0,
        avg_llm_calls=statistics.fmean(o.llm_calls for o in outcomes) if n else 0.0,
        avg_total_tokens=statistics.fmean(o.total_tokens for o in outcomes) if n else 0.0,
        tokens_estimated=any(o.tokens_estimated for o in outcomes),
        avg_latency_ms=statistics.fmean(o.latency_ms for o in outcomes) if n else 0.0,
        p95_latency_ms=latencies[p95_index] if n else 0.0,
        statuses=dict(Counter(o.status for o in outcomes)),
        failure_reasons=dict(reasons.most_common()),
        pass_by_case_type={t: f"{sum(v)}/{len(v)}" for t, v in sorted(by_type.items())},
        first_iteration_pass_rate=(
            sum(o.passed_first_iteration for o in outcomes) / n if n else 0.0
        ),
        outcomes=outcomes,
    )


@dataclass
class ComparisonReport:
    provider: str
    model: str
    n_cases: int
    baseline: ApproachSummary | None
    agent: ApproachSummary | None
    signal_detection: dict[str, str]
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def run_comparison(
    cases: list[EvalCase],
    provider: LLMProvider,
    settings: Settings,
    *,
    approaches: tuple[str, ...] = ("baseline", "agent"),
    progress: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] | None = None,
    save_traces: bool = False,
) -> ComparisonReport:
    retriever = build_retriever(settings)
    kwargs: dict[str, Any] = {"retriever": retriever}
    if sleep is not None:
        kwargs["sleep"] = sleep
    summaries: dict[str, ApproachSummary] = {}
    signal_hits: Counter[str] = Counter()
    signal_total: Counter[str] = Counter()

    if "baseline" in approaches:
        runner = BaselineGenerator(provider, settings, **kwargs)
        outcomes = []
        for case in cases:
            if progress:
                progress(f"baseline {case.product.id}")
            outcomes.append(_outcome(case, runner.run(case.product, case.languages)))
        summaries["baseline"] = summarize("single-prompt baseline", outcomes)

    if "agent" in approaches:
        agent = MarketingAgent(provider, settings, **kwargs)
        outcomes = []
        for case in cases:
            if progress:
                progress(f"agent    {case.product.id}")
            result = agent.run(case.product, case.languages, save_trace=save_traces)
            outcome = _outcome(case, result)
            outcomes.append(outcome)
            for sig, hit in outcome.signals.items():
                signal_total[sig] += 1
                signal_hits[sig] += int(hit)
        summaries["agent"] = summarize("agent loop", outcomes)

    return ComparisonReport(
        provider=provider.name,
        model=provider.model,
        n_cases=len(cases),
        baseline=summaries.get("baseline"),
        agent=summaries.get("agent"),
        signal_detection={s: f"{signal_hits[s]}/{signal_total[s]}" for s in sorted(signal_total)},
    )


# --------------------------------------------------------------------------- rendering


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_markdown(report: ComparisonReport, heading: str = "Evaluation results") -> str:
    b, a = report.baseline, report.agent
    lines = [f"## {heading}", ""]
    lines.append(
        f"Provider: `{report.provider}` / model `{report.model}` - {report.n_cases} products"
    )
    lines.append("")

    def row(label: object, base: object, agent: object) -> None:
        lines.append(f"| {label} | {base} | {agent} |")

    if b and a:
        est = " (estimated, ~4 chars/token)" if (a.tokens_estimated or b.tokens_estimated) else ""
        lines += ["| Metric | Single-prompt baseline | Agent loop |", "|---|---|---|"]
        row(
            "Pass rate",
            f"{_pct(b.pass_rate)} ({b.passed}/{b.n})",
            f"{_pct(a.pass_rate)} ({a.passed}/{a.n})",
        )
        row(
            "Passed on first draft",
            _pct(b.first_iteration_pass_rate),
            _pct(a.first_iteration_pass_rate),
        )
        row(
            "Avg iterations (draft+evaluate cycles)",
            f"{b.avg_iterations:.2f}",
            f"{a.avg_iterations:.2f}",
        )
        row("Avg LLM calls per product", f"{b.avg_llm_calls:.1f}", f"{a.avg_llm_calls:.1f}")
        row(f"Avg total tokens{est}", f"{b.avg_total_tokens:.0f}", f"{a.avg_total_tokens:.0f}")
        row("Avg wall-clock latency", f"{b.avg_latency_ms:.1f} ms", f"{a.avg_latency_ms:.1f} ms")
        row("p95 wall-clock latency", f"{b.p95_latency_ms:.1f} ms", f"{a.p95_latency_ms:.1f} ms")
        row("Run statuses", b.statuses, a.statuses)
        lines += [
            "",
            "### Most common failure reasons (final evaluation of runs that did not pass)",
            "",
            "| Check | Baseline | Agent |",
            "|---|---|---|",
        ]
        checks = sorted(
            set(b.failure_reasons) | set(a.failure_reasons),
            key=lambda c: -(b.failure_reasons.get(c, 0) + a.failure_reasons.get(c, 0)),
        )
        for check in checks or ["(none)"]:
            row(f"`{check}`", b.failure_reasons.get(check, 0), a.failure_reasons.get(check, 0))
        lines += [
            "",
            "### Pass rate by case type",
            "",
            "| Case type | Baseline | Agent |",
            "|---|---|---|",
        ]
        for case_type in sorted(set(b.pass_by_case_type) | set(a.pass_by_case_type)):
            row(
                case_type,
                b.pass_by_case_type.get(case_type, "-"),
                a.pass_by_case_type.get(case_type, "-"),
            )
    else:
        only = a or b
        if only:
            lines.append(
                f"{only.name}: pass rate {_pct(only.pass_rate)} ({only.passed}/{only.n}), "
                f"avg iterations {only.avg_iterations:.2f}, avg LLM calls {only.avg_llm_calls:.1f}"
            )
    if report.signal_detection:
        lines += [
            "",
            "### Pipeline signals (deterministic, independent of the model)",
            "",
            "| Signal | Flagged / expected |",
            "|---|---|",
        ]
        lines += [f"| {s} | {v} |" for s, v in report.signal_detection.items()]
    for note in report.notes:
        lines += ["", f"> {note}"]
    return "\n".join(lines) + "\n"
