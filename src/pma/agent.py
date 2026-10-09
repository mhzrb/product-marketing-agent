"""The marketing agent: plan -> draft -> evaluate -> revise (bounded) -> report.

Control flow lives here in plain Python. The LLM is asked for four things only: a plan (with tool
use), a draft, a revision, and a judgement. Everything that decides pass/fail is deterministic or
audited by the evaluator, and the final "why this passed" report is assembled from evaluator
evidence, never from what the model says about itself.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from pma.config import Settings
from pma.errors import BudgetExceeded, OutputValidationError, PMAError
from pma.evaluator import Evaluator
from pma.facts import FactBase
from pma.limits import LIMITS
from pma.llm_client import LLMClient
from pma.logging_setup import get_logger
from pma.prompts import PromptStore
from pma.providers.base import ChatMessage, LLMProvider
from pma.rag.retriever import GuidelineRetriever, build_retriever
from pma.reliability import Budget
from pma.schemas import (
    AgentResult,
    CheckResult,
    EvaluationReport,
    JudgeVerdict,
    MarketingDraft,
    Plan,
    ProductInput,
    RevisionNote,
    RunReport,
    SecurityReport,
)
from pma.security import sanitize_product
from pma.tool_runtime import run_tool_loop, tool_list_text
from pma.tools import ToolRegistry
from pma.tracing import Trace

log = get_logger("agent")

MAX_GUIDELINE_CHARS = 3500


def json_for_prompt(obj: object) -> str:
    """Compact JSON that cannot close an XML-style data block (``<`` is escaped)."""
    return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c")


def limits_json() -> str:
    return json.dumps({k: {"min": lo, "max": hi} for k, (lo, hi) in LIMITS.items()})


@dataclass
class RunContext:
    """Everything one run needs; built by ``prepare_run`` and shared by agent and baseline."""

    languages: list[str]
    trace: Trace
    budget: Budget
    client: LLMClient
    prompts: PromptStore
    security: SecurityReport
    facts: FactBase
    registry: ToolRegistry
    evaluator: Evaluator
    started: float


def prepare_run(
    provider: LLMProvider,
    settings: Settings,
    product: ProductInput,
    languages: list[str],
    retriever: GuidelineRetriever,
    prompts: PromptStore,
    *,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> RunContext:
    trace = Trace(full_io=settings.trace_full_io)
    trace.meta = {
        "provider": provider.name,
        "model": provider.model,
        "prompt_version": settings.prompt_version,
        "retriever": settings.retriever,
        "max_iterations": settings.agent_max_iterations,
        "sanitize_input": settings.sanitize_input,
        "judge_enabled": settings.judge_enabled,
        "tool_transport": "native" if provider.supports_tools else "json_protocol",
        "languages": languages,
        "product_id": product.id,
    }
    budget = Budget(
        max_tokens=settings.max_total_tokens,
        max_cost_usd=settings.max_cost_usd,
        max_calls=settings.max_llm_calls,
        price_per_1k_prompt=settings.price_per_1k_prompt_usd,
        price_per_1k_completion=settings.price_per_1k_completion_usd,
    )
    client = LLMClient(provider, settings, budget, trace, prompts, sleep=sleep, rng=rng)

    with trace.span("prepare") as extra:
        cleaned, security = sanitize_product(product)
        if not settings.sanitize_input:
            cleaned = product
            security = security.model_copy(update={"sanitized": False})
        facts = FactBase.build(cleaned)
        extra.update(
            injection_detected=security.injection_detected,
            injection_rules=sorted({f.rule for f in security.findings}),
            sanitized=security.sanitized,
            data_warnings=facts.warnings,
        )
    registry = ToolRegistry(facts, retriever)
    evaluator = Evaluator(registry, facts, languages, security)
    return RunContext(
        languages,
        trace,
        budget,
        client,
        prompts,
        security,
        facts,
        registry,
        evaluator,
        time.perf_counter(),
    )


def make_judge(ctx: RunContext) -> Callable[[MarketingDraft, list[CheckResult]], JudgeVerdict]:
    judge_prompt = ctx.prompts.get("judge")
    system = ctx.prompts.get("system").render()

    def judge(draft: MarketingDraft, checks: list[CheckResult]) -> JudgeVerdict:
        summary = ", ".join(f"{c.name}={'pass' if c.passed else 'FAIL'}" for c in checks)
        user = judge_prompt.render(
            product_json=json_for_prompt(ctx.facts.prompt_view()),
            draft_json=json_for_prompt(draft.model_dump(exclude_none=True)),
            deterministic_summary=summary,
        )
        messages = [ChatMessage("system", system), ChatMessage("user", user)]
        try:
            # temperature 0: a reviewer that changes its mind between identical calls is useless
            verdict, _ = ctx.client.structured(
                "judge",
                messages,
                JudgeVerdict,
                prompt=judge_prompt,
                max_tokens=800,
                temperature=0.0,
            )
            return verdict
        except OutputValidationError as exc:
            # fail closed: an unusable judge must never silently approve a draft
            return JudgeVerdict(passed=False, reasons=[f"judge output was unusable: {exc}"])

    return judge


class MarketingAgent:
    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings,
        *,
        retriever: GuidelineRetriever | None = None,
        prompts: PromptStore | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.retriever = retriever or build_retriever(settings)
        self.prompts = prompts or PromptStore(settings.prompt_version)
        self._sleep = sleep
        self._rng = rng

    # ------------------------------------------------------------------ main entry
    def run(
        self,
        product: ProductInput,
        languages: list[str] | None = None,
        *,
        save_trace: bool = True,
    ) -> AgentResult:
        langs = list(languages or ["en", "nl"])
        ctx = prepare_run(
            self.provider, self.settings, product, langs, self.retriever, self.prompts,
            sleep=self._sleep, rng=self._rng,
        )  # fmt: skip
        status = "failed"
        error: str | None = None
        draft: MarketingDraft | None = None
        evaluations: list[EvaluationReport] = []
        revisions: list[RevisionNote] = []
        try:
            plan = self._plan(ctx)
            guidelines = self._guideline_text(ctx, product)
            judge = make_judge(ctx) if self.settings.judge_enabled else None
            for iteration in range(1, self.settings.agent_max_iterations + 1):
                feedback = evaluations[-1].feedback_lines() if evaluations else []
                try:
                    draft_new = self._write(ctx, plan, guidelines, draft, feedback, iteration)
                except OutputValidationError as exc:
                    report = EvaluationReport(
                        iteration=iteration,
                        passed=False,
                        checks=[
                            CheckResult(
                                name="schema_valid",
                                passed=False,
                                message=f"output was not valid JSON for the schema: {exc}",
                            )
                        ],
                        judge_skipped="draft invalid",
                    )
                else:
                    draft = draft_new
                    with ctx.trace.span("evaluate", iteration=iteration) as extra:
                        report = ctx.evaluator.evaluate(draft, iteration, judge)
                        extra.update(
                            passed=report.passed,
                            failed_checks=[c.name for c in report.failed_checks],
                            judge_passed=None if report.judge is None else report.judge.passed,
                        )
                evaluations.append(report)
                if report.passed:
                    status = "passed"
                    break
                revisions.append(
                    RevisionNote(iteration=iteration, problems=report.feedback_lines())
                )
        except BudgetExceeded as exc:
            status, error = "budget_exceeded", str(exc)
            log.warning("budget exceeded", extra={"run_id": ctx.trace.run_id, "reason": str(exc)})
        except PMAError as exc:
            status, error = "error", f"{exc.__class__.__name__}: {exc}"
            log.error("run failed", extra={"run_id": ctx.trace.run_id, "error": error})
        return self._finish(ctx, product, status, error, draft, evaluations, revisions, save_trace)

    # ------------------------------------------------------------------ steps
    def _plan(self, ctx: RunContext) -> Plan:
        plan_prompt = ctx.prompts.get("plan")
        system = ctx.prompts.get("system").render()
        allowed = ["lookup_product_fact", "retrieve_guidelines"]
        if ctx.client.provider.supports_tools:
            instructions = (
                "Use the provided tools through the tool-calling interface and call only the "
                "tools that are listed. When you have finished with the tools, write the final "
                "JSON object as ordinary message text. Never put the final JSON in a tool call "
                "and never call a tool named json."
            )
        else:
            instructions = ctx.prompts.get("tool_protocol").render(
                tool_list=tool_list_text(ctx.registry, allowed)
            )
        user = plan_prompt.render(
            product_json=json_for_prompt(ctx.facts.prompt_view()),
            languages=",".join(ctx.languages),
            tool_instructions=instructions,
        )
        messages = [ChatMessage("system", system), ChatMessage("user", user)]
        with ctx.trace.span("plan") as extra:
            text, transport = run_tool_loop(
                ctx.client, ctx.registry, messages,
                purpose="plan", prompt=plan_prompt, allowed=allowed,
            )  # fmt: skip
            plan, repaired = ctx.client.validate(text, Plan, purpose="plan")
            extra.update(transport=transport, repaired=repaired)
        return plan

    def _guideline_text(self, ctx: RunContext, product: ProductInput) -> str:
        if not ctx.registry.guideline_texts:  # the model skipped retrieval: do it ourselves
            ctx.registry.execute(
                "retrieve_guidelines",
                {"query": f"{ctx.facts.product.category} tone claims policy brand voice", "k": 3},
                invoked_by="agent",
            )
        parts = [f"[{ref}]\n{text}" for ref, text in ctx.registry.guideline_texts.items()]
        return "\n\n".join(parts)[:MAX_GUIDELINE_CHARS]

    def _write(
        self,
        ctx: RunContext,
        plan: Plan,
        guidelines: str,
        previous: MarketingDraft | None,
        feedback: list[str],
        iteration: int,
    ) -> MarketingDraft:
        system = ctx.prompts.get("system").render()
        common = {
            "product_json": json_for_prompt(ctx.facts.prompt_view()),
            "plan_json": json_for_prompt(plan.model_dump()),
            "guidelines": guidelines,
            "languages": ",".join(ctx.languages),
            "limits_json": limits_json(),
            "attempt": iteration,
        }
        if iteration == 1:
            purpose, prompt = "draft", ctx.prompts.get("draft")
            user = prompt.render(**common)
        else:
            purpose, prompt = "revise", ctx.prompts.get("revise")
            user = prompt.render(
                **common,
                previous_draft=json_for_prompt(previous.model_dump(exclude_none=True))
                if previous
                else "null",
                feedback="\n".join(f"- {line}" for line in feedback) or "- (none)",
            )
        messages = [ChatMessage("system", system), ChatMessage("user", user)]
        draft, _ = ctx.client.structured(
            purpose, messages, MarketingDraft, prompt=prompt, max_tokens=2500
        )
        return draft

    # ------------------------------------------------------------------ result
    def _finish(
        self,
        ctx: RunContext,
        product: ProductInput,
        status: str,
        error: str | None,
        draft: MarketingDraft | None,
        evaluations: list[EvaluationReport],
        revisions: list[RevisionNote],
        save_trace: bool,
    ) -> AgentResult:
        report = build_report(ctx, status, error, evaluations, revisions)
        usage = ctx.budget.summary()
        latency_ms = round((time.perf_counter() - ctx.started) * 1000, 2)
        result = AgentResult(
            run_id=ctx.trace.run_id,
            status=status,  # type: ignore[arg-type]
            product_id=product.id,
            languages=ctx.languages,
            draft=draft,
            iterations=len(evaluations),
            evaluations=evaluations,
            report=report,
            security=ctx.security,
            tool_calls=[e.to_record() for e in ctx.registry.executions],
            usage=usage,
            provider=self.provider.name,
            model=self.provider.model,
            latency_ms=latency_ms,
            error=error,
        )
        ctx.trace.result = {
            "status": status,
            "iterations": result.iterations,
            "summary": report.summary,
            "usage": usage.model_dump(),
            "latency_ms": latency_ms,
        }
        if save_trace:
            path = ctx.trace.save(self.settings.trace_dir)
            result.trace_path = str(path)
        log.info(
            "run finished",
            extra={
                "run_id": result.run_id,
                "status": status,
                "iterations": result.iterations,
                "llm_calls": usage.llm_calls,
                "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                "latency_ms": latency_ms,
            },
        )
        return result


def build_report(
    ctx: RunContext,
    status: str,
    error: str | None,
    evaluations: list[EvaluationReport],
    revisions: list[RevisionNote],
) -> RunReport:
    """Assemble the 'why this passed/failed' report from evaluator evidence."""
    last = evaluations[-1] if evaluations else None
    passed_names = [c.name for c in last.checks if c.passed] if last else []
    failed_names = [c.name for c in last.checks if not c.passed] if last else []
    judge_reasons = list(last.judge.reasons) if last and last.judge else []
    n = len(evaluations)
    if status == "passed" and last is not None:
        judge_part = (
            "the LLM judge approved it"
            if last.judge is not None
            else f"the LLM judge was not run ({last.judge_skipped})"
        )
        fixed = f" after {len(revisions)} revision(s)" if revisions else " on the first draft"
        summary = (
            f"Passed on iteration {n}{fixed}: all {len(last.checks)} deterministic checks passed "
            f"and {judge_part}."
        )
    elif status == "budget_exceeded":
        summary = f"Stopped before approval: {error}. The draft was not verified."
    elif status == "error":
        summary = f"Run failed: {error}. No approved draft is available."
    else:
        problems = last.feedback_lines()[:3] if last else []
        summary = (
            f"Not approved after {n} iteration(s). Remaining problems: " + " | ".join(problems)
            if problems
            else f"Not approved after {n} iteration(s)."
        )
    return RunReport(
        summary=summary,
        checks_passed=passed_names,
        checks_failed=failed_names,
        judge_reasons=judge_reasons,
        revisions=revisions,
        guidelines_used=list(ctx.registry.guideline_refs),
        data_warnings=list(ctx.facts.warnings),
    )
