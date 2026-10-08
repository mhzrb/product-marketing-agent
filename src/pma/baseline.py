"""Single-prompt baseline: one LLM call, no plan, no tools, no feedback loop.

The baseline output is scored by the *same* evaluator as the agent so the comparison is fair: the
only difference is that the agent gets to see the evaluator's feedback and revise.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

from pma.agent import (
    build_report,
    json_for_prompt,
    limits_json,
    make_judge,
    prepare_run,
)
from pma.config import Settings
from pma.errors import BudgetExceeded, OutputValidationError, PMAError
from pma.logging_setup import get_logger
from pma.prompts import PromptStore
from pma.providers.base import ChatMessage, LLMProvider
from pma.rag.retriever import GuidelineRetriever, build_retriever
from pma.schemas import (
    AgentResult,
    CheckResult,
    EvaluationReport,
    MarketingDraft,
    ProductInput,
)

log = get_logger("baseline")


class BaselineGenerator:
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

    def run(
        self,
        product: ProductInput,
        languages: list[str] | None = None,
        *,
        save_trace: bool = False,
    ) -> AgentResult:
        langs = list(languages or ["en", "nl"])
        ctx = prepare_run(
            self.provider, self.settings, product, langs, self.retriever, self.prompts,
            sleep=self._sleep, rng=self._rng,
        )  # fmt: skip
        ctx.trace.meta["mode"] = "baseline"
        status, error = "failed", None
        draft: MarketingDraft | None = None
        evaluations: list[EvaluationReport] = []
        try:
            prompt = self.prompts.get("baseline")
            system = self.prompts.get("system").render()
            user = prompt.render(
                product_json=json_for_prompt(ctx.facts.prompt_view()),
                languages=",".join(langs),
                limits_json=limits_json(),
            )
            messages = [ChatMessage("system", system), ChatMessage("user", user)]
            try:
                draft, _ = ctx.client.structured(
                    "baseline", messages, MarketingDraft, prompt=prompt, max_tokens=2500
                )
            except OutputValidationError as exc:
                report = EvaluationReport(
                    iteration=1,
                    passed=False,
                    checks=[
                        CheckResult(
                            name="schema_valid", passed=False, message=f"invalid output: {exc}"
                        )
                    ],
                    judge_skipped="draft invalid",
                )
            else:
                judge = make_judge(ctx) if self.settings.judge_enabled else None
                report = ctx.evaluator.evaluate(draft, 1, judge)
            evaluations.append(report)
            status = "passed" if report.passed else "failed"
        except BudgetExceeded as exc:
            status, error = "budget_exceeded", str(exc)
        except PMAError as exc:
            status, error = "error", f"{exc.__class__.__name__}: {exc}"

        report_obj = build_report(ctx, status, error, evaluations, [])
        usage = ctx.budget.summary()
        latency_ms = round((time.perf_counter() - ctx.started) * 1000, 2)
        result = AgentResult(
            run_id=ctx.trace.run_id,
            status=status,  # type: ignore[arg-type]
            product_id=product.id,
            languages=langs,
            draft=draft,
            iterations=len(evaluations),
            evaluations=evaluations,
            report=report_obj,
            security=ctx.security,
            tool_calls=[e.to_record() for e in ctx.registry.executions],
            usage=usage,
            provider=self.provider.name,
            model=self.provider.model,
            latency_ms=latency_ms,
            error=error,
        )
        ctx.trace.result = {"status": status, "usage": usage.model_dump(), "latency_ms": latency_ms}
        if save_trace:
            result.trace_path = str(ctx.trace.save(self.settings.trace_dir))
        return result
