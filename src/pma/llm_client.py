"""The single choke point for LLM calls: retries, budget, timing, tracing, JSON validation.

The agent never talks to a provider directly, so every call is budgeted, retried and traced the
same way regardless of which provider is configured.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from pma.config import Settings
from pma.errors import OutputValidationError
from pma.logging_setup import get_logger
from pma.prompts import Prompt, PromptStore
from pma.providers.base import (
    ChatMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ToolSpec,
    estimate_request_tokens,
)
from pma.reliability import Budget, parse_with_repair, retry_call
from pma.tracing import Trace

log = get_logger("llm")
M = TypeVar("M", bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings,
        budget: Budget,
        trace: Trace,
        prompts: PromptStore,
        *,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.budget = budget
        self.trace = trace
        self.prompts = prompts
        self._sleep = sleep
        self._rng = rng

    # ------------------------------------------------------------------ raw call
    def call(
        self,
        purpose: str,
        messages: list[ChatMessage],
        *,
        prompt: Prompt | None = None,
        tools: list[ToolSpec] | None = None,
        json_mode: bool = False,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> LLMResponse:
        request = LLMRequest(
            messages=messages,
            tools=tools or [],
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=self.settings.llm_timeout_s,
            purpose=purpose,
        )
        self.budget.check(estimate_request_tokens(request))
        retries: list[dict[str, object]] = []

        def on_retry(n: int, delay: float, exc: Exception) -> None:
            retries.append({"retry": n, "delay_s": round(delay, 3), "error": str(exc)})
            log.warning(
                "retrying", extra={"purpose": purpose, "retry": n, "delay_s": round(delay, 3)}
            )

        t0 = time.perf_counter()
        try:
            response = retry_call(
                lambda: self.provider.complete(request),
                retries=self.settings.llm_max_retries,
                base=self.settings.llm_backoff_base_s,
                cap=self.settings.llm_backoff_max_s,
                sleep=self._sleep,
                rng=self._rng,
                on_retry=on_retry,
            )
        except Exception as exc:
            self.trace.add(
                "llm_call",
                purpose=purpose,
                prompt=prompt.ref if prompt else None,
                ok=False,
                error=f"{exc.__class__.__name__}: {exc}",
                retries=retries,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            raise
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self.budget.record(response.usage)
        record: dict[str, object] = {
            "purpose": purpose,
            "prompt": prompt.ref if prompt else None,
            "provider": self.provider.name,
            "model": response.model or self.provider.model,
            "ok": True,
            "latency_ms": latency_ms,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "tokens_estimated": response.usage.estimated,
            "finish_reason": response.finish_reason,
            "tool_calls": [c.name for c in response.tool_calls],
            "retries": retries,
            "output": self.trace.io(response.content),
        }
        if self.trace.full_io:
            record["input"] = [{"role": m.role, "content": m.content} for m in messages]
        self.trace.add("llm_call", **record)
        return response

    # ------------------------------------------------------------------ validated output
    def validate(
        self, text: str, model: type[M], *, purpose: str, max_tokens: int = 2000
    ) -> tuple[M, bool]:
        """Validate ``text`` against ``model``; on failure ask the model to repair it once."""
        repair_prompt = self.prompts.get("repair")

        def repair(error: str, raw: str) -> str:
            rendered = repair_prompt.render(
                error=error,
                previous_output=raw.replace("<", "\\u003c"),
                schema=json.dumps(model.model_json_schema()),
            )
            return self.call(
                "repair",
                [ChatMessage("user", rendered)],
                prompt=repair_prompt,
                json_mode=True,
                max_tokens=max_tokens,
            ).content

        try:
            parsed, repaired = parse_with_repair(text, model, repair)
        except OutputValidationError as exc:
            self.trace.add(
                "validate", purpose=purpose, model=model.__name__, ok=False, error=str(exc)
            )
            raise
        self.trace.add(
            "validate", purpose=purpose, model=model.__name__, repaired=repaired, ok=True
        )
        return parsed, repaired

    def structured(
        self,
        purpose: str,
        messages: list[ChatMessage],
        model: type[M],
        *,
        prompt: Prompt,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> tuple[M, bool]:
        """One LLM call whose output must validate against ``model`` (one repair attempt)."""
        response = self.call(
            purpose,
            messages,
            prompt=prompt,
            json_mode=True,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self.validate(response.content, model, purpose=purpose, max_tokens=max_tokens)
