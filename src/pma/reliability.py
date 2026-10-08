"""Retries with backoff, JSON extraction/validation with one repair attempt, run budgets."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from pma.errors import BudgetExceeded, OutputValidationError, RetryableProviderError
from pma.providers.base import Usage
from pma.schemas import UsageSummary

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


# --------------------------------------------------------------------------- retries


def backoff_delay(
    attempt: int,
    *,
    base: float,
    cap: float,
    rng: random.Random | None = None,
    retry_after: float | None = None,
) -> float:
    """Exponential backoff with "equal jitter": half fixed, half random, capped.

    ``attempt`` is 0 for the first retry. A server-provided ``retry_after`` is honoured as a
    lower bound (still capped).
    """
    rng = rng or random.Random()
    ceiling = min(cap, base * (2**attempt))
    delay = ceiling / 2 + rng.uniform(0, ceiling / 2)
    if retry_after is not None:
        delay = max(delay, retry_after)
    return min(delay, cap)


def retry_call(
    fn: Callable[[], T],
    *,
    retries: int,
    base: float,
    cap: float,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> T:
    """Call ``fn`` and retry on ``RetryableProviderError`` up to ``retries`` extra times."""
    attempt = 0
    while True:
        try:
            return fn()
        except RetryableProviderError as exc:
            if attempt >= retries:
                raise
            delay = backoff_delay(attempt, base=base, cap=cap, rng=rng, retry_after=exc.retry_after)
            if on_retry:
                on_retry(attempt + 1, delay, exc)
            sleep(delay)
            attempt += 1


# --------------------------------------------------------------------------- JSON handling


def extract_json(text: str) -> object:
    """Parse JSON from model output, tolerating Markdown fences and surrounding prose.

    Deliberately strict about JSON itself (no trailing commas, no comments): malformed JSON is
    handled by the one-shot repair step so that the failure is visible in the trace.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    candidate = _first_balanced_object(stripped)
    if candidate is None:
        raise ValueError("no JSON object found in model output")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})") from exc


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def summarize_validation_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors()[:6]:
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            parts.append(f"{loc}: {err['msg']}")
        return "; ".join(parts)
    return str(exc)


def parse_model(text: str, model: type[M]) -> M:
    try:
        data = extract_json(text)
        return model.model_validate(data)
    except (ValueError, ValidationError) as exc:
        raise OutputValidationError(summarize_validation_error(exc), raw=text) from exc


def parse_with_repair(
    text: str,
    model: type[M],
    repair: Callable[[str, str], str],
) -> tuple[M, bool]:
    """Validate ``text`` against ``model``. On failure call ``repair(error, raw)`` exactly once.

    Returns ``(parsed, repaired)``. Raises ``OutputValidationError`` if the repair also fails.
    """
    try:
        return parse_model(text, model), False
    except OutputValidationError as first:
        repaired_text = repair(str(first), text)
        try:
            return parse_model(repaired_text, model), True
        except OutputValidationError as second:
            raise OutputValidationError(
                f"invalid after one repair attempt: {second}", raw=repaired_text
            ) from second


# --------------------------------------------------------------------------- budget


@dataclass
class Budget:
    """Per-run limits. ``check`` is called before every LLM call, ``record`` after."""

    max_tokens: int
    max_cost_usd: float
    max_calls: int
    price_per_1k_prompt: float = 0.0
    price_per_1k_completion: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    estimated: bool = False
    _cost: float = field(default=0.0, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return self._cost

    def check(self, upcoming_prompt_tokens: int = 0) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"LLM call limit reached ({self.max_calls})")
        if self.total_tokens + upcoming_prompt_tokens > self.max_tokens:
            raise BudgetExceeded(
                f"token budget exceeded ({self.total_tokens}+{upcoming_prompt_tokens} "
                f"> {self.max_tokens})"
            )
        if self.max_cost_usd > 0 and self._cost >= self.max_cost_usd:
            raise BudgetExceeded(
                f"cost budget exceeded (${self._cost:.4f} >= ${self.max_cost_usd})"
            )

    def record(self, usage: Usage) -> None:
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.estimated = self.estimated or usage.estimated
        self._cost += (
            usage.prompt_tokens / 1000 * self.price_per_1k_prompt
            + usage.completion_tokens / 1000 * self.price_per_1k_completion
        )

    def summary(self) -> UsageSummary:
        return UsageSummary(
            llm_calls=self.calls,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            estimated_cost_usd=round(self._cost, 6),
            tokens_estimated=self.estimated,
        )
