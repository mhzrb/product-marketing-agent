"""Deterministic, scriptable provider for tests and the default demo.

Three ways to drive it (checked in this order):

1. ``script=[...]``  - a queue of canned replies, consumed one per call. Items may be a
   ``str``, a ``dict``/``list`` (serialised to JSON), an ``LLMResponse``, an ``Exception``
   (raised), or a callable ``(LLMRequest) -> any of the above``. Exhausting the queue raises
   ``MockScriptExhausted`` so a test never silently falls through to default behaviour.
2. ``handler=callable`` - full control: ``(LLMRequest) -> LLMResponse``.
3. default - ``SimulatedMarketingLLM`` (rule-based copywriter with scriptable flaws).

Every request is kept in ``provider.requests`` for assertions.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from pma.providers.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    Usage,
    estimate_request_tokens,
    estimate_tokens,
)
from pma.providers.simulated import SimulatedMarketingLLM


class MockScriptExhausted(AssertionError):
    pass


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(
        self,
        script: list[Any] | None = None,
        *,
        handler: Callable[[LLMRequest], LLMResponse] | None = None,
        flaw_scripts: dict[str, list[str]] | None = None,
        behavior: str = "clean",
        supports_tools: bool = True,
        latency_s: float = 0.0,
        model: str = "mock-1",
        record_requests: bool = True,
    ) -> None:
        self.model = model
        self.supports_tools = supports_tools
        self.requests: list[LLMRequest] = []
        self._script = deque(script) if script is not None else None
        self._handler = handler
        self._latency_s = latency_s
        self._record = record_requests
        self.simulated = SimulatedMarketingLLM(flaw_scripts, behavior)

    @property
    def calls(self) -> int:
        return len(self.requests)

    def purposes(self) -> list[str]:
        return [r.purpose for r in self.requests]

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self._record:
            self.requests.append(request)
        if self._latency_s:
            time.sleep(self._latency_s)
        if self._script is not None:
            if not self._script:
                raise MockScriptExhausted(
                    f"mock script exhausted at call #{len(self.requests)} "
                    f"(purpose={request.purpose!r})"
                )
            return self._materialise(self._script.popleft(), request)
        if self._handler is not None:
            return self._handler(request)
        return self.simulated(request)

    def _materialise(self, item: Any, request: LLMRequest) -> LLMResponse:
        if callable(item) and not isinstance(item, LLMResponse):
            item = item(request)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, LLMResponse):
            return item
        content = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return LLMResponse(
            content=content,
            usage=Usage(estimate_request_tokens(request), estimate_tokens(content), estimated=True),
            model=self.model,
        )
