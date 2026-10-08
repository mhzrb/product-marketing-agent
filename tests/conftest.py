from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pma.agent import MarketingAgent
from pma.config import Settings
from pma.providers.mock import MockProvider
from pma.rag.retriever import GuidelineRetriever, build_retriever
from pma.schemas import ProductInput

PRODUCT_ID = "t-1"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings with a temp trace dir and instant backoff."""
    return replace(
        Settings(),
        trace_dir=tmp_path / "traces",
        llm_backoff_base_s=0.0,
        llm_backoff_max_s=0.0,
    )


@pytest.fixture(scope="session")
def retriever() -> GuidelineRetriever:
    return build_retriever(Settings())


@pytest.fixture
def product() -> ProductInput:
    return ProductInput(
        id=PRODUCT_ID,
        name="Dell Latitude 5440",
        category="Laptop",
        specs={"RAM": "16 GB", "Storage": "512 GB SSD", "Display": "14 inch"},
        price=1299.0,
        audience="IT managers",
        notes="Includes TPM 2.0.",
    )


@pytest.fixture
def make_agent(settings, retriever):
    """Factory: ``make_agent(provider, **setting_overrides) -> MarketingAgent`` (no real sleeping)."""

    def _make(provider: MockProvider, **overrides) -> MarketingAgent:
        return MarketingAgent(
            provider,
            replace(settings, **overrides) if overrides else settings,
            retriever=retriever,
            sleep=lambda _s: None,
        )

    return _make


def scripted(*flaws: str, **kwargs) -> MockProvider:
    """A simulated mock that applies ``flaws`` (one per attempt) to the test product."""
    return MockProvider(flaw_scripts={PRODUCT_ID: list(flaws)}, **kwargs)


def make_provider(
    flaws: tuple[str, ...] | list[str] = (),
    overrides: dict[str, list] | None = None,
    *,
    supports_tools: bool = True,
) -> MockProvider:
    """Simulated mock for the test product, with per-purpose canned replies taking precedence.

    ``overrides={"draft": ["garbage"], "plan": [ProviderTimeout("x")]}`` serves those items (in
    order, one per call) for that purpose, then falls back to the simulated model.
    ``str``/``dict`` become the reply text, ``LLMResponse`` is used as is, exceptions are raised.
    """
    import json
    from collections import deque

    from pma.providers.base import LLMResponse, Usage

    queues = {k: deque(v) for k, v in (overrides or {}).items()}
    provider = MockProvider(
        flaw_scripts={PRODUCT_ID: list(flaws)}, supports_tools=supports_tools, model="mock-test"
    )

    def handler(request):
        provider.requests.append(request)
        queue = queues.get(request.purpose)
        if queue:
            item = queue.popleft()
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, LLMResponse):
                return item
            text = item if isinstance(item, str) else json.dumps(item)
            return LLMResponse(content=text, usage=Usage(10, 10, estimated=True), model="mock-test")
        return provider.simulated(request)

    provider._handler = handler
    provider._record = False  # the handler above records, avoiding double entries
    return provider
