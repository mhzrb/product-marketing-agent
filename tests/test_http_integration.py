"""OpenAICompatibleProvider over real HTTP (localhost fake server) driving the full agent.

This verifies our client code (wire format, status handling, retries, timeouts, both tool
transports). It does NOT verify any real vendor API: the fake server is ours.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pma.agent import MarketingAgent
from pma.config import Settings
from pma.providers.openai_compat import OpenAICompatibleProvider

from .conftest import PRODUCT_ID
from .fake_openai import FakeServer


@pytest.fixture(scope="module")
def server():
    with FakeServer() as s:
        yield s


@pytest.fixture
def fake(server):
    server.reset()
    return server


def agent_for(server, retriever, tmp_path, *, supports_tools: bool, **overrides) -> MarketingAgent:
    settings = replace(
        Settings(), trace_dir=tmp_path, llm_backoff_base_s=0.0, llm_backoff_max_s=0.0, **overrides
    )
    provider = OpenAICompatibleProvider(
        base_url=server.base_url,
        model="fake-1",
        api_key="sk-test",
        supports_tools=supports_tools,
        timeout_s=settings.llm_timeout_s,
    )
    return MarketingAgent(provider, settings, retriever=retriever, sleep=lambda _s: None)


@pytest.mark.parametrize("native", [False, True], ids=["json_protocol", "native_tools"])
def test_full_run_over_http(fake, retriever, tmp_path, product, native):
    fake.state.flaw_scripts[PRODUCT_ID] = ["unsupported_number", "none"]
    result = agent_for(fake, retriever, tmp_path, supports_tools=native).run(product)
    assert result.passed and result.iterations == 2
    assert result.provider == "openai_compatible" and result.model == "fake-1"
    sent_tools = [bool(r.get("tools")) for r in fake.state.requests]
    assert any(sent_tools) is native  # native mode advertises tools in the plan step only
    assert all(r["model"] == "fake-1" for r in fake.state.requests)
    assert not result.usage.tokens_estimated  # the server reported usage
    trace = json.loads(Path(result.trace_path).read_text())
    assert trace["meta"]["tool_transport"] == ("native" if native else "json_protocol")


def test_rate_limit_and_server_errors_are_retried(fake, retriever, tmp_path, product):
    fake.state.fail_next = [429, 503]
    result = agent_for(fake, retriever, tmp_path, supports_tools=False).run(product)
    assert result.passed
    trace = json.loads(Path(result.trace_path).read_text())
    retries = [r for s in trace["steps"] for r in s.get("retries", [])]
    assert len(retries) == 2 and "429" in retries[0]["error"] and "503" in retries[1]["error"]


def test_persistent_server_errors_end_in_error_status(fake, retriever, tmp_path, product):
    fake.state.fail_next = [500] * 20
    result = agent_for(fake, retriever, tmp_path, supports_tools=False, llm_max_retries=2).run(
        product
    )
    assert result.status == "error" and "ProviderServerError" in result.error
    assert len(fake.state.requests) == 3


def test_auth_style_errors_are_not_retried(fake, retriever, tmp_path, product):
    fake.state.fail_next = [401]
    result = agent_for(fake, retriever, tmp_path, supports_tools=False).run(product)
    assert result.status == "error" and "ProviderError" in result.error
    assert len(fake.state.requests) == 1


def test_timeouts_are_enforced_and_reported(fake, retriever, tmp_path, product):
    fake.state.delay_next_s = 1.0
    result = agent_for(
        fake, retriever, tmp_path, supports_tools=False, llm_timeout_s=0.3, llm_max_retries=0
    ).run(product)
    assert result.status == "error" and "ProviderTimeout" in result.error
