from __future__ import annotations

import json

import httpx
import pytest

from pma.config import Settings
from pma.errors import (
    ConfigError,
    EmptyResponse,
    InvalidToolCall,
    ProviderError,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
)
from pma.providers import build_provider
from pma.providers.anthropic import AnthropicProvider
from pma.providers.base import ChatMessage, LLMRequest, LLMResponse, ToolCall, ToolSpec
from pma.providers.mock import MockProvider, MockScriptExhausted
from pma.providers.openai_compat import OpenAICompatibleProvider
from pma.reliability import retry_call

REQ = LLMRequest(
    messages=[ChatMessage("system", "be brief"), ChatMessage("user", "hi")], purpose="draft"
)
TOOL = ToolSpec(
    "lookup", "look something up", {"type": "object", "properties": {"k": {"type": "string"}}}
)


# ------------------------------------------------------------------ MockProvider


def test_mock_script_queue_serves_items_in_order_and_records_requests():
    p = MockProvider(script=["plain text", {"a": 1}, LLMResponse(content="raw")])
    assert p.complete(REQ).content == "plain text"
    assert json.loads(p.complete(REQ).content) == {"a": 1}
    assert p.complete(REQ).content == "raw"
    assert p.calls == 3 and p.purposes() == ["draft"] * 3


def test_mock_script_can_raise_and_call_callables():
    p = MockProvider(script=[ProviderTimeout("t"), lambda req: f"echo:{req.messages[-1].content}"])
    with pytest.raises(ProviderTimeout):
        p.complete(REQ)
    assert p.complete(REQ).content == "echo:hi"


def test_mock_script_exhaustion_is_loud():
    p = MockProvider(script=["only one"])
    p.complete(REQ)
    with pytest.raises(MockScriptExhausted):
        p.complete(REQ)


def test_mock_handler_mode_and_usage_estimates():
    p = MockProvider(handler=lambda req: LLMResponse(content="h"))
    assert p.complete(REQ).content == "h"
    scripted = MockProvider(script=["x" * 40]).complete(REQ)
    assert scripted.usage.estimated and scripted.usage.completion_tokens == 10


def test_mock_can_skip_recording():
    p = MockProvider(script=["a"], record_requests=False)
    p.complete(REQ)
    assert p.requests == []


# ------------------------------------------------------------------ OpenAI-compatible


def oa(handler, **kw) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://llm.local/v1",
        model="m1",
        api_key="sk-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kw,
    )


def ok_body(**message):
    return {
        "model": "m1-actual",
        "choices": [{"message": {"content": "hello", **message}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }


def test_openai_request_shape_and_auth_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    resp = oa(handler).complete(REQ)
    assert seen["url"] == "http://llm.local/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-secret"
    assert seen["body"]["model"] == "m1" and seen["body"]["messages"][0] == {
        "role": "system",
        "content": "be brief",
    }
    assert "tools" not in seen["body"] and "response_format" not in seen["body"]
    assert resp.content == "hello" and resp.model == "m1-actual"
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.estimated) == (
        11,
        3,
        False,
    )


def test_openai_no_auth_header_without_key_and_estimated_usage_without_usage_field():
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "abcd"}}]})

    p = OpenAICompatibleProvider(
        base_url="http://x/v1",
        model="m",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    resp = p.complete(REQ)
    assert "authorization" not in seen["headers"] and resp.usage.estimated


def test_openai_tools_and_json_mode_only_when_enabled():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    json_req = LLMRequest(messages=REQ.messages, json_mode=True)
    oa(handler, supports_tools=False, json_mode=False).complete(
        LLMRequest(messages=REQ.messages, tools=[TOOL])
    )
    oa(handler, supports_tools=True, json_mode=True).complete(
        LLMRequest(messages=REQ.messages, tools=[TOOL])
    )
    oa(handler, supports_tools=False, json_mode=True).complete(json_req)
    assert "tools" not in bodies[0]
    assert (
        bodies[1]["tools"][0]["function"]["name"] == "lookup" and "response_format" not in bodies[1]
    )
    assert bodies[2]["response_format"] == {"type": "json_object"}


def test_openai_parses_tool_calls_and_serialises_tool_history():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=ok_body(
                content=None,
                tool_calls=[
                    {"id": "c1", "function": {"name": "lookup", "arguments": '{"k": "price"}'}}
                ],
            ),
        )

    p = oa(handler, supports_tools=True)
    resp = p.complete(LLMRequest(messages=REQ.messages, tools=[TOOL]))
    assert resp.tool_calls == [ToolCall("c1", "lookup", {"k": "price"})]
    history = [
        ChatMessage("user", "q"),
        ChatMessage("assistant", "", tool_calls=resp.tool_calls),
        ChatMessage("tool", '{"v": 1}', tool_call_id="c1", name="lookup"),
    ]
    p.complete(LLMRequest(messages=history))
    wire = bodies[1]["messages"]
    assert wire[1]["tool_calls"][0]["function"]["arguments"] == '{"k": "price"}'
    assert wire[2] == {"role": "tool", "tool_call_id": "c1", "content": '{"v": 1}'}


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, ProviderRateLimited),
        (500, ProviderServerError),
        (503, ProviderServerError),
        (401, ProviderError),
        (400, ProviderError),
    ],
)
def test_openai_status_codes_map_to_retryable_or_permanent_errors(status, error):
    p = oa(lambda r: httpx.Response(status, json={"error": "x"}, headers={"retry-after": "3"}))
    with pytest.raises(error) as err:
        p.complete(REQ)
    assert type(err.value) is error or isinstance(err.value, error)
    if status in (429, 500, 503):
        assert err.value.retry_after == 3.0


TOOL_USE_FAILED = {
    "error": {
        "message": "Tool call validation failed: attempted to call tool 'json'",
        "type": "invalid_request_error",
        "code": "tool_use_failed",
        "failed_generation": "SECRET-MODEL-OUTPUT",
    }
}


def test_openai_tool_use_failed_is_a_retryable_error_without_the_model_output():
    p = oa(lambda r: httpx.Response(400, json=TOOL_USE_FAILED))
    with pytest.raises(InvalidToolCall) as err:
        p.complete(REQ)
    assert isinstance(err.value, ProviderError)
    assert "SECRET-MODEL-OUTPUT" not in str(err.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, json={"error": {"code": "context_length_exceeded"}}),
        httpx.Response(400, json={"error": "tool_use_failed"}),
        httpx.Response(400, json=["tool_use_failed"]),
        httpx.Response(400, text="tool_use_failed but not json"),
    ],
)
def test_other_400_errors_are_not_treated_as_invalid_tool_calls(response):
    p = oa(lambda r: response)
    with pytest.raises(ProviderError) as err:
        p.complete(REQ)
    assert not isinstance(err.value, InvalidToolCall)


def test_openai_tool_use_failed_is_retried_and_the_next_answer_is_used():
    answers = [
        httpx.Response(400, json=TOOL_USE_FAILED),
        httpx.Response(
            200, json={"choices": [{"message": {"content": "fine"}, "finish_reason": "stop"}]}
        ),
    ]
    p = oa(lambda r: answers.pop(0))
    response = retry_call(lambda: p.complete(REQ), retries=2, base=0, cap=0, sleep=lambda _: None)
    assert response.content == "fine" and answers == []


def test_openai_error_messages_never_contain_the_key_or_response_body():
    p = oa(lambda r: httpx.Response(401, text="your key sk-secret is invalid"))
    with pytest.raises(ProviderError) as err:
        p.complete(REQ)
    assert "sk-secret" not in str(err.value)


def test_openai_timeout_and_transport_errors():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderTimeout):
        oa(timeout).complete(REQ)
    with pytest.raises(ProviderServerError):
        oa(refused).complete(REQ)


@pytest.mark.parametrize("content", ["", "   ", None])
def test_openai_empty_completion_is_a_retryable_error_not_an_answer(content):
    body = {
        "choices": [{"message": {"content": content, "reasoning": "..."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 105},
    }
    with pytest.raises(EmptyResponse) as err:
        oa(lambda r: httpx.Response(200, json=body)).complete(REQ)
    assert err.value.retry_after is None
    assert "finish_reason='stop'" in str(err.value) and "reasoning" in str(err.value)


def test_openai_empty_content_is_fine_when_the_model_called_a_tool():
    body = ok_body(
        content="",
        tool_calls=[{"id": "c1", "function": {"name": "lookup", "arguments": '{"k": "price"}'}}],
    )
    response = oa(lambda r: httpx.Response(200, json=body)).complete(REQ)
    assert response.tool_calls[0].name == "lookup"


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {"nope": 1},
        {
            "choices": [
                {"message": {"tool_calls": [{"function": {"name": "x", "arguments": "{bad"}}]}}
            ]
        },
    ],
)
def test_openai_malformed_responses_raise_provider_error(body):
    with pytest.raises(ProviderError):
        oa(lambda r: httpx.Response(200, json=body)).complete(REQ)


# ------------------------------------------------------------------ Anthropic


def an(handler) -> AnthropicProvider:
    return AnthropicProvider(
        api_key="ak-secret",
        model="claude-test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_anthropic_request_shape_splits_system_and_sends_tools():
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "model": "claude-test",
                "content": [
                    {"type": "text", "text": "thinking"},
                    {"type": "tool_use", "id": "tu1", "name": "lookup", "input": {"k": "price"}},
                ],
                "usage": {"input_tokens": 20, "output_tokens": 5},
                "stop_reason": "tool_use",
            },
        )

    resp = an(handler).complete(LLMRequest(messages=REQ.messages, tools=[TOOL]))
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert (
        seen["headers"]["x-api-key"] == "ak-secret"
        and seen["headers"]["anthropic-version"] == "2023-06-01"
    )
    assert seen["body"]["system"] == "be brief" and seen["body"]["messages"] == [
        {"role": "user", "content": "hi"}
    ]
    assert seen["body"]["tools"][0]["input_schema"]["type"] == "object"
    assert resp.content == "thinking" and resp.tool_calls == [
        ToolCall("tu1", "lookup", {"k": "price"})
    ]
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (20, 5)
    assert resp.finish_reason == "tool_use"


def test_anthropic_groups_consecutive_tool_results_into_one_user_message():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "usage": {}})

    calls = [ToolCall("a", "lookup", {}), ToolCall("b", "lookup", {})]
    msgs = [
        ChatMessage("user", "q"),
        ChatMessage("assistant", "", tool_calls=calls),
        ChatMessage("tool", "r1", tool_call_id="a"),
        ChatMessage("tool", "r2", tool_call_id="b"),
    ]
    an(handler).complete(LLMRequest(messages=msgs))
    wire = bodies[0]["messages"]
    assert wire[1]["content"][0]["type"] == "tool_use"
    assert [b["tool_use_id"] for b in wire[2]["content"]] == ["a", "b"] and len(wire) == 3


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, ProviderRateLimited),
        (529, ProviderRateLimited),
        (500, ProviderServerError),
        (403, ProviderError),
    ],
)
def test_anthropic_status_mapping(status, error):
    with pytest.raises(error):
        an(lambda r: httpx.Response(status, json={})).complete(REQ)


def test_anthropic_requires_a_key():
    with pytest.raises(ProviderError):
        AnthropicProvider(api_key="", model="m")


# ------------------------------------------------------------------ factory / config


def test_build_provider_from_settings():
    assert build_provider(Settings.from_env({})).name == "mock"
    oa_settings = Settings.from_env(
        {
            "LLM_PROVIDER": "openai_compatible",
            "OPENAI_BASE_URL": "http://h:1/v1/",
            "OPENAI_MODEL": "q",
            "LLM_SUPPORTS_TOOLS": "true",
        }
    )
    p = build_provider(oa_settings)
    assert (
        isinstance(p, OpenAICompatibleProvider)
        and p.base_url == "http://h:1/v1"
        and p.supports_tools
    )
    a = build_provider(Settings.from_env({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"}))
    assert isinstance(a, AnthropicProvider) and a.supports_tools


@pytest.mark.parametrize(
    "env",
    [
        {"LLM_PROVIDER": "anthropic"},
        {"LLM_PROVIDER": "skynet"},
        {"AGENT_MAX_ITERATIONS": "0"},
        {"AGENT_MAX_ITERATIONS": "abc"},
        {"LLM_TIMEOUT_S": "-1"},
        {"JUDGE_ENABLED": "maybe"},
        {"RETRIEVER": "magic"},
    ],
)
def test_invalid_configuration_fails_fast(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_settings_do_not_expose_secrets_in_repr():
    s = Settings.from_env({"OPENAI_API_KEY": "sk-top-secret", "ANTHROPIC_API_KEY": "ak-top-secret"})
    assert "top-secret" not in repr(s)
    assert s.openai_api_key == "sk-top-secret"
