"""Generic OpenAI-compatible chat-completions provider.

Works with any server that implements ``POST {base_url}/chat/completions``: hosted free tiers,
vLLM, LM Studio, or a local Ollama server (``OPENAI_BASE_URL=http://localhost:11434/v1``).

STATUS: exercised only against a local fake server and ``httpx.MockTransport`` in this repository.
NOT VERIFIED against any real hosted API or a real Ollama instance.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from pma.errors import (
    ProviderError,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
)
from pma.providers.base import (
    ChatMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ToolCall,
    Usage,
    estimate_request_tokens,
    estimate_tokens,
)


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _to_wire(message: ChatMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return wire


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        supports_tools: bool = False,
        json_mode: bool = False,
        timeout_s: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.supports_tools = supports_tools
        self._json_mode = json_mode
        self._timeout_s = timeout_s
        self._api_key = api_key
        self._client = client or httpx.Client()

    # -- request building -------------------------------------------------------------------
    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_wire(m) for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.tools and self.supports_tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
        if request.json_mode and self._json_mode and not request.tools:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # -- call -------------------------------------------------------------------------------
    def complete(self, request: LLMRequest) -> LLMResponse:
        timeout = request.timeout_s or self._timeout_s
        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self.build_payload(request),
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"request timed out after {timeout}s") from exc
        except httpx.TransportError as exc:
            raise ProviderServerError(f"transport error: {exc.__class__.__name__}") from exc

        if response.status_code == 429:
            raise ProviderRateLimited("rate limited (429)", retry_after=_retry_after(response))
        if response.status_code >= 500:
            raise ProviderServerError(
                f"server error {response.status_code}", retry_after=_retry_after(response)
            )
        if response.status_code >= 400:
            # Never echo the response body into logs verbatim: it may contain request content.
            raise ProviderError(f"request rejected with status {response.status_code}")
        return self._parse(response, request)

    def _parse(self, response: httpx.Response, request: LLMRequest) -> LLMResponse:
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("malformed response from provider") from exc

        content = message.get("content") or ""
        tool_calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            try:
                fn = raw["function"]
                args = fn.get("arguments") or "{}"
                parsed = json.loads(args) if isinstance(args, str) else dict(args)
                tool_calls.append(
                    ToolCall(raw.get("id") or f"call_{len(tool_calls)}", fn["name"], parsed)
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise ProviderError("malformed tool call in provider response") from exc

        usage_raw = data.get("usage") or {}
        if "prompt_tokens" in usage_raw:
            usage = Usage(
                int(usage_raw.get("prompt_tokens", 0)), int(usage_raw.get("completion_tokens", 0))
            )
        else:
            usage = Usage(
                estimate_request_tokens(request), estimate_tokens(content), estimated=True
            )
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=str(data.get("model") or self.model),
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )

    def close(self) -> None:
        self._client.close()
