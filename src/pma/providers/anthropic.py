"""Optional Anthropic Messages API provider (plain ``httpx``, no SDK dependency).

STATUS: exercised only against ``httpx.MockTransport`` in this repository.
NOT VERIFIED against the real Anthropic API (no network access or API key in the build workspace).
"""

from __future__ import annotations

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
)

ANTHROPIC_VERSION = "2023-06-01"


def _to_wire(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
    """Split the system prompt out and convert tool traffic to Anthropic content blocks."""
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        elif m.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id or "",
                "content": m.content,
            }
            # consecutive tool results must live in ONE user message
            if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                wire[-1]["content"].append(block)
            else:
                wire.append({"role": "user", "content": [block]})
        elif m.role == "assistant" and m.tool_calls:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks += [
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in m.tool_calls
            ]
            wire.append({"role": "assistant", "content": blocks})
        else:
            wire.append({"role": m.role, "content": m.content})
    return "\n\n".join(system_parts), wire


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    supports_tools = True

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com",
        timeout_s: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is required for AnthropicProvider")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._client = client or httpx.Client()

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        system, wire = _to_wire(request.messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": wire,
        }
        if system:
            payload["system"] = system
        if request.tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in request.tools
            ]
        return payload

    def complete(self, request: LLMRequest) -> LLMResponse:
        timeout = request.timeout_s or self._timeout_s
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        try:
            response = self._client.post(
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=self.build_payload(request),
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"request timed out after {timeout}s") from exc
        except httpx.TransportError as exc:
            raise ProviderServerError(f"transport error: {exc.__class__.__name__}") from exc

        retry_after: float | None = None
        try:
            retry_after = float(response.headers.get("retry-after", ""))
        except ValueError:
            retry_after = None
        if response.status_code in (429, 529):
            raise ProviderRateLimited(
                f"rate limited or overloaded ({response.status_code})", retry_after=retry_after
            )
        if response.status_code >= 500:
            raise ProviderServerError(
                f"server error {response.status_code}", retry_after=retry_after
            )
        if response.status_code >= 400:
            raise ProviderError(f"request rejected with status {response.status_code}")

        try:
            data = response.json()
            blocks = data["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError("malformed response from provider") from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in blocks:
            if block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(str(block["id"]), str(block["name"]), dict(block.get("input") or {}))
                )
        usage_raw = data.get("usage") or {}
        return LLMResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            usage=Usage(
                int(usage_raw.get("input_tokens", 0)), int(usage_raw.get("output_tokens", 0))
            ),
            model=str(data.get("model") or self.model),
            finish_reason=str(data.get("stop_reason") or "stop"),
        )

    def close(self) -> None:
        self._client.close()
