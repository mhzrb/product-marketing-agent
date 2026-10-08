"""Provider-neutral request/response types and the provider interface."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant -> tools
    tool_call_id: str | None = None  # tool result -> which call
    name: str | None = None  # tool name for tool results


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False  # True when the provider did not report real token counts


@dataclass
class LLMRequest:
    messages: list[ChatMessage]
    tools: list[ToolSpec] = field(default_factory=list)
    json_mode: bool = False
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout_s: float | None = None
    # Free-form label ("plan", "draft", "revise", "judge", "repair", "baseline"). Real providers
    # ignore it; the mock uses it for dispatch and traces record it.
    purpose: str = ""


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = "stop"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Only used when a provider reports no usage."""
    return max(1, math.ceil(len(text) / 4)) if text else 0


def estimate_request_tokens(request: LLMRequest) -> int:
    return sum(estimate_tokens(m.content) for m in request.messages)


class LLMProvider(ABC):
    """Minimal synchronous chat-completion interface."""

    name: str = "provider"
    model: str = ""
    supports_tools: bool = False

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return one completion. Raise ``RetryableProviderError`` subclasses for transient
        failures and ``ProviderError`` for permanent ones."""

    def close(self) -> None:  # pragma: no cover - default no-op
        return None
