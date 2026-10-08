"""LLM providers and the factory that builds one from ``Settings``."""

from __future__ import annotations

from pma.config import Settings
from pma.providers.anthropic import AnthropicProvider
from pma.providers.base import LLMProvider
from pma.providers.mock import MockProvider
from pma.providers.openai_compat import OpenAICompatibleProvider


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockProvider(
            behavior=settings.mock_behavior, supports_tools=True, record_requests=False
        )
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleProvider(
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            supports_tools=settings.llm_supports_tools,
            json_mode=settings.openai_json_mode,
            timeout_s=settings.llm_timeout_s,
        )
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            base_url=settings.anthropic_base_url,
            timeout_s=settings.llm_timeout_s,
        )
    raise ValueError(f"unknown provider {settings.llm_provider!r}")  # pragma: no cover


__all__ = [
    "AnthropicProvider",
    "LLMProvider",
    "MockProvider",
    "OpenAICompatibleProvider",
    "build_provider",
]
