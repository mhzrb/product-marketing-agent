"""Runtime configuration, read from environment variables (see .env.example).

Implemented with the standard library only so the dependency list stays small. Every value has a
safe default; invalid values raise ``ConfigError`` early instead of failing deep inside a run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from pma.errors import ConfigError

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(f"{key} must be a boolean (true/false), got {raw!r}")


def _num(env: Mapping[str, str], key: str, default: float, *, lo: float, hi: float) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{key} must be between {lo} and {hi}, got {value}")
    return value


def _choice(env: Mapping[str, str], key: str, default: str, choices: tuple[str, ...]) -> str:
    value = env.get(key, default).strip().lower() or default
    if value not in choices:
        raise ConfigError(f"{key} must be one of {', '.join(choices)}; got {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    # provider selection
    llm_provider: str = "mock"
    # OpenAI-compatible endpoint (hosted free tier, or Ollama at http://localhost:11434/v1)
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = field(default="", repr=False)
    openai_model: str = "llama3.1"
    llm_supports_tools: bool = False
    openai_json_mode: bool = False
    # Anthropic (optional)
    anthropic_api_key: str = field(default="", repr=False)
    anthropic_model: str = "claude-haiku-4-5-20251001"
    anthropic_base_url: str = "https://api.anthropic.com"
    # reliability
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    llm_backoff_base_s: float = 1.0
    llm_backoff_max_s: float = 20.0
    # agent limits
    agent_max_iterations: int = 3
    max_total_tokens: int = 60_000
    max_cost_usd: float = 0.25
    max_llm_calls: int = 30
    price_per_1k_prompt_usd: float = 0.0
    price_per_1k_completion_usd: float = 0.0
    judge_enabled: bool = True
    sanitize_input: bool = True
    # retrieval
    retriever: str = "bm25"  # bm25 | embeddings | hybrid
    embeddings_backend: str = "hashing"  # hashing | openai_compatible
    embedding_model: str = "nomic-embed-text"
    # observability
    prompt_version: str = "v1"
    trace_dir: Path = Path("traces")
    trace_full_io: bool = False
    log_level: str = "INFO"
    log_format: str = "text"  # text | json
    # mock provider behaviour
    mock_behavior: str = "demo"  # demo | clean

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        settings = cls(
            llm_provider=_choice(
                e, "LLM_PROVIDER", "mock", ("mock", "openai_compatible", "anthropic")
            ),
            openai_base_url=e.get("OPENAI_BASE_URL", cls.openai_base_url).rstrip("/"),
            openai_api_key=e.get("OPENAI_API_KEY", ""),
            openai_model=e.get("OPENAI_MODEL", cls.openai_model),
            llm_supports_tools=_bool(e, "LLM_SUPPORTS_TOOLS", False),
            openai_json_mode=_bool(e, "OPENAI_JSON_MODE", False),
            anthropic_api_key=e.get("ANTHROPIC_API_KEY", ""),
            anthropic_model=e.get("ANTHROPIC_MODEL", cls.anthropic_model),
            anthropic_base_url=e.get("ANTHROPIC_BASE_URL", cls.anthropic_base_url).rstrip("/"),
            llm_timeout_s=_num(e, "LLM_TIMEOUT_S", 60.0, lo=0.1, hi=600),
            llm_max_retries=int(_num(e, "LLM_MAX_RETRIES", 3, lo=0, hi=10)),
            llm_backoff_base_s=_num(e, "LLM_BACKOFF_BASE_S", 1.0, lo=0, hi=60),
            llm_backoff_max_s=_num(e, "LLM_BACKOFF_MAX_S", 20.0, lo=0, hi=300),
            agent_max_iterations=int(_num(e, "AGENT_MAX_ITERATIONS", 3, lo=1, hi=10)),
            max_total_tokens=int(_num(e, "MAX_TOTAL_TOKENS", 60_000, lo=100, hi=10_000_000)),
            max_cost_usd=_num(e, "MAX_COST_USD", 0.25, lo=0, hi=1000),
            max_llm_calls=int(_num(e, "MAX_LLM_CALLS", 30, lo=1, hi=1000)),
            price_per_1k_prompt_usd=_num(e, "PRICE_PER_1K_PROMPT_USD", 0.0, lo=0, hi=1000),
            price_per_1k_completion_usd=_num(e, "PRICE_PER_1K_COMPLETION_USD", 0.0, lo=0, hi=1000),
            judge_enabled=_bool(e, "JUDGE_ENABLED", True),
            sanitize_input=_bool(e, "SANITIZE_INPUT", True),
            retriever=_choice(e, "RETRIEVER", "bm25", ("bm25", "embeddings", "hybrid")),
            embeddings_backend=_choice(
                e, "EMBEDDINGS_BACKEND", "hashing", ("hashing", "openai_compatible")
            ),
            embedding_model=e.get("EMBEDDING_MODEL", cls.embedding_model),
            prompt_version=e.get("PROMPT_VERSION", "v1"),
            trace_dir=Path(e.get("TRACE_DIR", "traces")),
            trace_full_io=_bool(e, "TRACE_FULL_IO", False),
            log_level=e.get("LOG_LEVEL", "INFO").upper(),
            log_format=_choice(e, "LOG_FORMAT", "text", ("text", "json")),
            mock_behavior=_choice(e, "MOCK_BEHAVIOR", "demo", ("demo", "clean")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Cross-field validation that depends on the chosen provider."""
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ConfigError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        if self.llm_provider == "openai_compatible" and not self.openai_base_url:
            raise ConfigError("LLM_PROVIDER=openai_compatible requires OPENAI_BASE_URL")
