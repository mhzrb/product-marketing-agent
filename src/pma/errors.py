"""Exception hierarchy. Retryable and non-retryable failures are separate types on purpose."""

from __future__ import annotations


class PMAError(Exception):
    """Base class for all application errors."""


class ConfigError(PMAError):
    """Invalid or missing configuration."""


class ProviderError(PMAError):
    """The LLM provider failed in a way that retrying will not fix (auth, bad request, ...)."""


class RetryableProviderError(ProviderError):
    """Transient provider failure: timeout, rate limit, 5xx."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderTimeout(RetryableProviderError):
    pass


class ProviderRateLimited(RetryableProviderError):
    pass


class ProviderServerError(RetryableProviderError):
    pass


class EmptyResponse(RetryableProviderError):
    """The model returned no text and no tool call (seen with some reasoning models)."""


class InvalidToolCall(RetryableProviderError):
    """The provider rejected the model's own output as a malformed tool call (HTTP 400,
    code ``tool_use_failed``). Sampling is random, so another attempt can succeed."""


class OutputValidationError(PMAError):
    """The model output could not be parsed/validated, even after one repair attempt."""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


class BudgetExceeded(PMAError):
    """A per-run limit (tokens, cost, LLM calls) would be exceeded."""


class ToolProtocolError(PMAError):
    """The model violated the JSON tool protocol (and the repair attempt did not fix it)."""
