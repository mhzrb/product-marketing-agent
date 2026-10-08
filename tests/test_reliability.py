from __future__ import annotations

import random

import pytest
from pydantic import BaseModel

from pma.errors import (
    BudgetExceeded,
    OutputValidationError,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
)
from pma.providers.base import Usage
from pma.reliability import (
    Budget,
    backoff_delay,
    extract_json,
    parse_model,
    parse_with_repair,
    retry_call,
)


class Item(BaseModel):
    name: str
    qty: int


# ------------------------------------------------------------------ backoff and retries


def test_backoff_grows_exponentially_with_jitter_and_respects_cap():
    rng = random.Random(1)
    delays = [backoff_delay(n, base=1.0, cap=10.0, rng=rng) for n in range(6)]
    for n, d in enumerate(delays):
        ceiling = min(10.0, 2**n)
        assert ceiling / 2 <= d <= ceiling
    assert max(delays) <= 10.0


def test_backoff_honours_retry_after_but_not_beyond_cap():
    assert backoff_delay(0, base=1.0, cap=60.0, retry_after=7.0) == 7.0
    assert backoff_delay(0, base=1.0, cap=5.0, retry_after=30.0) == 5.0


def test_retry_call_retries_transient_errors_then_succeeds():
    attempts, slept = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ProviderTimeout("slow")
        return "ok"

    result = retry_call(flaky, retries=3, base=0.5, cap=4, sleep=slept.append, rng=random.Random(0))
    assert result == "ok" and len(attempts) == 3 and len(slept) == 2


def test_retry_call_gives_up_after_the_limit():
    attempts = []

    def always():
        attempts.append(1)
        raise ProviderRateLimited("429")

    with pytest.raises(ProviderRateLimited):
        retry_call(always, retries=2, base=0, cap=0, sleep=lambda _: None)
    assert len(attempts) == 3  # first try + 2 retries


def test_retry_call_does_not_retry_permanent_errors():
    attempts = []

    def bad_request():
        attempts.append(1)
        raise ProviderError("401")

    with pytest.raises(ProviderError):
        retry_call(bad_request, retries=5, base=0, cap=0, sleep=lambda _: None)
    assert len(attempts) == 1


def test_retry_call_reports_each_retry():
    seen = []
    calls = iter([ProviderTimeout("a"), ProviderTimeout("b"), "done"])

    def fn():
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    retry_call(
        fn, retries=3, base=0, cap=0, sleep=lambda _: None, on_retry=lambda n, d, e: seen.append(n)
    )
    assert seen == [1, 2]


# ------------------------------------------------------------------ JSON extraction/validation


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "a", "qty": 1}',
        '```json\n{"name": "a", "qty": 1}\n```',
        '```\n{"name": "a", "qty": 1}\n```',
        'Sure! Here you go:\n{"name": "a", "qty": 1}\nHope that helps.',
        '  \n {"name": "a", "qty": 1}  ',
    ],
)
def test_extract_json_tolerates_fences_and_prose(text):
    assert extract_json(text) == {"name": "a", "qty": 1}


def test_extract_json_handles_braces_inside_strings():
    assert extract_json('prefix {"name": "a}{b", "qty": 2} suffix')["name"] == "a}{b"


@pytest.mark.parametrize("text", ["no json here", '{"name": "a", "qty": 1,}', '{"name": ', ""])
def test_extract_json_rejects_garbage_and_trailing_commas(text):
    with pytest.raises(ValueError):
        extract_json(text)


def test_parse_model_reports_schema_errors_readably():
    with pytest.raises(OutputValidationError) as err:
        parse_model('{"name": "a", "qty": "many"}', Item)
    assert "qty" in str(err.value)


def test_parse_with_repair_valid_input_never_calls_repair():
    def repair(error, raw):
        raise AssertionError("must not be called")

    item, repaired = parse_with_repair('{"name": "a", "qty": 1}', Item, repair)
    assert item.qty == 1 and repaired is False


def test_parse_with_repair_calls_repair_exactly_once_and_succeeds():
    calls = []

    def repair(error, raw):
        calls.append((error, raw))
        return '{"name": "fixed", "qty": 2}'

    item, repaired = parse_with_repair('{"name": "a", "qty": 1,}', Item, repair)
    assert item.name == "fixed" and repaired is True and len(calls) == 1
    assert "invalid JSON" in calls[0][0]


def test_parse_with_repair_gives_up_after_one_attempt():
    calls = []

    def repair(error, raw):
        calls.append(1)
        return "still broken"

    with pytest.raises(OutputValidationError) as err:
        parse_with_repair("broken", Item, repair)
    assert len(calls) == 1 and "after one repair attempt" in str(err.value)


# ------------------------------------------------------------------ budget


def test_budget_tracks_usage_and_cost():
    b = Budget(
        max_tokens=1000,
        max_cost_usd=1.0,
        max_calls=10,
        price_per_1k_prompt=2.0,
        price_per_1k_completion=4.0,
    )
    b.record(Usage(500, 100, estimated=True))
    s = b.summary()
    assert (s.llm_calls, s.prompt_tokens, s.completion_tokens) == (1, 500, 100)
    assert s.estimated_cost_usd == pytest.approx(1.4)
    assert s.tokens_estimated and s.total_tokens == 600


def test_budget_blocks_before_the_token_limit_is_crossed():
    b = Budget(max_tokens=1000, max_cost_usd=0, max_calls=10)
    b.record(Usage(800, 100))
    b.check(upcoming_prompt_tokens=100)  # exactly at the limit is allowed
    with pytest.raises(BudgetExceeded, match="token budget"):
        b.check(upcoming_prompt_tokens=101)


def test_budget_blocks_on_cost_and_call_count():
    b = Budget(max_tokens=10**9, max_cost_usd=0.01, max_calls=2, price_per_1k_prompt=10.0)
    b.record(Usage(1000, 0))
    with pytest.raises(BudgetExceeded, match="cost"):
        b.check()
    b2 = Budget(max_tokens=10**9, max_cost_usd=0, max_calls=2)
    b2.record(Usage())
    b2.record(Usage())
    with pytest.raises(BudgetExceeded, match="call limit"):
        b2.check()
