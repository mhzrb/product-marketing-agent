from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from pma.agent import MarketingAgent
from pma.baseline import BaselineGenerator
from pma.errors import EmptyResponse, ProviderError, ProviderServerError, ProviderTimeout
from pma.providers.base import LLMResponse, ToolCall
from pma.schemas import AgentResult, ProductInput

from .conftest import PRODUCT_ID, make_provider


def run(make_agent, provider, product, **overrides) -> AgentResult:
    return make_agent(provider, **overrides).run(product)


def prompts_sent(provider, purpose: str) -> list[str]:
    return [
        "\n".join(m.content for m in r.messages) for r in provider.requests if r.purpose == purpose
    ]


# ------------------------------------------------------------------ happy path


def test_clean_run_passes_on_the_first_iteration(make_agent, product):
    provider = make_provider()
    result = run(make_agent, provider, product)
    assert result.status == "passed" and result.passed and result.iterations == 1
    assert result.draft is not None and result.draft.languages() == ["en", "nl"]
    assert provider.purposes() == ["plan", "plan", "plan", "plan", "draft", "judge"]
    assert "all 11 deterministic checks passed" in result.report.summary
    assert result.report.checks_failed == [] and len(result.report.checks_passed) == 11
    assert result.report.revisions == [] and result.report.guidelines_used


def test_tools_are_used_by_the_model_and_by_the_evaluator(make_agent, product):
    result = run(make_agent, make_provider(), product)
    by_model = [c.name for c in result.tool_calls if c.invoked_by == "model"]
    by_evaluator = {c.name for c in result.tool_calls if c.invoked_by == "evaluator"}
    assert by_model == ["lookup_product_fact", "retrieve_guidelines", "retrieve_guidelines"]
    assert by_evaluator == {"check_length", "check_banned_claims"}
    assert all(c.ok for c in result.tool_calls)


def test_retrieved_guidelines_reach_the_draft_prompt(make_agent, product):
    provider = make_provider()
    run(make_agent, provider, product)
    draft_prompt = prompts_sent(provider, "draft")[0]
    assert (
        "<guidelines>" in draft_prompt
        and "claims_policy.md" in draft_prompt
        or "brand_voice.md" in draft_prompt
    )
    assert "<plan>" in draft_prompt and "<limits>" in draft_prompt


def test_run_is_deterministic(make_agent, product):
    a = run(make_agent, make_provider(["unsupported_number", "none"]), product)
    b = run(make_agent, make_provider(["unsupported_number", "none"]), product)
    assert a.draft == b.draft and a.report == b.report and a.iterations == b.iterations


def test_single_language_request(make_agent, product):
    result = make_agent(make_provider()).run(product, ["nl"])
    assert result.passed and result.draft.en is None and result.draft.languages() == ["nl"]


def test_product_without_specs_or_price_passes_without_inventing_numbers(make_agent):
    bare = ProductInput(id=PRODUCT_ID, name="Generic USB-C Cable", category="Cable")
    result = make_agent(make_provider()).run(bare)
    assert result.passed
    text = result.draft.en.all_text() + result.draft.nl.all_text()
    assert not re.search(r"\d", text.replace("USB-C", ""))
    assert any("no price" in w for w in result.report.data_warnings)


# ------------------------------------------------------------------ revise loop


@pytest.mark.parametrize(
    ("flaw", "expected_tag"),
    [
        ("unsupported_number", "[numbers_supported]"),
        ("banned_claim", "[banned_claims]"),
        ("too_long", "[length]"),
        ("qualitative_claim", "[llm_judge]"),
        ("missing_language", "[schema_complete]"),
        ("obey_injection", "[injection_echo]"),
    ],
)
def test_each_flaw_is_detected_fed_back_and_fixed(make_agent, product, flaw, expected_tag):
    provider = make_provider([flaw, "none"])
    result = run(make_agent, provider, product)
    assert result.passed and result.iterations == 2
    assert not result.evaluations[0].passed and result.evaluations[1].passed
    assert any(expected_tag in p for p in result.report.revisions[0].problems)
    revise_prompt = prompts_sent(provider, "revise")[0]
    assert expected_tag in revise_prompt and "<previous_draft>" in revise_prompt
    assert "<attempt>2</attempt>" in revise_prompt
    assert "revision(s)" in result.report.summary


def test_flaw_is_only_fixed_if_the_feedback_names_it(make_agent, product):
    """Judge disabled -> the qualitative claim is never reported -> the 'model' never fixes it."""
    provider = make_provider(["qualitative_claim", "none"])
    result = run(make_agent, provider, product, judge_enabled=False)
    assert result.passed and result.iterations == 1
    assert "Trusted by thousands" in result.draft.en.description
    assert "judge was not run" in result.report.summary


def test_loop_stops_at_max_iterations_and_reports_why(make_agent, product):
    provider = make_provider(["banned_claim"] * 3)
    result = run(make_agent, provider, product)
    assert result.status == "failed" and result.iterations == 3 and not result.passed
    assert "Not approved after 3 iteration(s)" in result.report.summary
    assert "banned_claims" in result.report.checks_failed
    assert [r.iteration for r in result.report.revisions] == [1, 2, 3]
    assert provider.purposes().count("revise") == 2


def test_max_iterations_is_configurable(make_agent, product):
    result = run(make_agent, make_provider(["banned_claim"] * 3), product, agent_max_iterations=2)
    assert result.iterations == 2 and result.status == "failed"
    one = run(make_agent, make_provider(["banned_claim", "none"]), product, agent_max_iterations=1)
    assert one.iterations == 1 and one.status == "failed"


def test_slow_fix_passes_on_the_last_allowed_iteration(make_agent, product):
    result = run(
        make_agent, make_provider(["unsupported_number", "unsupported_number", "none"]), product
    )
    assert result.passed and result.iterations == 3


# ------------------------------------------------------------------ JSON validation + repair


def test_invalid_json_is_repaired_once(make_agent, product):
    provider = make_provider(["invalid_json"])
    result = run(make_agent, provider, product)
    assert result.passed and result.iterations == 1
    assert provider.purposes().count("repair") == 1


def test_unrepairable_draft_becomes_a_failed_iteration_not_a_crash(make_agent, product):
    provider = make_provider(overrides={"draft": ["not json"], "repair": ["still not json"]})
    result = run(make_agent, provider, product)
    assert result.passed and result.iterations == 2  # iteration 2 revises from scratch and succeeds
    first = result.evaluations[0]
    assert [c.name for c in first.failed_checks] == ["schema_valid"]
    assert "after one repair attempt" in first.failed_checks[0].message
    assert result.draft is not None


def test_schema_violations_are_caught_by_validation(make_agent, product):
    wrong_shape = {
        "en": {
            "description": "x",
            "ad_copy": "y",
            "social_post": "z",
            "email_subjects": ["only one"],
        }
    }
    provider = make_provider(overrides={"draft": [wrong_shape], "repair": [wrong_shape]})
    result = run(make_agent, provider, product)
    assert result.evaluations[0].failed_checks[0].name == "schema_valid"


def test_unusable_judge_fails_closed(make_agent, product):
    provider = make_provider(overrides={"judge": ["???"], "repair": ["???"]})
    result = run(make_agent, provider, product)
    assert result.iterations == 2 and result.passed
    assert "unusable" in result.evaluations[0].judge.reasons[0]


# ------------------------------------------------------------------ tool transports


def test_json_protocol_transport_matches_native_results(make_agent, product):
    native = run(make_agent, make_provider(["banned_claim", "none"]), product)
    provider = make_provider(["banned_claim", "none"], supports_tools=False)
    proto = run(make_agent, provider, product)
    assert proto.passed and proto.draft == native.draft
    assert [c.name for c in proto.tool_calls if c.invoked_by == "model"] == [
        c.name for c in native.tool_calls if c.invoked_by == "model"
    ]
    plan_request = next(r for r in provider.requests if r.purpose == "plan")
    assert plan_request.tools == [] and "Tool calling protocol" in plan_request.messages[1].content
    assert json.loads(proto.report.model_dump_json())  # serialises


def test_native_transport_sends_tool_specs_to_the_provider(make_agent, product):
    provider = make_provider()
    run(make_agent, provider, product)
    plan_request = next(r for r in provider.requests if r.purpose == "plan")
    assert {t.name for t in plan_request.tools} == {"lookup_product_fact", "retrieve_guidelines"}
    assert not any(r.tools for r in provider.requests if r.purpose != "plan")


def test_protocol_violations_end_in_an_error_status(make_agent, product):
    provider = make_provider(
        overrides={"plan": ['{"action": "tool"}'], "repair": ['{"action": "tool"}']},
        supports_tools=False,
    )
    result = run(make_agent, provider, product)
    assert result.status == "error" and "ToolProtocolError" in result.error and result.draft is None


def test_unknown_tool_requests_are_reported_back_and_the_run_continues(make_agent, product):
    bogus = LLMResponse(content="", tool_calls=[ToolCall("x1", "format_disk", {"path": "/"})])
    result = run(make_agent, make_provider(overrides={"plan": [bogus]}), product)
    assert result.passed
    bad = [c for c in result.tool_calls if c.name == "format_disk"]
    assert len(bad) == 1 and not bad[0].ok and "unknown tool" in bad[0].result_preview


def test_invalid_tool_arguments_are_reported_back(make_agent, product):
    bad_args = LLMResponse(
        content="", tool_calls=[ToolCall("x1", "check_length", {"text": "x", "kind": "poem"})]
    )
    result = run(make_agent, make_provider(overrides={"plan": [bad_args]}), product)
    assert result.passed and any(
        c.name == "check_length" and not c.ok and c.invoked_by == "model" for c in result.tool_calls
    )


def test_agent_retrieves_guidelines_itself_if_the_model_skips_the_tool(make_agent, product):
    plan = {"angle": "a", "tone": "t", "key_facts": [], "guideline_notes": [], "must_avoid": []}
    provider = make_provider(overrides={"plan": [LLMResponse(content=json.dumps(plan))]})
    result = run(make_agent, provider, product)
    assert result.passed
    assert [c.invoked_by for c in result.tool_calls if c.name == "retrieve_guidelines"] == ["agent"]


# ------------------------------------------------------------------ reliability


def test_transient_provider_errors_are_retried_with_backoff(settings, retriever, product):
    slept: list[float] = []
    provider = make_provider(overrides={"plan": [ProviderTimeout("t1"), ProviderServerError("s2")]})
    agent = MarketingAgent(
        provider,
        replace(settings, llm_backoff_base_s=1.0, llm_backoff_max_s=8.0),
        retriever=retriever,
        sleep=slept.append,
    )
    result = agent.run(product)
    assert result.passed and len(slept) == 2
    assert 0.5 <= slept[0] <= 1.0 and 1.0 <= slept[1] <= 2.0  # equal jitter, doubling
    trace = json.loads(Path(result.trace_path).read_text())
    retried = [s for s in trace["steps"] if s.get("retries")]
    assert len(retried) == 1 and len(retried[0]["retries"]) == 2


def test_retries_are_bounded_and_end_in_an_error_status(make_agent, product):
    provider = make_provider(overrides={"plan": [ProviderTimeout("slow")] * 10})
    result = run(make_agent, provider, product, llm_max_retries=2)
    assert result.status == "error" and "ProviderTimeout" in result.error
    assert provider.purposes().count("plan") == 3  # 1 try + 2 retries
    assert result.draft is None and "Run failed" in result.report.summary


def test_permanent_provider_errors_are_not_retried(make_agent, product):
    provider = make_provider(overrides={"plan": [ProviderError("401 unauthorised")] * 5})
    result = run(make_agent, provider, product)
    assert result.status == "error" and provider.purposes().count("plan") == 1


def test_token_budget_stops_the_run(make_agent, product):
    result = run(make_agent, make_provider(), product, max_total_tokens=1500)
    assert result.status == "budget_exceeded" and "token budget" in result.error
    assert "not verified" in result.report.summary
    assert result.usage.total_tokens <= 1500


def test_llm_call_budget_stops_the_run(make_agent, product):
    provider = make_provider(["banned_claim"] * 3)
    result = run(make_agent, provider, product, max_llm_calls=3)
    assert result.status == "budget_exceeded" and "call limit" in result.error
    assert provider.calls == 3 and result.draft is None


def test_cost_budget_stops_the_run(make_agent, product):
    result = run(
        make_agent, make_provider(["banned_claim"] * 3), product,
        max_cost_usd=0.01, price_per_1k_prompt_usd=5.0,
    )  # fmt: skip
    assert result.status == "budget_exceeded" and "cost budget" in result.error
    assert result.usage.estimated_cost_usd > 0.01


def test_budget_exceeded_keeps_the_last_unapproved_draft_visible(make_agent, product):
    provider = make_provider(["banned_claim"] * 3)
    result = run(make_agent, provider, product, max_llm_calls=6)  # plan x4, draft, revise
    assert result.status == "budget_exceeded" and result.draft is not None and not result.passed
    assert result.iterations == 2


# ------------------------------------------------------------------ observability


def test_trace_file_has_steps_prompt_versions_latency_and_tokens(make_agent, product):
    result = run(make_agent, make_provider(["unsupported_number", "none"]), product)
    trace = json.loads(Path(result.trace_path).read_text())
    assert trace["run_id"] == result.run_id and trace["result"]["status"] == "passed"
    assert trace["meta"]["tool_transport"] == "native" and trace["meta"]["prompt_version"] == "v1"
    kinds = [s["step"] for s in trace["steps"]]
    assert kinds[0] == "prepare" and {
        "plan",
        "evaluate",
        "llm_call",
        "tool_call",
        "validate",
    } <= set(kinds)
    llm = [s for s in trace["steps"] if s["step"] == "llm_call"]
    assert {s["purpose"] for s in llm} == {"plan", "draft", "judge", "revise"}
    for s in llm:
        assert re.fullmatch(r"[a-z_]+@\d+\.\d+\.\d+#[0-9a-f]{8}", s["prompt"])
        assert s["latency_ms"] >= 0 and s["prompt_tokens"] > 0 and s["completion_tokens"] > 0
    assert trace["total_latency_ms"] > 0 and trace["result"]["usage"]["llm_calls"] == len(llm)
    evaluate = [s for s in trace["steps"] if s["step"] == "evaluate"]
    assert evaluate[0]["passed"] is False and "numbers_supported" in evaluate[0]["failed_checks"]


def test_traces_contain_previews_not_full_prompts_unless_enabled(make_agent, product):
    secret = ProductInput(
        id=PRODUCT_ID,
        name="Dell Latitude 5440",
        category="Laptop",
        notes="CONFIDENTIAL-MARKER " * 40,
    )
    short = json.loads(Path(run(make_agent, make_provider(), secret).trace_path).read_text())
    assert "CONFIDENTIAL-MARKER" not in json.dumps(short)
    full = json.loads(
        Path(run(make_agent, make_provider(), secret, trace_full_io=True).trace_path).read_text()
    )
    assert "CONFIDENTIAL-MARKER" in json.dumps(full)


def test_save_trace_can_be_disabled(make_agent, product):
    result = make_agent(make_provider()).run(product, save_trace=False)
    assert result.trace_path is None


def test_result_roundtrips_through_json(make_agent, product):
    result = run(make_agent, make_provider(["banned_claim", "none"]), product)
    again = AgentResult.model_validate_json(result.model_dump_json())
    assert again.draft == result.draft and again.report == result.report


# ------------------------------------------------------------------ baseline


def test_baseline_makes_one_draft_call_and_is_scored_by_the_same_evaluator(
    settings, retriever, product
):
    provider = make_provider()
    result = BaselineGenerator(provider, settings, retriever=retriever).run(product)
    assert result.passed and result.iterations == 1
    assert (
        provider.purposes() == ["baseline", "judge"]
        and result.tool_calls[0].invoked_by == "evaluator"
    )


def test_baseline_does_not_revise(settings, retriever, product):
    provider = make_provider(["banned_claim", "none"])
    result = BaselineGenerator(provider, settings, retriever=retriever).run(product)
    assert result.status == "failed" and result.iterations == 1
    assert "revise" not in provider.purposes()


def test_baseline_uses_json_repair_too(settings, retriever, product):
    provider = make_provider(["invalid_json"])
    result = BaselineGenerator(provider, settings, retriever=retriever).run(product)
    assert result.passed and provider.purposes().count("repair") == 1


def test_baseline_reports_unrepairable_output(settings, retriever, product):
    provider = make_provider(overrides={"baseline": ["nope"], "repair": ["nope"]})
    result = BaselineGenerator(provider, settings, retriever=retriever).run(product)
    assert (
        result.status == "failed" and result.evaluations[0].failed_checks[0].name == "schema_valid"
    )


def test_empty_model_reply_in_the_json_protocol_is_retried_not_repaired(make_agent, product):
    provider = make_provider(
        overrides={"plan": [EmptyResponse("empty completion")]}, supports_tools=False
    )
    result = make_agent(provider).run(product, ["en", "nl"])
    assert result.status == "passed"
    assert not [r for r in provider.requests if r.purpose == "repair"]


def test_the_judge_is_called_with_temperature_zero(make_agent, product):
    provider = make_provider()
    make_agent(provider).run(product, ["en", "nl"])
    judge_requests = [r for r in provider.requests if r.purpose == "judge"]
    assert judge_requests and all(r.temperature == 0.0 for r in judge_requests)
    assert all(r.temperature > 0 for r in provider.requests if r.purpose == "draft")
