from __future__ import annotations

import pytest

from pma.facts import FactBase
from pma.tools import ToolRegistry, check_banned_claims, check_length


@pytest.fixture
def registry(product, retriever):
    return ToolRegistry(FactBase.build(product), retriever)


def test_check_length_ok_and_too_long():
    assert check_length("A" * 100, "ad_copy")["ok"] is True
    long = check_length("A" * 200, "ad_copy")
    assert long["ok"] is False and long["over_by"] == 50 and long["max"] == 150


def test_check_length_too_short():
    short = check_length("Hi", "social_post")
    assert short["ok"] is False and short["under_by"] == 28


@pytest.mark.parametrize(
    "text",
    [
        "The best in the world for IT teams",
        "100% guaranteed to work",
        "A risk-free purchase",
        "Unhackable security",
        "Our award-winning range",
        "Act now while stocks last",
        "Faster than the competition",
        "Eco-friendly and carbon-neutral",
        "De beste ter wereld",
        "Gegarandeerd tevreden",
        "Onhackbare beveiliging",
        "De goedkoopste van Nederland",
        "Laatste kans!",
    ],
)
def test_banned_claims_detected_in_both_languages(text):
    assert check_banned_claims(text)["ok"] is False, text


@pytest.mark.parametrize(
    "text",
    [
        "A dependable laptop with a 3 year warranty",
        "Een betrouwbare laptop met 3 jaar garantie",
        "Supports RAID 1 and 5",
        "Designed for hybrid offices",
    ],
)
def test_clean_text_is_not_flagged(text):
    assert check_banned_claims(text)["ok"] is True, text


def test_banned_claims_report_reason_and_category():
    match = check_banned_claims("guaranteed")["matches"][0]
    assert match["category"] == "guarantee" and "warranty" in match["reason"]


def test_registry_executes_and_records(registry):
    ex = registry.execute("lookup_product_fact", {"key": "price"})
    assert ex.ok and ex.result["value"] == "1299.00 EUR"
    assert registry.executions[-1] is ex
    assert ex.to_record().invoked_by == "model"


def test_registry_rejects_unknown_tool_without_raising(registry):
    ex = registry.execute("delete_everything", {})
    assert not ex.ok and "unknown tool" in ex.result["error"]


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("check_length", {"text": "x", "kind": "poem"}),
        ("check_length", {"text": "x"}),
        ("lookup_product_fact", {"key": ""}),
        ("retrieve_guidelines", {"query": "ok", "k": 99}),
        ("check_banned_claims", {"text": "x", "extra": 1}),
    ],
)
def test_registry_validates_arguments(registry, name, args):
    ex = registry.execute(name, args)
    assert not ex.ok and "invalid arguments" in ex.result["error"]


def test_retrieve_guidelines_tracks_sources(registry):
    ex = registry.execute("retrieve_guidelines", {"query": "forbidden claims guaranteed", "k": 2})
    assert ex.ok and len(ex.result["results"]) == 2
    assert registry.guideline_refs and set(registry.guideline_refs) == set(registry.guideline_texts)
    assert ex.result["fallback_to_bm25"] is False


def test_tool_specs_are_valid_json_schema_objects(registry):
    specs = {s.name: s for s in registry.specs()}
    assert set(specs) == {
        "check_length",
        "check_banned_claims",
        "lookup_product_fact",
        "retrieve_guidelines",
    }
    for spec in specs.values():
        assert spec.parameters["type"] == "object" and "properties" in spec.parameters
        assert "title" not in spec.parameters
    assert [s.name for s in registry.specs(["check_length"])] == ["check_length"]
