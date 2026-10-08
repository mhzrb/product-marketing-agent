from __future__ import annotations

import pytest

from pma.facts import (
    FactBase,
    extract_numbers,
    extract_unit_pairs,
    normalize_number,
)
from pma.schemas import ProductInput


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.299,00", "1299"),
        ("1,299.00", "1299"),
        ("1299", "1299"),
        ("1.299", "1299"),  # 3 digits after a single separator -> thousands
        ("2,5", "2.5"),
        ("2.5", "2.5"),
        ("0,500", "0.5"),  # leading zero -> decimal
        ("1.234.567", "1234567"),
        ("1299.90", "1299.9"),
    ],
)
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_extract_numbers_is_notation_independent():
    assert extract_numbers("EUR 1.299,00 or 1,299.00") == {"1299"}


def test_unit_pairs_normalise_spelling():
    pairs = extract_unit_pairs('16GB RAM, 16 gb, 27-inch, 27", 65 W, 99%, 3 hours')
    assert ("16", "gb") in pairs
    assert ("27", "inch") in pairs
    assert ("65", "w") in pairs
    assert ("99", "%") in pairs
    assert ("3", "h") in pairs


def test_unit_pairs_ignore_words_that_merely_start_with_a_unit():
    assert extract_unit_pairs("5 gbit and 3 hello") == set()


def test_factbase_collects_numbers_from_all_fields(product):
    fb = FactBase.build(product)
    assert {"16", "512", "14", "1299", "5440", "2"} <= fb.allowed_numbers
    assert ("16", "gb") in fb.allowed_pairs
    assert fb.warnings == []


def test_missing_price_specs_and_audience_produce_warnings():
    fb = FactBase.build(ProductInput(name="Cable", category="Cable"))
    text = " ".join(fb.warnings)
    assert "no price" in text and "no specs" in text and "no audience" in text


def test_conflicting_specs_are_excluded_and_reported():
    p = ProductInput(
        name="Laptop X",
        category="Laptop",
        specs={"RAM": "16 GB", "Memory": "32 GB", "Storage": "512 GB"},
    )
    fb = FactBase.build(p)
    assert "ram" in fb.conflicts
    assert "ram" not in fb.facts
    assert "32" not in fb.allowed_numbers and "16" not in fb.allowed_numbers
    assert any("conflicting values for 'ram'" in w for w in fb.warnings)
    assert "RAM" not in fb.prompt_view()["specs"]


def test_duplicate_but_identical_values_are_not_a_conflict():
    p = ProductInput(name="Laptop X", category="Laptop", specs={"RAM": "16 GB", "Memory": "16gb"})
    assert FactBase.build(p).conflicts == {}


def test_lookup_statuses(product):
    fb = FactBase.build(product)
    assert fb.lookup("price")["value"] == "1299.00 EUR"
    assert fb.lookup("RAM")["value"] == "16 GB"
    assert fb.lookup("memory")["value"] == "16 GB"  # synonym
    assert fb.lookup("warranty") == {"found": False, "key": "warranty", "status": "missing"}
    assert "ram" in fb.lookup("*")["keys"]


def test_lookup_reports_conflicts():
    fb = FactBase.build(
        ProductInput(name="N", category="C", specs={"Warranty": "1 year", "Garantie": "3 jaar"})
    )
    result = fb.lookup("warranty")
    assert result["found"] is False and result["status"] == "conflicting"
