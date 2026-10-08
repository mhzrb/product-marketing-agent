"""Loading the evaluation products from CSV.

Columns: ``id, name, category, price, currency, audience, specs, notes, languages, case_type,
mock_script, expect``

* ``specs``       - ``Key=Value|Key=Value`` (duplicate keys are kept: that is how conflicting data
                    is expressed)
* ``languages``   - ``en|nl`` (default) or a single language
* ``mock_script`` - flaw sequence for the *simulated* model, e.g. ``unsupported_number|none``;
                    only used with the mock provider and ignored for real providers
* ``expect``      - pipeline signals the harness verifies independent of any model:
                    ``injection``, ``conflict``, ``missing_specs``, ``missing_price``
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from pma.schemas import ProductInput


@dataclass
class EvalCase:
    product: ProductInput
    languages: list[str]
    case_type: str
    mock_script: list[str] = field(default_factory=list)
    expect: set[str] = field(default_factory=set)


def parse_specs(raw: str) -> dict[str, str]:
    specs: dict[str, str] = {}
    for pair in filter(None, (p.strip() for p in raw.split("|"))):
        key, sep, value = pair.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if (
            key in specs
        ):  # duplicate key: keep both by suffixing (still conflicts by normalised key)
            n = 2
            while f"{key} ({n})" in specs:
                n += 1
            key = f"{key} ({n})"
        specs[key] = value
    return specs


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            price = (row.get("price") or "").strip()
            product = ProductInput(
                id=row["id"].strip(),
                name=row["name"],
                category=row["category"],
                specs=parse_specs(row.get("specs", "")),
                price=float(price) if price else None,
                currency=(row.get("currency") or "EUR").strip() or "EUR",
                audience=row.get("audience", "") or "",
                notes=row.get("notes", "") or "",
            )
            languages = [x for x in (row.get("languages") or "en|nl").split("|") if x] or [
                "en",
                "nl",
            ]
            script = [x for x in (row.get("mock_script") or "").split("|") if x]
            expect = {x for x in (row.get("expect") or "").split("|") if x}
            cases.append(
                EvalCase(
                    product, languages, (row.get("case_type") or "normal").strip(), script, expect
                )
            )
    return cases
