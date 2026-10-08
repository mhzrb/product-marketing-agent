"""Prompt-injection defence for untrusted product data.

Layers (defence in depth, none of them is claimed to be complete):

1. **Detect + quarantine** (this module): heuristic patterns (EN + NL) are matched sentence by
   sentence in every free-text field. Matching sentences are removed from the data before it is
   shown to the model and are reported in ``SecurityReport`` (flag, don't silently drop).
2. **Data/instruction separation** (prompt design): product data is passed as a JSON block that
   the system prompt declares to be untrusted data; ``<`` is escaped so it cannot close the block.
3. **Output verification** (``evaluator``): the generated copy is scanned for injection phrases and
   verbatim payload fragments, and every number/claim must be supported by the *sanitised* data.

Known limits: regex heuristics miss paraphrased or obfuscated attacks and can false-positive on
innocent text ("act as one unit"). That trade-off is documented in the README.
"""

from __future__ import annotations

import re

from pma.schemas import InjectionFinding, ProductInput, SecurityReport

REMOVED_MARKER = "[removed: suspected prompt injection]"

# Invisible characters commonly used to hide instructions from human reviewers.
_HIDDEN_CHARS = re.compile("[​-‏⁠﻿‪-‮⁦-⁩]")

_F = re.IGNORECASE
_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
            r"\b(instructions?|rules?|prompts?|guidelines?|directions?|context)\b",
            _F,
        ),
    ),
    (
        "ignore_instructions_nl",
        re.compile(
            r"\b(negeer|vergeet|omzeil|passeer)\b[^.\n]{0,40}"
            r"\b(instructies|regels|richtlijnen|prompt|opdracht|voorgaande|vorige|eerdere)\b",
            _F,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\b(you are now|from now on,? you|pretend (to be|you are)"
            r"|act as (an? |the |my )?(unrestricted|different|new|evil|assistant|system"
            r"|developer|admin|ai|chatbot)"
            r"|je bent nu|vanaf nu ben je|doe alsof je|gedraag je als)\b",
            _F,
        ),
    ),
    (
        "reveal_prompt",
        re.compile(
            r"\b(reveal|show|print|repeat|leak|output|toon|onthul|geef)\b[^.\n]{0,40}"
            r"\b(system prompt|your (instructions|prompt)|hidden instructions|systeemprompt"
            r"|je instructies|verborgen instructies)\b",
            _F,
        ),
    ),
    (
        "output_hijack",
        re.compile(
            r"\b(say|write|state|claim|declare|mention|tell (the )?(reader|customer)s?)\s+that\s+"
            r"(this|the|our|it)\b"
            r"|\b(schrijf|zeg|vermeld|beweer|verklaar)\s+dat\s+(dit|het|deze|onze)\b"
            r"|\b(respond|reply|answer) (only )?with\b",
            _F,
        ),
    ),
    (
        "fake_role_marker",
        re.compile(
            r"(^|\n)\s*(system|assistant|developer)\s*:"
            r"|<\s*/?\s*(system|assistant|instructions?|product_data|plan|feedback)\s*>"
            r"|\[/?INST\]|<\|im_(start|end)\|>|###\s*(system|instruction)",
            _F,
        ),
    ),
    ("mode_switch", re.compile(r"\b(developer mode|jailbreak|do anything now)\b", _F)),
    ("tool_abuse", re.compile(r"\b(call|invoke|run|execute)\s+(the\s+)?(tool|function)\b", _F)),
]

# Output-only extras: text that must never appear in finished marketing copy.
_OUTPUT_EXTRA: list[tuple[str, re.Pattern[str]]] = [
    ("llm_meta_text", re.compile(r"\b(as an ai( language model)?|system prompt)\b", _F)),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _snippet(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def scan_text(text: str, field: str = "text") -> list[InjectionFinding]:
    """Return one finding per (sentence, rule) match."""
    findings: list[InjectionFinding] = []
    if _HIDDEN_CHARS.search(text):
        findings.append(
            InjectionFinding(field=field, rule="hidden_characters", snippet="(invisible)")
        )
    for sentence in _sentences(_HIDDEN_CHARS.sub("", text)):
        for rule, pattern in _RULES:
            if pattern.search(sentence):
                findings.append(
                    InjectionFinding(field=field, rule=rule, snippet=_snippet(sentence))
                )
    return findings


def scan_output(text: str, payload_fragments: list[str] | None = None) -> list[InjectionFinding]:
    """Scan *generated* copy: injection phrases, LLM meta text, or echoed payload fragments."""
    findings = scan_text(text, field="output")
    cleaned = _HIDDEN_CHARS.sub("", text)
    for rule, pattern in _OUTPUT_EXTRA:
        if pattern.search(cleaned):
            findings.append(InjectionFinding(field="output", rule=rule, snippet=_snippet(cleaned)))
    lowered = " ".join(cleaned.lower().split())
    for fragment in payload_fragments or []:
        norm = " ".join(fragment.lower().split())
        if len(norm) >= 15 and norm in lowered:
            findings.append(
                InjectionFinding(field="output", rule="payload_echo", snippet=_snippet(fragment))
            )
    return findings


def _clean_field(value: str, field: str, findings: list[InjectionFinding]) -> str:
    """Remove flagged sentences from one field, collecting findings."""
    findings.extend(scan_text(value, field))
    value = _HIDDEN_CHARS.sub("", value)
    sentences = _sentences(value)
    kept = [s for s in sentences if not any(p.search(s) for _, p in _RULES)]
    if len(kept) == len(sentences):
        return value  # nothing removed: keep the original spacing
    return " ".join(kept).strip() or REMOVED_MARKER


def sanitize_product(product: ProductInput) -> tuple[ProductInput, SecurityReport]:
    """Return a cleaned copy of ``product`` plus a report of what was removed."""
    findings: list[InjectionFinding] = []
    name = _clean_field(product.name, "name", findings)
    category = _clean_field(product.category, "category", findings)
    if name != product.name and name != REMOVED_MARKER:
        name = name.strip(" .;:,-")
    if category != product.category and category != REMOVED_MARKER:
        category = category.strip(" .;:,-")
    audience = _clean_field(product.audience, "audience", findings)
    notes = _clean_field(product.notes, "notes", findings)
    specs: dict[str, str] = {}
    for key, value in product.specs.items():
        key_findings: list[InjectionFinding] = []
        clean_key = _clean_field(key, f"specs.{key}(key)", key_findings)
        findings.extend(key_findings)
        if clean_key == REMOVED_MARKER or clean_key != _HIDDEN_CHARS.sub("", key):
            continue  # a spec *name* that carries instructions is dropped entirely
        clean_value = _clean_field(value, f"specs.{key}", findings)
        if clean_value == REMOVED_MARKER:
            continue  # a spec whose value is only a payload carries no facts
        specs[clean_key] = clean_value
    cleaned = product.model_copy(
        update={
            "name": name,
            "category": category,
            "audience": audience,
            "notes": "" if notes == REMOVED_MARKER else notes,
            "specs": specs,
        }
    )
    report = SecurityReport(
        injection_detected=bool(findings),
        sanitized=bool(findings),
        findings=findings,
    )
    return cleaned, report


def payload_fragments(report: SecurityReport) -> list[str]:
    """Snippets of removed payloads, used to detect verbatim echo in the generated copy."""
    return [f.snippet for f in report.findings if f.rule != "hidden_characters"]
