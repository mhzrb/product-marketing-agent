"""Evaluation of a draft: deterministic checks first, then an optional LLM judge.

Deterministic checks (each produces a ``CheckResult`` whose ``name`` is the code used in feedback):

==================== ==============================================================================
``schema_complete``   every requested language present, no unrequested one
``length``            every field within its channel limit (via the ``check_length`` tool)
``email_subjects_distinct`` the three subject lines of a language differ
``banned_claims``     no forbidden claim (via the ``check_banned_claims`` tool)
``numbers_supported`` every number in the copy exists in the (sanitised) product data
``specs_supported``   every ``number + unit`` pair (16 GB, 27 inch ...) exists in the data
``conflicts_not_used`` values of conflicting attributes do not appear
``language_match``    description/social text looks like the language it is labelled with
``mentions_product``  the description names the product
``injection_echo``    no injection phrases or echoed payload fragments in the copy
``no_placeholders``   no template leftovers (``[removed``, ``${x}``, ``TODO`` ...)
==================== ==============================================================================

The LLM judge only runs when all deterministic checks pass: it exists for what regexes cannot see
(qualitative unsupported claims such as "trusted by thousands of IT departments").
"""

from __future__ import annotations

import re
from collections.abc import Callable

from pma.facts import NUMBER_RE, FactBase, extract_numbers, extract_unit_pairs, normalize_number
from pma.schemas import (
    CheckResult,
    EvaluationReport,
    JudgeVerdict,
    MarketingDraft,
    SecurityReport,
)
from pma.security import payload_fragments, scan_output
from pma.tools import ToolRegistry

_EN_STOP = {
    "the", "and", "for", "with", "are", "an", "of", "to", "your", "our", "it", "that", "this", "a",
}  # fmt: skip
_NL_STOP = {
    "de", "het", "een", "en", "voor", "met", "zijn", "van", "uw", "u", "op", "dat", "dit", "deze",
    "bij", "als",
}  # fmt: skip
_PLACEHOLDER = re.compile(r"\[removed|\bTODO\b|\$\{|\{\w*\}|lorem ipsum|<\w+>", re.IGNORECASE)

JudgeFn = Callable[[MarketingDraft, list[CheckResult]], JudgeVerdict]


def _canon(text: str) -> str:
    """Comparison form: lower case, canonical numbers, ``16gb`` == ``16 GB``, hyphens as spaces."""
    t = NUMBER_RE.sub(lambda m: normalize_number(m.group(0)), text.lower())
    t = re.sub(r"(?<=\d)(?=[a-z%])", " ", t)
    return re.sub(r"[\s\-]+", " ", t).strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zà-ÿ]+", text.lower())


def _stop_counts(text: str) -> tuple[int, int]:
    words = _words(text)
    return sum(w in _EN_STOP for w in words), sum(w in _NL_STOP for w in words)


class Evaluator:
    def __init__(
        self,
        registry: ToolRegistry,
        facts: FactBase,
        languages: list[str],
        security: SecurityReport | None = None,
    ) -> None:
        self.registry = registry
        self.facts = facts
        self.languages = languages
        self.security = security or SecurityReport()

    # ------------------------------------------------------------------ public API
    def evaluate(
        self, draft: MarketingDraft, iteration: int, judge: JudgeFn | None = None
    ) -> EvaluationReport:
        checks = self.deterministic(draft)
        failed = [c for c in checks if not c.passed]
        verdict: JudgeVerdict | None = None
        skipped: str | None = None
        if judge is None:
            skipped = "judge disabled"
        elif failed:
            skipped = "deterministic checks failed; judge not run"
        else:
            verdict = judge(draft, checks)
        passed = not failed and (verdict is None or verdict.passed)
        return EvaluationReport(
            iteration=iteration, passed=passed, checks=checks, judge=verdict, judge_skipped=skipped
        )

    def deterministic(self, draft: MarketingDraft) -> list[CheckResult]:
        results = [self._schema_complete(draft)]
        present = [lang for lang in self.languages if draft.get(lang) is not None]
        results += [
            self._length(draft, present),
            self._subjects_distinct(draft, present),
            self._banned(draft, present),
            self._numbers(draft, present),
            self._units(draft, present),
            self._conflicts(draft, present),
            self._language(draft, present),
            self._mentions_product(draft, present),
            self._injection(draft, present),
            self._placeholders(draft, present),
        ]
        return results

    # ------------------------------------------------------------------ checks
    def _schema_complete(self, draft: MarketingDraft) -> CheckResult:
        have = set(draft.languages())
        want = set(self.languages)
        missing, extra = sorted(want - have), sorted(have - want)
        if not missing and not extra:
            return CheckResult(name="schema_complete", passed=True, message="all languages present")
        parts = []
        if missing:
            parts.append(f"missing language(s): {', '.join(missing)}")
        if extra:
            parts.append(f"unrequested language(s): {', '.join(extra)}")
        return CheckResult(
            name="schema_complete",
            passed=False,
            message="; ".join(parts),
            details={"missing": missing, "extra": extra},
        )

    def _length(self, draft: MarketingDraft, langs: list[str]) -> CheckResult:
        problems: list[str] = []
        for lang in langs:
            content = draft.get(lang)
            assert content is not None
            for kind, text in content.iter_fields():
                ex = self.registry.execute(
                    "check_length", {"text": text, "kind": kind}, invoked_by="evaluator"
                )
                r = ex.result
                if not r.get("ok"):
                    if r.get("over_by"):
                        problems.append(
                            f"{lang}.{kind} is {r['length']} chars, limit {r['max']} "
                            f"(shorten by {r['over_by']})"
                        )
                    else:
                        problems.append(f"{lang}.{kind} is {r['length']} chars, minimum {r['min']}")
        return self._result("length", problems, "all fields within channel limits")

    def _subjects_distinct(self, draft: MarketingDraft, langs: list[str]) -> CheckResult:
        problems = []
        for lang in langs:
            content = draft.get(lang)
            assert content is not None
            lowered = [s.strip().lower() for s in content.email_subjects]
            if len(set(lowered)) != len(lowered):
                problems.append(f"{lang}.email_subjects contain duplicates")
        return self._result("email_subjects_distinct", problems, "subject lines are distinct")

    def _banned(self, draft: MarketingDraft, langs: list[str]) -> CheckResult:
        problems = []
        for lang in langs:
            content = draft.get(lang)
            assert content is not None
            for kind, text in content.iter_fields():
                ex = self.registry.execute(
                    "check_banned_claims", {"text": text}, invoked_by="evaluator"
                )
                for m in ex.result.get("matches", []):
                    problems.append(f"{lang}.{kind}: '{m['phrase']}' ({m['reason']})")
        return self._result("banned_claims", problems, "no forbidden claims")

    def _numbers(self, draft: MarketingDraft, langs: list[str]) -> CheckResult:
        problems = []
        for lang in langs:
            content = draft.get(lang)
            assert content is not None
            for kind, text in content.iter_fields():
                bad = sorted(extract_numbers(text) - self.facts.allowed_numbers)
                if bad:
                    listed = ", ".join(bad)
                    problems.append(
                        f"{lang}.{kind} contains number(s) not in the product data: {listed}"
                    )
        return self._result(
            "numbers_supported", problems, "every number exists in the product data"
        )

    def _units(self, draft: MarketingDraft, langs: list[str]) -> CheckResult:
        problems = []
        for lang in langs:
            content = draft.get(lang)
            assert content is not None
            for kind, text in content.iter_fields():
                bad = sorted(extract_unit_pairs(text) - self.facts.allowed_pairs)
                if bad:
                    shown = ", ".join(f"{n} {u}" for n, u in bad)
                    problems.append(
                        f"{lang}.{kind} states spec(s) not in the product data: {shown}"
                    )
        return self._result(
            "specs_supported", problems, "every number+unit spec exists in the data"
        )

    def _conflicts(self, draft: MarketingDraft, langs: list[str]) -> CheckResult:
        if not self.facts.conflicts:
            return CheckResult(
                name="conflicts_not_used", passed=True, message="no conflicts in data"
            )
        problems = []
        name = _canon(self.facts.product.name)
        for lang in langs:
            content = draft.get(lang)
            assert content is not None
            text = _canon(content.all_text())
            if name:  # numbers inside the product name (USW-Pro-24) are not spec claims
                text = text.replace(name, " ")
            for key, entries in self.facts.conflicts.items():
                for entry in entries:
                    value = _canon(entry.split(":", 1)[1])
                    if value and re.search(rf"(?<![\w.]){re.escape(value)}(?!\w)", text):
                        problems.append(
                            f"{lang} uses '{value}' although '{key}' has conflicting values"
                        )
        return self._result("conflicts_not_used", problems, "conflicting attributes were left out")

    def _language(self, draft: MarketingDraft, langs: list[str]) -> CheckResult:
        problems = []
        for lang in langs:
            content = draft.get(lang)
            assert content is not None
            for kind, text in (
                ("description", content.description),
                ("social_post", content.social_post),
            ):
                en, nl = _stop_counts(text)
                own, other = (en, nl) if lang == "en" else (nl, en)
                if other > own and other >= 3:
                    problems.append(f"{lang}.{kind} does not read as {lang.upper()} text")
        return self._result("language_match", problems, "text matches its language label")

    def _mentions_product(self, draft: MarketingDraft, langs: list[str]) -> CheckResult:
        name = self.facts.product.name.strip().lower()
        needle = name if len(name) <= 40 else name[:30]
        problems = []
        for lang in langs:
            content = draft.get(lang)
            assert content is not None
            if needle and needle not in content.description.lower():
                shown = self.facts.product.name
                problems.append(f'{lang}.description does not contain the product name "{shown}"')
        return self._result("mentions_product", problems, "description names the product")

    def _injection(self, draft: MarketingDraft, langs: list[str]) -> CheckResult:
        fragments = payload_fragments(self.security)
        problems = []
        for lang in langs:
            content = draft.get(lang)
            assert content is not None
            for f in scan_output(content.all_text(), fragments):
                problems.append(f"{lang} copy contains injected/instruction text ({f.rule})")
        return self._result("injection_echo", problems, "no injected instructions in the copy")

    def _placeholders(self, draft: MarketingDraft, langs: list[str]) -> CheckResult:
        problems = []
        for lang in langs:
            content = draft.get(lang)
            assert content is not None
            if _PLACEHOLDER.search(content.all_text()):
                problems.append(f"{lang} copy contains a placeholder or template leftover")
        return self._result("no_placeholders", problems, "no template leftovers")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _result(name: str, problems: list[str], ok_message: str) -> CheckResult:
        if not problems:
            return CheckResult(name=name, passed=True, message=ok_message)
        return CheckResult(
            name=name,
            passed=False,
            message="; ".join(problems[:6]) + (" ..." if len(problems) > 6 else ""),
            details={"problems": problems},
        )
