"""Pydantic models: API contracts, model outputs, evaluation and run reports."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Lang = Literal["en", "nl"]
ALL_LANGS: tuple[Lang, ...] = ("en", "nl")

FIELD_KINDS = ("description", "ad_copy", "social_post", "email_subject")


class ProductInput(BaseModel):
    """Structured product data. ``notes``/``audience``/``specs`` are untrusted free text."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: Annotated[str | None, Field(max_length=64)] = None
    name: Annotated[str, Field(min_length=1, max_length=160)]
    category: Annotated[str, Field(min_length=1, max_length=80)]
    specs: dict[str, str] = Field(default_factory=dict)
    price: Annotated[float | None, Field(ge=0, le=10_000_000)] = None
    currency: Annotated[str, Field(min_length=3, max_length=3)] = "EUR"
    audience: Annotated[str, Field(max_length=300)] = ""
    notes: Annotated[str, Field(max_length=1500)] = ""

    @field_validator("specs", mode="before")
    @classmethod
    def _coerce_specs(cls, value: object) -> object:
        if value is None:
            return {}
        if isinstance(value, dict):
            if len(value) > 40:
                raise ValueError("at most 40 specs are allowed")
            out: dict[str, str] = {}
            for key, val in value.items():
                skey, sval = str(key).strip(), str(val).strip()
                if not skey or len(skey) > 60 or len(sval) > 300:
                    raise ValueError("spec keys must be 1-60 chars and values at most 300 chars")
                out[skey] = sval
            return out
        return value

    @field_validator("currency")
    @classmethod
    def _upper_currency(cls, value: str) -> str:
        return value.upper()


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: ProductInput
    languages: Annotated[list[Lang], Field(min_length=1, max_length=2)] = ["en", "nl"]

    @field_validator("languages")
    @classmethod
    def _unique(cls, value: list[Lang]) -> list[Lang]:
        if len(set(value)) != len(value):
            raise ValueError("languages must be unique")
        return value


# --------------------------------------------------------------------------- model outputs


class LanguageContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    ad_copy: str
    social_post: str
    email_subjects: Annotated[list[str], Field(min_length=3, max_length=3)]

    def iter_fields(self) -> list[tuple[str, str]]:
        """(field kind, text) pairs; each email subject is its own entry."""
        items = [
            ("description", self.description),
            ("ad_copy", self.ad_copy),
            ("social_post", self.social_post),
        ]
        items += [("email_subject", s) for s in self.email_subjects]
        return items

    def all_text(self) -> str:
        return "\n".join(text for _, text in self.iter_fields())


class MarketingDraft(BaseModel):
    """What the drafting model must return (JSON), validated before anything else sees it."""

    model_config = ConfigDict(extra="forbid")

    en: LanguageContent | None = None
    nl: LanguageContent | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> MarketingDraft:
        if self.en is None and self.nl is None:
            raise ValueError("draft must contain at least one language")
        return self

    def get(self, lang: str) -> LanguageContent | None:
        return getattr(self, lang, None)

    def languages(self) -> list[str]:
        return [lang for lang in ALL_LANGS if self.get(lang) is not None]


class Plan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    angle: Annotated[str, Field(min_length=1)]
    tone: Annotated[str, Field(min_length=1)]
    key_facts: list[str] = Field(default_factory=list)
    guideline_notes: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    passed: bool
    reasons: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fail_needs_reason(self) -> JudgeVerdict:
        if not self.passed and not (self.reasons or self.unsupported_claims):
            raise ValueError("a failing verdict must include at least one reason")
        return self


# --------------------------------------------------------------------------- evaluation


class CheckResult(BaseModel):
    name: str
    passed: bool
    message: str = ""
    details: dict[str, object] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    iteration: int
    passed: bool
    checks: list[CheckResult]
    judge: JudgeVerdict | None = None
    judge_skipped: str | None = None

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def feedback_lines(self) -> list[str]:
        """Human/LLM readable list of what must be fixed. Starts with the check code."""
        lines = [f"[{c.name}] {c.message}" for c in self.failed_checks]
        if self.judge is not None and not self.judge.passed:
            lines += [f"[llm_judge] {r}" for r in self.judge.reasons]
            lines += [f"[llm_judge] unsupported claim: {c}" for c in self.judge.unsupported_claims]
        return lines


# --------------------------------------------------------------------------- security


class InjectionFinding(BaseModel):
    field: str
    rule: str
    snippet: str


class SecurityReport(BaseModel):
    injection_detected: bool = False
    sanitized: bool = False
    findings: list[InjectionFinding] = Field(default_factory=list)


# --------------------------------------------------------------------------- run result


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, object]
    ok: bool
    result_preview: str
    invoked_by: Literal["model", "evaluator", "agent"] = "model"


class UsageSummary(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    tokens_estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RevisionNote(BaseModel):
    iteration: int
    problems: list[str]


class RunReport(BaseModel):
    """The "why this passed" (or failed) report. Built from evaluator evidence, never from the
    model's own claims about itself."""

    summary: str
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)
    judge_reasons: list[str] = Field(default_factory=list)
    revisions: list[RevisionNote] = Field(default_factory=list)
    guidelines_used: list[str] = Field(default_factory=list)
    data_warnings: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    run_id: str
    status: Literal["passed", "failed", "budget_exceeded", "error"]
    product_id: str | None = None
    languages: list[str]
    draft: MarketingDraft | None = None
    iterations: int = 0
    evaluations: list[EvaluationReport] = Field(default_factory=list)
    report: RunReport
    security: SecurityReport = Field(default_factory=SecurityReport)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    usage: UsageSummary = Field(default_factory=UsageSummary)
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    trace_path: str | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"
