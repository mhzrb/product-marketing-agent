"""The four agent tools and the registry that validates arguments and records every call.

* ``check_length``          - length of a text against its channel limit
* ``check_banned_claims``   - forbidden marketing claims (EN + NL regex rules)
* ``lookup_product_fact``   - read one fact from the sanitised product data
* ``retrieve_guidelines``   - BM25 (or embeddings/hybrid) search over the brand guidelines

The model may call them (native tool calling or the JSON protocol); the evaluator calls the first
two itself so the checks are identical whoever invokes them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pma.facts import FactBase
from pma.limits import LIMITS
from pma.providers.base import ToolSpec
from pma.rag.retriever import GuidelineRetriever
from pma.schemas import ToolCallRecord
from pma.tracing import preview

# --------------------------------------------------------------------------- pure functions


@lru_cache(maxsize=1)
def _banned_rules() -> list[tuple[str, str, re.Pattern[str]]]:
    raw = (resources.files("pma") / "data" / "banned_phrases.json").read_text(encoding="utf-8")
    rules = json.loads(raw)["rules"]
    return [(r["category"], r["reason"], re.compile(r["pattern"], re.IGNORECASE)) for r in rules]


def check_length(text: str, kind: str) -> dict[str, Any]:
    low, high = LIMITS[kind]
    length = len(text.strip())
    return {
        "ok": low <= length <= high,
        "kind": kind,
        "length": length,
        "min": low,
        "max": high,
        "over_by": max(0, length - high),
        "under_by": max(0, low - length),
    }


def check_banned_claims(text: str) -> dict[str, Any]:
    matches = []
    for category, reason, pattern in _banned_rules():
        for m in pattern.finditer(text):
            matches.append({"phrase": m.group(0), "category": category, "reason": reason})
    return {"ok": not matches, "matches": matches}


# --------------------------------------------------------------------------- argument models


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CheckLengthArgs(_Args):
    text: Annotated[str, Field(max_length=5000)]
    kind: Literal["description", "ad_copy", "social_post", "email_subject"]


class CheckBannedArgs(_Args):
    text: Annotated[str, Field(max_length=5000)]


class LookupArgs(_Args):
    key: Annotated[str, Field(min_length=1, max_length=80)]


class RetrieveArgs(_Args):
    query: Annotated[str, Field(min_length=2, max_length=300)]
    k: Annotated[int, Field(ge=1, le=5)] = 3


@dataclass
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: dict[str, Any]
    invoked_by: Literal["model", "evaluator", "agent"] = "model"

    def to_record(self) -> ToolCallRecord:
        return ToolCallRecord(
            name=self.name,
            arguments=self.arguments,
            ok=self.ok,
            result_preview=preview(json.dumps(self.result, ensure_ascii=False), 300),
            invoked_by=self.invoked_by,
        )

    def to_message_text(self) -> str:
        return json.dumps(self.result, ensure_ascii=False)


# --------------------------------------------------------------------------- registry

_DESCRIPTIONS = {
    "check_length": "Check whether a text fits the length limits of its channel.",
    "check_banned_claims": (
        "Find forbidden marketing claims (superlatives, guarantees, ...) in a text."
    ),
    "lookup_product_fact": (
        "Look up one fact about the product (e.g. price, ram, storage, audience). "
        "Use '*' to list available keys. Conflicting attributes are reported as such."
    ),
    "retrieve_guidelines": "Search the brand guidelines and return the most relevant passages.",
}
_MODELS: dict[str, type[_Args]] = {
    "check_length": CheckLengthArgs,
    "check_banned_claims": CheckBannedArgs,
    "lookup_product_fact": LookupArgs,
    "retrieve_guidelines": RetrieveArgs,
}


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


class ToolRegistry:
    def __init__(self, facts: FactBase, retriever: GuidelineRetriever) -> None:
        self.facts = facts
        self.retriever = retriever
        self.executions: list[ToolExecution] = []
        self.guideline_refs: list[str] = []
        self.guideline_texts: dict[str, str] = {}

    def specs(self, allowed: list[str] | None = None) -> list[ToolSpec]:
        names = allowed if allowed is not None else list(_MODELS)
        return [ToolSpec(n, _DESCRIPTIONS[n], _schema(_MODELS[n])) for n in names]

    def names(self) -> list[str]:
        return list(_MODELS)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        invoked_by: Literal["model", "evaluator", "agent"] = "model",
    ) -> ToolExecution:
        """Never raises: bad tool names/arguments come back as ``ok=False`` results the model can
        read and correct on its next turn."""
        if name not in _MODELS:
            return self._record(
                ToolExecution(
                    name, arguments, False, {"error": f"unknown tool '{name}'"}, invoked_by
                )
            )
        try:
            args = _MODELS[name].model_validate(arguments)
        except ValidationError as exc:
            msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:4])
            return self._record(
                ToolExecution(
                    name, arguments, False, {"error": f"invalid arguments: {msg}"}, invoked_by
                )
            )
        result = self._run(name, args)
        return self._record(ToolExecution(name, arguments, True, result, invoked_by))

    def _run(self, name: str, args: BaseModel) -> dict[str, Any]:
        if name == "check_length":
            assert isinstance(args, CheckLengthArgs)
            return check_length(args.text, args.kind)
        if name == "check_banned_claims":
            assert isinstance(args, CheckBannedArgs)
            return check_banned_claims(args.text)
        if name == "lookup_product_fact":
            assert isinstance(args, LookupArgs)
            return self.facts.lookup(args.key)
        assert isinstance(args, RetrieveArgs)
        hits = self.retriever.search(args.query, args.k)
        for h in hits:
            if h.chunk.ref not in self.guideline_refs:
                self.guideline_refs.append(h.chunk.ref)
                self.guideline_texts[h.chunk.ref] = h.chunk.text
        return {
            "results": [
                {"source": h.chunk.ref, "score": round(h.score, 4), "text": h.chunk.text}
                for h in hits
            ],
            "fallback_to_bm25": self.retriever.last_fallback,
        }

    def _record(self, execution: ToolExecution) -> ToolExecution:
        self.executions.append(execution)
        return execution
