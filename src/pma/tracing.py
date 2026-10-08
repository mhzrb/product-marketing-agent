"""Per-run JSON trace: every step with timing, prompt versions and token counts."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pma.logging_setup import get_logger

log = get_logger("trace")

_PREVIEW = 400


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


def preview(text: str, limit: int = _PREVIEW) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Trace:
    """Append-only list of steps, serialisable to JSON.

    By default prompts/outputs are stored as short previews. Set ``full_io=True`` (env
    ``TRACE_FULL_IO=true``) to keep complete messages for debugging; they may contain product
    data, so keep that off in shared environments.
    """

    def __init__(self, run_id: str | None = None, *, full_io: bool = False) -> None:
        self.run_id = run_id or new_run_id()
        self.full_io = full_io
        self.started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
        self._t0 = time.perf_counter()
        self.meta: dict[str, Any] = {}
        self.steps: list[dict[str, Any]] = []
        self.result: dict[str, Any] = {}

    def add(self, step: str, **fields: Any) -> dict[str, Any]:
        record = {"n": len(self.steps) + 1, "step": step, **fields}
        self.steps.append(record)
        log.debug("step", extra={"run_id": self.run_id, "step": step, **_loggable(fields)})
        return record

    @contextmanager
    def span(self, step: str, **fields: Any):
        """Time a non-LLM step. The yielded dict may be updated with extra fields."""
        extra: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            yield extra
        finally:
            self.add(
                step, latency_ms=round((time.perf_counter() - t0) * 1000, 2), **fields, **extra
            )

    def io(self, text: str) -> str:
        return text if self.full_io else preview(text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "total_latency_ms": round((time.perf_counter() - self._t0) * 1000, 2),
            "meta": self.meta,
            "steps": self.steps,
            "result": self.result,
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def _loggable(fields: dict[str, Any]) -> dict[str, Any]:
    keep = ("purpose", "latency_ms", "prompt", "prompt_tokens", "completion_tokens", "ok", "tool")
    return {k: v for k, v in fields.items() if k in keep}
