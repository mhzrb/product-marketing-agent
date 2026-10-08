"""FastAPI service: JSON API plus the minimal web page (plain HTML/JS, no build step).

There is deliberately NO authentication or rate limiting: run it behind a gateway if you expose it.
Product data is validated and size-limited by ``ProductInput``; model output is rendered in the page
with ``textContent`` only (never ``innerHTML``) and a strict CSP is sent with every response.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from pma import __version__
from pma.agent import MarketingAgent
from pma.baseline import BaselineGenerator
from pma.config import Settings
from pma.limits import LIMITS
from pma.logging_setup import configure_logging
from pma.providers import build_provider
from pma.providers.base import LLMProvider
from pma.rag.retriever import GuidelineRetriever, build_retriever
from pma.schemas import AgentResult, GenerateRequest, ProductInput

_RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}-[0-9a-f]{8}$")
_CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


def _static_dir() -> Path:
    return Path(str(resources.files("pma") / "static"))


def create_app(
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    retriever: GuidelineRetriever | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level, settings.log_format)
    provider = provider or build_provider(settings)
    retriever = retriever or build_retriever(settings)
    agent = MarketingAgent(provider, settings, retriever=retriever)
    baseline = BaselineGenerator(provider, settings, retriever=retriever)

    app = FastAPI(
        title="Product Marketing Agent",
        version=__version__,
        description="Plan -> draft -> evaluate -> revise agent for EN/NL B2B marketing copy.",
    )
    app.state.settings = settings
    app.state.provider = provider

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", _CSP)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    app.mount("/static", StaticFiles(directory=_static_dir()), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (_static_dir() / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "version": __version__,
            "provider": provider.name,
            "model": provider.model,
        }

    @app.get("/api/config")
    def config() -> dict[str, object]:
        """Non-secret runtime configuration (API keys are never included)."""
        return {
            "provider": provider.name,
            "model": provider.model,
            "tool_transport": "native" if provider.supports_tools else "json_protocol",
            "retriever": settings.retriever,
            "max_iterations": settings.agent_max_iterations,
            "judge_enabled": settings.judge_enabled,
            "sanitize_input": settings.sanitize_input,
            "prompt_version": settings.prompt_version,
            "limits": {k: {"min": lo, "max": hi} for k, (lo, hi) in LIMITS.items()},
            "simulated": provider.name == "mock",
        }

    @app.get("/api/examples")
    def examples() -> list[ProductInput]:
        raw = (resources.files("pma") / "data" / "examples.json").read_text(encoding="utf-8")
        return [ProductInput.model_validate(item) for item in json.loads(raw)]

    @app.post("/api/generate")
    def generate(request: GenerateRequest) -> AgentResult:
        """Run the full agent loop. Business outcomes (``failed``, ``budget_exceeded``, ``error``)
        are reported in ``status`` with HTTP 200 so the caller always gets the report."""
        return agent.run(request.product, list(request.languages))

    @app.post("/api/baseline")
    def run_baseline(request: GenerateRequest) -> AgentResult:
        """Single-prompt baseline, evaluated with the same checks (for comparison)."""
        return baseline.run(request.product, list(request.languages), save_trace=True)

    @app.get("/api/traces/{run_id}")
    def get_trace(run_id: str) -> JSONResponse:
        if not _RUN_ID.match(run_id):  # also blocks path traversal
            raise HTTPException(status_code=400, detail="invalid run id")
        path = settings.trace_dir / f"{run_id}.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="trace not found")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/guidelines/search")
    def search_guidelines(
        q: str = Query(min_length=2, max_length=200), k: int = Query(3, ge=1, le=5)
    ) -> list[dict[str, object]]:
        hits = retriever.search(q, k)
        return [
            {"source": h.chunk.ref, "score": round(h.score, 4), "text": h.chunk.text} for h in hits
        ]

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> Response:
        # Never leak internals; the structured log has the details.
        from pma.logging_setup import get_logger

        get_logger("api").exception("unhandled error", exc_info=exc)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    return app


def app_factory() -> FastAPI:
    """``uvicorn pma.api:app_factory --factory`` (used by the Dockerfile)."""
    return create_app()
