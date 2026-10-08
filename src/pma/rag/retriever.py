"""Guideline retrieval: chunking, BM25 (default), optional embeddings and hybrid fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from pma.config import Settings
from pma.errors import PMAError
from pma.logging_setup import get_logger
from pma.rag.bm25 import BM25, tokenize
from pma.rag.embeddings import (
    Embedder,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    cosine,
)

log = get_logger("rag")


@dataclass(frozen=True)
class Chunk:
    source: str  # file name, e.g. claims_policy.md
    heading: str
    text: str

    @property
    def ref(self) -> str:
        return f"{self.source}#{self.heading}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def chunk_markdown(source: str, text: str) -> list[Chunk]:
    """Split on ``##`` headings; the document title is kept in each chunk for context."""
    title_match = re.match(r"#\s+(.+)", text)
    title = title_match.group(1).strip() if title_match else source
    chunks: list[Chunk] = []
    for part in re.split(r"(?m)^##\s+", text)[1:]:
        heading, _, body = part.partition("\n")
        body = body.strip()
        if len(body) < 40:
            continue
        chunks.append(Chunk(source, heading.strip(), f"{title} / {heading.strip()}\n{body}"))
    return chunks


def load_chunks(directory: Path | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    if directory is None:
        root = resources.files("pma") / "rag" / "guidelines"
        files = sorted((p for p in root.iterdir() if p.name.endswith(".md")), key=lambda p: p.name)
        for f in files:
            chunks += chunk_markdown(f.name, f.read_text(encoding="utf-8"))
    else:
        for f in sorted(directory.glob("*.md")):
            chunks += chunk_markdown(f.name, f.read_text(encoding="utf-8"))
    return chunks


class GuidelineRetriever:
    def __init__(
        self, chunks: list[Chunk], *, mode: str = "bm25", embedder: Embedder | None = None
    ) -> None:
        if not chunks:
            raise ValueError("no guideline chunks to index")
        self.chunks = chunks
        self.mode = mode
        self._bm25 = BM25([tokenize(c.text) for c in chunks])
        self._embedder = embedder
        self._vectors: list[list[float]] | None = None
        self.last_fallback = False

    def _bm25_hits(self, query: str, k: int) -> list[Hit]:
        return [Hit(self.chunks[i], s) for i, s in self._bm25.rank(query, k)]

    def _embedding_hits(self, query: str, k: int) -> list[Hit]:
        assert self._embedder is not None
        if self._vectors is None:
            self._vectors = self._embedder.embed([c.text for c in self.chunks])
        qv = self._embedder.embed([query])[0]
        scored = [(i, cosine(qv, v)) for i, v in enumerate(self._vectors)]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: (-s[1], s[0]))
        return [Hit(self.chunks[i], s) for i, s in scored[:k]]

    def search(self, query: str, k: int = 3) -> list[Hit]:
        self.last_fallback = False
        if self.mode == "bm25" or self._embedder is None:
            return self._bm25_hits(query, k)
        try:
            emb = self._embedding_hits(query, max(k, 8))
        except (PMAError, OSError) as exc:
            log.warning("embeddings failed, falling back to BM25", extra={"error": str(exc)})
            self.last_fallback = True
            return self._bm25_hits(query, k)
        if self.mode == "embeddings":
            return emb[:k]
        # hybrid: reciprocal rank fusion of the two rankings
        lexical = self._bm25_hits(query, max(k, 8))
        fused: dict[str, float] = {}
        by_ref: dict[str, Chunk] = {}
        for ranking in (lexical, emb):
            for rank, hit in enumerate(ranking):
                fused[hit.chunk.ref] = fused.get(hit.chunk.ref, 0.0) + 1.0 / (60 + rank + 1)
                by_ref[hit.chunk.ref] = hit.chunk
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(by_ref[ref], score) for ref, score in ordered]


def build_retriever(settings: Settings, directory: Path | None = None) -> GuidelineRetriever:
    chunks = load_chunks(directory)
    embedder: Embedder | None = None
    if settings.retriever in ("embeddings", "hybrid"):
        if settings.embeddings_backend == "openai_compatible":
            embedder = OpenAICompatibleEmbedder(
                base_url=settings.openai_base_url,
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
            )
        else:
            embedder = HashingEmbedder()
    return GuidelineRetriever(chunks, mode=settings.retriever, embedder=embedder)
