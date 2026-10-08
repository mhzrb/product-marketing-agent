"""Optional embeddings retrieval, enabled with ``RETRIEVER=embeddings`` or ``hybrid``.

Two embedders:

* ``HashingEmbedder`` - deterministic, offline, no model. It is a *lexical* feature-hashing
  vector (unigrams + bigrams), so it is NOT semantic. It exists so the embeddings code path
  (vector store, cosine search, hybrid fusion) is testable without a model or network.
* ``OpenAICompatibleEmbedder`` - calls ``POST {base_url}/embeddings``. NOT VERIFIED against a real
  server here; unit-tested with ``httpx.MockTransport`` only.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx

from pma.errors import ProviderError
from pma.rag.bm25 import tokenize


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        vec = [0.0] * self.dim
        for feature in features:
            digest = hashlib.md5(feature.encode("utf-8"), usedforsecurity=False).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._client = client or httpx.Client()

    def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = self._client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": texts},
                timeout=self._timeout_s,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"embeddings request failed: {exc.__class__.__name__}") from exc
        if response.status_code >= 400:
            raise ProviderError(f"embeddings request rejected with status {response.status_code}")
        try:
            rows = sorted(response.json()["data"], key=lambda r: r.get("index", 0))
            vectors = [[float(x) for x in row["embedding"]] for row in rows]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError("malformed embeddings response") from exc
        if len(vectors) != len(texts):
            raise ProviderError("embeddings response length mismatch")
        return vectors


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
