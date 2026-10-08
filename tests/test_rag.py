from __future__ import annotations

import httpx
import pytest

from pma.config import Settings
from pma.errors import ProviderError
from pma.rag.bm25 import BM25, tokenize
from pma.rag.embeddings import HashingEmbedder, OpenAICompatibleEmbedder, cosine
from pma.rag.retriever import (
    GuidelineRetriever,
    build_retriever,
    chunk_markdown,
    load_chunks,
)


def test_tokenize_drops_stopwords_and_folds_plurals():
    assert tokenize("The claims of the Teams") == ["claim", "team"]
    assert tokenize("De garantie voor een laptop") == ["garantie", "laptop"]


def test_bm25_ranks_the_relevant_document_first_and_is_deterministic():
    docs = [tokenize(t) for t in ["red apples", "green pears and pears", "blue sky"]]
    bm = BM25(docs)
    assert [i for i, _ in bm.rank("pears", 3)] == [1]
    assert bm.rank("apples pears", 3)[0][0] in (0, 1)
    assert bm.rank("apples pears", 3) == bm.rank("apples pears", 3)
    assert bm.rank("unknownterm", 3) == []


def test_chunk_markdown_splits_on_h2_and_keeps_title():
    md = (
        "# Doc Title\n\nintro\n\n## One\n"
        + "alpha " * 20
        + "\n\n## Two\nshort\n\n## Three\n"
        + "gamma " * 20
    )
    chunks = chunk_markdown("d.md", md)
    assert [c.heading for c in chunks] == ["One", "Three"]  # "Two" is too short to index
    assert chunks[0].text.startswith("Doc Title / One")
    assert chunks[0].ref == "d.md#One"


def test_guideline_files_load_and_cover_the_expected_topics():
    chunks = load_chunks()
    assert len(chunks) >= 20
    assert {c.source for c in chunks} == {
        "brand_voice.md",
        "channel_formats.md",
        "claims_policy.md",
        "dutch_style_guide.md",
        "legal_compliance.md",
        "product_category_notes.md",
    }


@pytest.mark.parametrize(
    ("query", "expected_source"),
    [
        ("forbidden claims guaranteed superlatives", "claims_policy.md"),
        ("email subject line length limit", "channel_formats.md"),
        ("aanspreekvorm u je Nederlandse stijl", "dutch_style_guide.md"),
        ("monitors USB-C docking ergonomics", "product_category_notes.md"),
        ("comparative advertising GDPR consent", "legal_compliance.md"),
    ],
)
def test_bm25_retrieval_finds_the_right_guideline(retriever, query, expected_source):
    top = retriever.search(query, 3)
    assert expected_source in {h.chunk.source for h in top}, [h.chunk.ref for h in top]


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(64)
    a, b = emb.embed(["price excl vat", "price excl vat"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, emb.embed(["completely unrelated gardening"])[0]) < 0.5


@pytest.mark.parametrize("mode", ["embeddings", "hybrid"])
def test_embedding_and_hybrid_modes_return_results(mode):
    r = build_retriever(Settings.from_env({"RETRIEVER": mode}))
    hits = r.search("forbidden claims guaranteed", 3)
    assert 1 <= len(hits) <= 3
    assert "claims_policy.md" in {
        h.chunk.source for h in r.search("forbidden claims guaranteed", 5)
    }


def test_embeddings_failure_falls_back_to_bm25():
    class Broken:
        def embed(self, texts):
            raise ProviderError("embedding service down")

    r = GuidelineRetriever(load_chunks(), mode="hybrid", embedder=Broken())
    hits = r.search("forbidden claims guaranteed", 2)
    assert hits and r.last_fallback is True


def test_openai_compatible_embedder_parses_response_in_index_order():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    emb = OpenAICompatibleEmbedder(
        base_url="http://x/v1",
        model="m",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert emb.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, json={"nope": 1}),
        httpx.Response(200, json={"data": []}),
    ],
)
def test_openai_compatible_embedder_errors(response):
    emb = OpenAICompatibleEmbedder(
        base_url="http://x/v1",
        model="m",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: response)),
    )
    with pytest.raises(ProviderError):
        emb.embed(["a"])
