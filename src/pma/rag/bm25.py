"""Okapi BM25 implemented from scratch (no dependency), plus the shared tokenizer."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"\w+", re.UNICODE)

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "your",
        "our",
        "we",
        "you",
        "they",
        "their",
        "them",
        "these",
        "those",
        "but",
        "not",
        "no",
        "do",
        "does",
        "can",
        "may",
        "must",
        "should",
        "de",
        "het",
        "een",
        "en",
        "van",
        "voor",
        "met",
        "op",
        "in",
        "is",
        "zijn",
        "dat",
        "die",
        "dit",
        "deze",
        "te",
        "niet",
        "of",
        "als",
        "aan",
        "bij",
        "uit",
        "naar",
        "om",
        "er",
        "ook",
        "maar",
        "dan",
        "wordt",
        "worden",
        "kan",
        "kunnen",
        "u",
        "uw",
        "je",
        "jouw",
        "we",
        "wij",
        "ons",
        "onze",
    ]
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in _STOPWORDS or len(raw) < 2:
            continue
        # crude plural folding so "claims" matches "claim" (EN) - good enough for guidelines
        if len(raw) > 4 and raw.endswith("s") and not raw.endswith("ss"):
            raw = raw[:-1]
        tokens.append(raw)
    return tokens


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = documents
        self.n = len(documents)
        self.doc_len = [len(d) for d in documents]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in documents]
        df: Counter[str] = Counter()
        for d in documents:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_tokens: list[str], index: int) -> float:
        score = 0.0
        tf = self.tf[index]
        length = self.doc_len[index]
        for term in query_tokens:
            freq = tf.get(term, 0)
            if not freq:
                continue
            denom = freq + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1.0))
            score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
        return score

    def rank(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        q = tokenize(query)
        scored = [(i, self.score(q, i)) for i in range(self.n)]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: (-s[1], s[0]))  # deterministic tie-break
        return scored[:k]
