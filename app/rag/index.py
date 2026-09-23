"""Lightweight retrieval index for the RAG pipeline.

Deliberately dependency-light (pure Python + numpy, no external embedding API):
this is what makes it viable in an air-gapped deployment where the platform
cannot reach an external embeddings endpoint. It implements TF-IDF vectors +
cosine similarity, plus metadata filters (source system, classification,
record type) so retrieval stays inside an agent's clearance and mission scope.

Swap `Indexer.vectorize` for a real embedding model when running connected.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from app.rag.ingest import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredDoc:
    doc: Document
    score: float


class Indexer:
    def __init__(self):
        self.docs: list[Document] = []
        self.doc_freq: Counter = Counter()
        self.vectors: list[dict[str, float]] = []
        self.vocab_idf: dict[str, float] = {}

    def build(self, docs: list[Document]) -> None:
        self.docs = docs
        term_doc_sets: dict[str, set[int]] = defaultdict(set)
        token_lists = []
        for i, d in enumerate(docs):
            toks = tokenize(f"{d.title} {d.text}")
            token_lists.append(toks)
            for t in set(toks):
                term_doc_sets[t].add(i)

        n = max(len(docs), 1)
        self.vocab_idf = {t: math.log((n + 1) / (len(ids) + 1)) + 1 for t, ids in term_doc_sets.items()}

        self.vectors = []
        for toks in token_lists:
            tf = Counter(toks)
            length = max(len(toks), 1)
            vec = {t: (c / length) * self.vocab_idf.get(t, 0.0) for t, c in tf.items()}
            self.vectors.append(vec)

    def _vectorize_query(self, text: str) -> dict[str, float]:
        toks = tokenize(text)
        tf = Counter(toks)
        length = max(len(toks), 1)
        return {t: (c / length) * self.vocab_idf.get(t, 0.0) for t, c in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return num / (na * nb)

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_systems: list[str] | None = None,
        record_types: list[str] | None = None,
        max_classification_level: int | None = None,
        classification_levels: dict[str, int] | None = None,
    ) -> list[ScoredDoc]:
        qvec = self._vectorize_query(query)
        levels = classification_levels or {"PUBLIC": 0, "CUI": 1, "CONFIDENTIAL": 2, "SECRET": 3, "TOP SECRET": 4}
        scored = []
        for d, vec in zip(self.docs, self.vectors):
            if source_systems and d.source_system not in source_systems:
                continue
            if record_types and d.record_type not in record_types:
                continue
            if max_classification_level is not None and levels.get(d.classification, 0) > max_classification_level:
                continue
            s = self._cosine(qvec, vec)
            if s > 0:
                scored.append(ScoredDoc(doc=d, score=s))
        scored.sort(key=lambda sd: sd.score, reverse=True)
        return scored[:top_k]

    def stats(self) -> dict:
        by_source = Counter(d.source_system for d in self.docs)
        by_type = Counter(d.record_type for d in self.docs)
        return {
            "total_documents": len(self.docs),
            "vocabulary_size": len(self.vocab_idf),
            "by_source_system": dict(by_source),
            "by_record_type": dict(by_type),
        }
