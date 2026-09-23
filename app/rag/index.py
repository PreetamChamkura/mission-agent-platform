"""Retrieval index for the RAG pipeline: hybrid dense + sparse search.

  - Dense: `BAAI/bge-small-en-v1.5` embeddings (app/rag/embeddings.py) held in
    a FAISS `IndexFlatIP` (app/rag/vector_store.py) - catches semantic/
    paraphrase matches a keyword index misses.
  - Sparse: TF-IDF + cosine similarity, computed in pure Python - catches
    exact identifiers and rare terms (permit IDs, case numbers) that a dense
    embedding can wash out.

Scores from each are min-max normalized over the filtered candidate set,
then combined with a weighted sum (dense-heavy by default). Both retrievers
run entirely in-process - no hosted embeddings API and no external vector
database - so the pipeline works unmodified in an air-gapped deployment
(see deploy/AIRGAP.md).

Metadata filters (source system, record type, classification) are applied
before scoring, so retrieval never surfaces a document outside an agent's
clearance or mission scope regardless of how well it scores.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from app.rag.embeddings import embed_query, embed_texts
from app.rag.ingest import Document
from app.rag.vector_store import FaissVectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredDoc:
    doc: Document
    score: float
    dense_score: float
    sparse_score: float


def _normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {i: 0.0 for i in scores}
    return {i: (s - lo) / (hi - lo) for i, s in scores.items()}


class Indexer:
    def __init__(self, dense_weight: float = 0.6, sparse_weight: float = 0.4):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.docs: list[Document] = []
        self.vectors: list[dict[str, float]] = []  # TF-IDF sparse vectors
        self.vocab_idf: dict[str, float] = {}
        self.doc_embeddings: np.ndarray = np.zeros((0, 1), dtype="float32")
        self.vector_store: FaissVectorStore | None = None

    def build(self, docs: list[Document]) -> None:
        self.docs = docs
        self._build_sparse(docs)
        self._build_dense(docs)

    def _build_sparse(self, docs: list[Document]) -> None:
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

    def _build_dense(self, docs: list[Document]) -> None:
        texts = [f"{d.title}. {d.text}" for d in docs]
        self.doc_embeddings = embed_texts(texts) if texts else np.zeros((0, 1), dtype="float32")
        dim = self.doc_embeddings.shape[1] if self.doc_embeddings.size else 1
        self.vector_store = FaissVectorStore(dim)
        self.vector_store.build(self.doc_embeddings)

    def _vectorize_query_sparse(self, text: str) -> dict[str, float]:
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
        if not self.docs:
            return []

        levels = classification_levels or {"PUBLIC": 0, "CUI": 1, "CONFIDENTIAL": 2, "SECRET": 3, "TOP SECRET": 4}
        candidate_idx = [
            i for i, d in enumerate(self.docs)
            if (not source_systems or d.source_system in source_systems)
            and (not record_types or d.record_type in record_types)
            and (max_classification_level is None or levels.get(d.classification, 0) <= max_classification_level)
        ]
        if not candidate_idx:
            return []

        # sparse (TF-IDF cosine) scores over candidates
        qvec_sparse = self._vectorize_query_sparse(query)
        sparse_scores = {i: self._cosine(qvec_sparse, self.vectors[i]) for i in candidate_idx}

        # dense (embedding cosine, via normalized dot product) scores over candidates
        qvec_dense = embed_query(query)
        dense_scores = {i: float(np.dot(self.doc_embeddings[i], qvec_dense)) for i in candidate_idx}

        norm_sparse = _normalize(sparse_scores)
        norm_dense = _normalize(dense_scores)

        combined = []
        for i in candidate_idx:
            score = self.dense_weight * norm_dense.get(i, 0.0) + self.sparse_weight * norm_sparse.get(i, 0.0)
            if score > 0 or sparse_scores[i] > 0 or dense_scores[i] > 0:
                combined.append(ScoredDoc(
                    doc=self.docs[i], score=score,
                    dense_score=dense_scores[i], sparse_score=sparse_scores[i],
                ))

        combined.sort(key=lambda sd: sd.score, reverse=True)
        return combined[:top_k]

    def stats(self) -> dict:
        by_source = Counter(d.source_system for d in self.docs)
        by_type = Counter(d.record_type for d in self.docs)
        return {
            "total_documents": len(self.docs),
            "vocabulary_size": len(self.vocab_idf),
            "embedding_dim": self.doc_embeddings.shape[1] if self.doc_embeddings.size else 0,
            "retrieval": "hybrid dense (FAISS/bge-small-en-v1.5) + sparse (TF-IDF)",
            "by_source_system": dict(by_source),
            "by_record_type": dict(by_type),
        }
