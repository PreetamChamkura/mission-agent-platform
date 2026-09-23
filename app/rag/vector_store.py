"""Local vector store: a FAISS `IndexFlatIP` (exact inner-product search over
L2-normalized vectors, i.e. cosine similarity) held in process memory.

FAISS runs entirely on-box with no external service, which is what makes it
usable in the air-gapped deployment target described in deploy/AIRGAP.md -
same reasoning as the local embedding model in embeddings.py. `IndexFlatIP`
is an exact (brute-force) index, the right choice at this corpus size; for a
much larger corpus this is the point to swap in `IndexIVFFlat` or `IndexHNSW`
without changing the calling code below.
"""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class VectorHit:
    index: int
    score: float


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._size = 0

    def build(self, vectors: np.ndarray) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        if vectors.shape[0]:
            self.index.add(vectors)
        self._size = vectors.shape[0]

    def search(self, query_vector: np.ndarray, top_k: int) -> list[VectorHit]:
        if self._size == 0:
            return []
        k = min(top_k, self._size)
        scores, indices = self.index.search(query_vector.reshape(1, -1), k)
        return [
            VectorHit(index=int(idx), score=float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]

    def __len__(self) -> int:
        return self._size
