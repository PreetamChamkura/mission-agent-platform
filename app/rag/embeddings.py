"""Local embedding model for the RAG pipeline.

Uses `fastembed` (ONNX Runtime, no torch/GPU required) to run
`BAAI/bge-small-en-v1.5` (384-dim) entirely in-process - no call to a hosted
embeddings API. The model's ONNX weights are fetched once from Hugging Face
Hub on first use and cached under `~/.cache/fastembed`; for an air-gapped
deployment, pre-download that cache on a connected build host and ship it
alongside the image (see deploy/AIRGAP.md).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    # explicit, not just relying on fastembed's implicit env lookup - this is what
    # points the runtime at the cache baked into the image by deploy/Dockerfile
    cache_dir = os.environ.get("FASTEMBED_CACHE_PATH")
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=cache_dir)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns an (N, EMBEDDING_DIM) float32 array of L2-normalized embeddings."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")
    vectors = np.array(list(_model().embed(texts)), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vectors / norms


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
