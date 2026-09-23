# Air-Gapped / Hardened Deployment Notes

This platform is built so it never needs to leave the enclave at runtime:

- **No external LLM or embeddings API calls at runtime.** `app/orchestration/llm.py`
  is a deterministic mock reasoning client (swap `LLMClient` for an on-prem/
  enclave model server when one is available - the interface boundary is
  already there). Retrieval (`app/rag/index.py`) runs a local `bge-small-en-v1.5`
  embedding model via `fastembed`/ONNX Runtime (`app/rag/embeddings.py`) into
  an in-process FAISS index (`app/rag/vector_store.py`), plus a TF-IDF sparse
  index computed in pure Python - both run entirely on-box, no hosted API call.
- **No external database.** SQLite (`app/db.py`) is bundled with the Python
  standard library; the audit trail persists to a local file, no network DB.
- **No CDN dependencies in the frontend.** `frontend/index.html` is a single
  file with inline CSS/JS - nothing fetched from a public CDN.

## Building offline

The embedding model's ONNX weights (`BAAI/bge-small-en-v1.5`, ~130MB) are
fetched from Hugging Face Hub the first time `fastembed.TextEmbedding` runs
and cached under `~/.cache/fastembed`. That download has to happen on a
connected host before the enclave transfer - there is no runtime fallback.

1. On a connected build host, vendor the Python wheels and pre-warm the
   embedding model cache:
   ```bash
   pip download -r requirements.txt -d deploy/wheelhouse --platform manylinux2014_x86_64 --python-version 3.12 --only-binary=:all:
   python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"
   ```
2. Transfer `deploy/wheelhouse`, the repo, and the populated
   `~/.cache/fastembed` directory into the enclave via your authorized media
   transfer process. Mount or copy the cache to the same path (or set
   `FASTEMBED_CACHE_PATH`) inside the container so no download is attempted
   at startup.
3. Inside the enclave:
   ```bash
   pip install --no-index --find-links deploy/wheelhouse -r requirements.txt
   docker build -f deploy/Dockerfile -t mission-agent-platform:local ..
   ```

## Hardening checklist

- [x] Runs as non-root (`USER mission` in the Dockerfile, `runAsNonRoot` in k8s)
- [x] Read-only root filesystem, all Linux capabilities dropped
- [x] `NetworkPolicy` default-denies egress except DNS (see `k8s/networkpolicy.yaml`)
- [x] No secrets or credentials embedded in the image
- [x] Guardrail policy enforces per-agent data classification clearance
  (`app/guardrails/policy.py::classification_check`) so a compromised agent
  role cannot read above its clearance even if RAG filters are bypassed upstream
- [ ] Wire the audit DB to your environment's approved long-term log sink
      (this repo ships a local SQLite file as the minimal viable audit store)
- [ ] Replace `MockLLM` with your enclave-approved model endpoint
