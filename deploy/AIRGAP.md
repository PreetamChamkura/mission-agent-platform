# Air-Gapped / Hardened Deployment Notes

This platform is built so it never needs to leave the enclave at runtime:

- **No external model or embedding API calls.** `app/orchestration/llm.py` and
  `app/rag/index.py` are self-contained (deterministic mock reasoning + TF-IDF
  retrieval). Swap the `LLMClient` implementation for an on-prem/enclave model
  server when one is available; the interface boundary is already there.
- **No external database.** SQLite (`app/db.py`) is bundled with the Python
  standard library; the audit trail persists to a local file, no network DB.
- **No CDN dependencies in the frontend.** `frontend/index.html` is a single
  file with inline CSS/JS - nothing fetched from a public CDN.

## Building offline

1. On a connected build host, vendor the wheels:
   ```bash
   pip download -r requirements.txt -d deploy/wheelhouse --platform manylinux2014_x86_64 --python-version 3.12 --only-binary=:all:
   ```
2. Transfer `deploy/wheelhouse` and the repo into the enclave via your
   authorized media transfer process.
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
