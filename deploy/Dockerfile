# Multi-stage build so the runtime image ships no compilers/build tooling -
# smaller attack surface, which matters when this lands in a hardened or
# air-gapped enclave.
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/deps -r requirements.txt

# Pre-warm the local embedding model cache at BUILD time (network allowed
# here) so the runtime container never needs egress to Hugging Face Hub -
# it loads BAAI/bge-small-en-v1.5 straight from this baked-in cache dir.
FROM builder AS model-cache
ENV PYTHONPATH=/deps
ENV FASTEMBED_CACHE_PATH=/model-cache
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

FROM python:3.12-slim
RUN groupadd -r mission && useradd -r -g mission -d /app mission
WORKDIR /app
COPY --from=builder /deps /usr/local/lib/python3.12/site-packages
COPY --from=model-cache /model-cache /app/.cache/fastembed
COPY app ./app
COPY data ./data
COPY frontend ./frontend
COPY main.py .

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

USER mission
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/stats')" || exit 1

CMD ["python", "main.py"]
