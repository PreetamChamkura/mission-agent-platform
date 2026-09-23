# Multi-stage build so the runtime image ships no compilers/build tooling -
# smaller attack surface, which matters when this lands in a hardened or
# air-gapped enclave.
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/deps -r requirements.txt

FROM python:3.12-slim
RUN groupadd -r mission && useradd -r -g mission -d /app mission
WORKDIR /app
COPY --from=builder /deps /usr/local/lib/python3.12/site-packages
COPY app ./app
COPY data ./data
COPY frontend ./frontend
COPY main.py .

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

USER mission
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/stats')" || exit 1

CMD ["python", "main.py"]
