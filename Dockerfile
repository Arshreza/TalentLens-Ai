# TalentLens AI — Production Dockerfile
# Multi-stage build: lean Python 3.11 runtime image

FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies needed by pdfplumber / python-docx
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies ──────────────────────────────────────────────────────────
FROM base AS deps
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Final Image ───────────────────────────────────────────────────────────
FROM deps AS runtime
COPY . .

# Download SBERT model at build time so first request is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
