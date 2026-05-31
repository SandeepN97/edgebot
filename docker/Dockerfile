# EdgeBot Docker image — Python 3.11 slim with non-root user
#
# Build: docker build -f docker/Dockerfile -t edgebot .
# Run:   docker run --env-file .env edgebot

FROM python:3.11-slim AS base

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        libssl-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN groupadd --gid 1001 edgebot \
    && useradd --uid 1001 --gid edgebot --shell /bin/bash --create-home edgebot

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY --chown=edgebot:edgebot . .

# ── Data directory ────────────────────────────────────────────────────────────
RUN mkdir -p /app/data /app/logs \
    && chown -R edgebot:edgebot /app/data /app/logs

# ── Drop privileges ───────────────────────────────────────────────────────────
USER edgebot

# ── Environment defaults ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD python -c "import domain; print('ok')"

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["python", "-m", "edgebot.main"]