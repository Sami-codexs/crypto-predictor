# =============================================================================
# CryptoMind Dockerfile
# Multi-stage build for production ML + Streamlit + FastAPI
# =============================================================================

# ─── Stage 1: Builder ───
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies (for numpy, scipy, tensorflow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ─── Stage 2: Runtime ───
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Make venv the default Python
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    # Environment defaults (override via docker-compose)
    ENV=production \
    DB_PATH=/app/data/crypto.db \
    MODELS_DIR=/app/models \
    LOGS_DIR=/app/logs \
    DATA_DIR=/app/data \
    API_PORT=8000 \
    DASHBOARD_PORT=8501 \
    OLLAMA_HOST=http://ollama:11434 \
    OLLAMA_MODEL=llama3.1 \
    FETCH_INTERVAL_MINUTES=60 \
    LLM_CACHE_TTL_HOURS=1

# Create required directories
RUN mkdir -p data models logs tests

# Copy application code
COPY src/ ./src/
COPY dashboard.py .
COPY train.py .
COPY seed_demo_data.py .
COPY tests/ ./tests/

# ─── NO .env COPY (security fix) ───
# All configuration is injected via environment variables at runtime
# via docker-compose or Render.com dashboard

# Expose ports for API and Streamlit
EXPOSE 8000 8501

# Health check using stdlib (no external dependency)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run API server
# Override via docker-compose for dashboard/scheduler
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]