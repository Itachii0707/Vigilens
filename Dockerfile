# VEY RAXIS SENTINEL - Production Container Image
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SENTINEL_DEVICE=auto

# Install essential system runtime libraries (OpenCV dependencies, libgl, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install project dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code and default configurations
COPY src/ ./src/
COPY configs/ ./configs/
COPY api/ ./api/
COPY scripts/ ./scripts/

# Install the veyraxis package in editable mode
RUN pip install -e .

# Create non-privileged service user for runtime security
RUN useradd -m -u 1000 sentinel && \
    mkdir -p /app/runs /app/outputs /app/data && \
    chown -R sentinel:sentinel /app

USER sentinel

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default launch command: production Uvicorn ASGI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
