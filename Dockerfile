# Multi-stage / Production Dockerfile for ExamRAG API
FROM python:3.11-slim as base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install security updates and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root application user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser evaluation/ ./evaluation/

# Switch to non-root user
USER appuser

# Expose default port
EXPOSE 8000

# Healthcheck against liveness probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health/live || exit 1

# Production server entrypoint binding to 0.0.0.0 with dynamic $PORT support (for Render)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

