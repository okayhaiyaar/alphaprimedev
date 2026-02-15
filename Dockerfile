# ============================================================
# ALPHA-PRIME v2.0 - Production Dockerfile
# ============================================================
# Multi-stage Python container optimized for:
# - Security (non-root user, minimal attack surface)
# - Size (<800MB including Playwright)
# - Layer caching (dependencies before code)
# - Production stability (health checks, proper logging)
# Build: docker build -t alpha-prime:v2 .
# Run:   docker run -p 8501:8501 --env-file .env alpha-prime:v2
# ============================================================


# ────────────────────────────────────────────────────────────
# STAGE 1: BASE RUNTIME IMAGE
# ────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS base

# Metadata labels
LABEL maintainer="your-email@example.com"
LABEL version="2.0"
LABEL description="ALPHA-PRIME AI Trading System"

# Python & pip environment (no .pyc, realtime logs, smaller images)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies:
# - Core utilities (curl, ca-certificates) for health checks & HTTPS
# - Libraries required by Playwright Chromium [web:51]
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and application directories
RUN groupadd -r alpha && useradd -r -g alpha -u 1000 alpha && \
    mkdir -p /app/data /app/logs /app/backups /app/data/cache && \
    chown -R alpha:alpha /app

WORKDIR /app


# ────────────────────────────────────────────────────────────
# STAGE 2: DEPENDENCY INSTALLATION (BUILDER)
# ────────────────────────────────────────────────────────────
FROM base AS builder

# Copy requirements first for better layer caching
COPY --chown=alpha:alpha requirements.txt .

# Upgrade pip tooling and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright and Chromium browser with system dependencies [web:51][web:55]
RUN playwright install --with-deps chromium


# ────────────────────────────────────────────────────────────
# STAGE 3: FINAL RUNTIME IMAGE
# ────────────────────────────────────────────────────────────
FROM base AS runtime

# Copy Python packages and Playwright data from builder
# (site-packages are already in the global env in slim image)
COPY --from=builder /usr/local /usr/local
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

# Re-assert environment in case downstream tooling modifies it
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Copy application code into image
COPY --chown=alpha:alpha . .

# Ensure optional shell scripts are executable (if present)
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Security: Run as non-root user 'alpha' (UID 1000)
# Secrets are loaded from environment variables at runtime, never baked into the image.
USER alpha

# Expose Streamlit dashboard port (primary UI)
EXPOSE 8501
# Optional secondary port (if you add a dedicated healthcheck or API later)
EXPOSE 8080

# Health check: use Streamlit's internal health endpoint [web:42][web:45]
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Declare volumes for persistent trading state and logs
VOLUME ["/app/data", "/app/logs", "/app/backups"]

# ────────────────────────────────────────────────────────────
# DEFAULT ENTRYPOINT (STREAMLIT DASHBOARD)
# ────────────────────────────────────────────────────────────
# Streamlit handles SIGTERM/SIGINT gracefully, allowing clean shutdown.
CMD ["streamlit", "run", "dashboard/app_v2.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]


# ============================================================
# BUILD & RUN INSTRUCTIONS
# ============================================================
# Build:
#   docker build -t alpha-prime:v2 .
#
# Run dashboard:
#   docker run -d \
#     --name alpha-prime \
#     -p 8501:8501 \
#     --env-file .env \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/logs:/app/logs \
#     -v $(pwd)/backups:/app/backups \
#     alpha-prime:v2
#
# Run scheduler instead of dashboard:
#   docker run -d \
#     --name alpha-scheduler \
#     --env-file .env \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/logs:/app/logs \
#     alpha-prime:v2 \
#     python scheduler.py
#
# Health check (manual):
#   docker exec alpha-prime curl http://localhost:8501/_stcore/health
#
# View logs:
#   docker logs -f alpha-prime
#
# Shell access (debug):
#   docker exec -it alpha-prime /bin/bash
# ============================================================
