# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps (for packages that compile C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="netguard"
LABEL version="1.0.0-phase1"

# libpcap is required by Scapy at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap0.8 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Create non-root user for the API process
# Note: live packet capture REQUIRES root (or CAP_NET_RAW).
# The API process runs as non-root; capture runs as a sidecar with privileges.
RUN groupadd -r netguard && useradd -r -g netguard -s /sbin/nologin netguard

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data/raw data/processed logs \
    && chown -R netguard:netguard /app

USER netguard

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "4", \
    "--log-config", "/dev/null"]