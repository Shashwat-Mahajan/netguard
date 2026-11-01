# NetGuard API image. Models and secrets are NOT in the image:
# data/processed is mounted at runtime, settings come from environment variables.
FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpcap-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap0.8 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
WORKDIR /app
RUN groupadd -r netguard && useradd -r -g netguard -s /sbin/nologin netguard
COPY api/ api/
COPY capture/ capture/
COPY config/ config/
COPY models/ models/
COPY processor/ processor/
COPY streaming/ streaming/
RUN mkdir -p data/raw data/processed logs \
    && chown -R netguard:netguard /app
USER netguard
EXPOSE 8000
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]