# Build stage: install the full package into a venv.
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY . .
RUN pip install .

# Runtime stage: non-root user + entrypoint (daemon + web dashboard).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    BTCUSDT_WEB_HOST=0.0.0.0 \
    BTCUSDT_WEB_PORT=8000 \
    BTCUSDT_DB_PATH=/app/data/btcusdt_signals.db \
    BTCUSDT_SECRETS_FILE=/app/data/btcusdt_secrets.json

COPY --from=builder /opt/venv /opt/venv

# The named volume is mounted on /app/data; pre-create it owned by the runtime
# user so the non-root process can write the SQLite ledger and secrets file.
RUN useradd --create-home appuser \
 && install -d -m 0775 -o appuser -g appuser /app/data

COPY --from=builder /build .
COPY docker/entrypoint-signalengine.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER appuser
WORKDIR /app

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

ENTRYPOINT ["/entrypoint.sh"]