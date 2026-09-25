#!/bin/sh
# BTCUSDT Signal Engine container entrypoint.
# Runs BOTH long-running processes in one container:
#   - the web dashboard (HTTP + Settings, background)
#   - the daemon (AI 1H analysis + TP/SL 1m monitor + demo ledger, foreground
#     for the container lifecycle).
#
# On SIGTERM (docker stop) both processes are stopped cleanly; the container
# exit status follows the daemon so the compose restart policy handles a crash.
set -eu

DATA_DIR="$(dirname "${BTCUSDT_DB_PATH:-/app/data/btcusdt_signals.db}")"
mkdir -p "$DATA_DIR"

HOST="${BTCUSDT_WEB_HOST:-0.0.0.0}"
PORT="${BTCUSDT_WEB_PORT:-8000}"

# DAEMON_ONLY=1 (extra per-symbol services in docker-compose.yml): skip the
# web dashboard and run just the daemon. The main service keeps the default
# (web + daemon) so there is exactly one dashboard per host.
if [ "${DAEMON_ONLY:-0}" = "1" ]; then
    echo "[signalengine] daemon-only (BTCUSDT_SYMBOL=${BTCUSDT_SYMBOL:-BTCUSDT}) -> AI(1H) analysis + TP/SL(1m) monitor"
    exec python -m signal_engine
fi

echo "[signalengine] web dashboard -> http://${HOST}:${PORT}"
python -m signal_engine web --host "$HOST" --port "$PORT" &
WEB_PID=$!

echo "[signalengine] daemon -> AI(1H) analysis + TP/SL(1m) monitor + demo ledger"
python -m signal_engine &
DAEMON_PID=$!

_shutdown() {
    echo "[signalengine] shutting down (web=$WEB_PID daemon=$DAEMON_PID)"
    kill "$WEB_PID" "$DAEMON_PID" 2>/dev/null || true
    exit $((128 + 15))
}
trap _shutdown INT TERM

# The container lifecycle follows the daemon (the trading core). When it exits
# the web process is stopped and the restart policy decides what happens next.
wait "$DAEMON_PID"
STATUS=$?
kill "$WEB_PID" 2>/dev/null || true
exit "$STATUS"