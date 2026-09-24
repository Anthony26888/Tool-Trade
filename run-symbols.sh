#!/usr/bin/env bash
# Multi-symbol supervisor entry point (plan B', phase B3).
#
# Reads the enabled combo from Settings (shared DB) and converges the process
# set: one daemon per enabled symbol + one web dashboard. Never starts a
# duplicate, never stops a symbol holding an OPEN position.
#
# Usage:
#   ./run-symbols.sh sync [extra args]    converge to Settings (default)
#   ./run-symbols.sh stop [extra args]    stop all (OPEN positions block)
#   ./run-symbols.sh status [extra args]  JSON process + position table
#
# Environment:
#   PYTHON_BIN   python with the project deps (default: current venv/active python3)
#   BTCUSDT_DB_PATH shared SQLite path (default data/btcusdt_signals.db)
#   WEB_PORT     dashboard port (default 8000)
#   LOG_DIR      pidfiles + logs (default logs/)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export BTCUSDT_DB_PATH="${BTCUSDT_DB_PATH:-data/btcusdt_signals.db}"
WEB_PORT="${WEB_PORT:-8000}"
LOG_DIR="${LOG_DIR:-logs}"

cd "$ROOT"
mkdir -p "$LOG_DIR"
if [ ! -f "$BTCUSDT_DB_PATH" ]; then
  echo "shared DB not found: $BTCUSDT_DB_PATH" >&2
  echo "start the BTCUSDT stack once first so the database (and Settings) exist." >&2
  exit 2
fi
exec "$PYTHON_BIN" -m signal_engine.supervisor "${1:-sync}" \
  --db "$BTCUSDT_DB_PATH" --rundir "$LOG_DIR" --web-port "$WEB_PORT" \
  "${@:2}"
