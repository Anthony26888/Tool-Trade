#!/usr/bin/env python3
"""Phase 14/15 command-line interface for the BTCUSDT Signal Engine.

Run the 24/7 daemon from the repository root::

    python -m signal_engine

or, for cron/CI, a single analysis+monitor pass::

    python -m signal_engine --once

Query the SQLite state (works while the daemon runs)::

    python -m signal_engine status
    python -m signal_engine signals
    python -m signal_engine active
    python -m signal_engine demo
    python -m signal_engine health          # exit 0 only when daemon is alive

Serve the Phase 15 Web Dashboard + Settings UI::

    python -m signal_engine web --host 127.0.0.1 --port 8000

Add ``--json`` for machine-readable output. Every number is rendered as its
exact ``Decimal`` string; nothing here ever sends a Binance order.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from dataclasses import replace
from decimal import Decimal
from typing import Any

from database.database import Database, DemoRepository, SignalRepository
from demo.position import DemoAccountRecord, DemoTrade
from demo.statistics import demo_statistics

from .runtime import (
    Runtime,
    RuntimeConfig,
    _jsonable,
    render_active,
    render_demo,
    render_health,
    render_signals,
    render_status,
    runtime_config_from_env,
)

logger = logging.getLogger(__name__)

QUERY_COMMANDS = {"status", "signals", "active", "demo", "health"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal_engine",
        description="BTCUSDT futures AI signal + DEMO trading system (research only).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "status", "signals", "active", "demo", "health", "web"],
        help="run starts the 24/7 daemon (default); the others query SQLite.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run exactly one analysis + monitor pass and exit (with 'run').",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Web dashboard bind host (default 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Web dashboard bind port (default 8000).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="machine-readable JSON output for query commands.",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="overrides BTCUSDT_DB_PATH for this invocation.",
    )
    return parser


def _config(argv_namespace: argparse.Namespace) -> RuntimeConfig:
    config = runtime_config_from_env()
    if argv_namespace.db:
        config = replace(config, db_path=argv_namespace.db)
    return config


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, DemoAccountRecord):
        return jsonable_account(value)
    if isinstance(value, DemoTrade):
        return jsonable_trade(value)
    return _jsonable(value)


def jsonable_signal(signal) -> dict[str, Any]:
    return {
        "id": signal.id,
        "symbol": signal.symbol,
        "timeframe": signal.timeframe,
        "direction": signal.direction,
        "status": signal.status,
        "entry": _jsonable(signal.entry),
        "stop_loss": _jsonable(signal.stop_loss),
        "take_profit": _jsonable(signal.take_profit),
        "created_at": signal.created_at,
        "opened_at": signal.opened_at,
        "closed_at": signal.closed_at,
        "close_price": _jsonable(signal.close_price),
        "close_reason": signal.close_reason,
        "confidence": signal.confidence,
        "model_name": signal.model_name,
    }


def jsonable_account(account: DemoAccountRecord) -> dict[str, Any]:
    return {
        "id": account.id,
        "name": account.name,
        "initial_balance": _jsonable(account.initial_balance),
        "balance": _jsonable(account.balance),
        "equity": _jsonable(account.equity),
        "peak_equity": _jsonable(account.peak_equity),
        "margin_per_trade": _jsonable(account.margin_per_trade),
        "leverage": account.leverage,
        "risk_percent": _jsonable(account.risk_percent),
        "fee_rate": _jsonable(account.fee_rate),
    }


def jsonable_trade(trade: DemoTrade) -> dict[str, Any]:
    return {
        "id": trade.id,
        "signal_id": trade.signal_id,
        "side": trade.side,
        "entry_price": _jsonable(trade.entry_price),
        "exit_price": _jsonable(trade.exit_price),
        "quantity": _jsonable(trade.quantity),
        "position_size": _jsonable(trade.position_size),
        "gross_pnl": _jsonable(trade.gross_pnl),
        "fee": _jsonable(trade.fee),
        "net_pnl": _jsonable(trade.net_pnl),
        "result": trade.result,
        "closed_at": trade.closed_at,
    }


# -- Daemon actions --------------------------------------------------------------


def _run_daemon(config: RuntimeConfig, once: bool) -> int:
    runtime = Runtime(config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if once:
        summary = runtime.run_once()
        for key in sorted(summary):
            print(f"{key}: {summary[key]}")
        return 0

    def _shutdown(signum: int, _frame: Any) -> None:
        logger.info("[CLI] received signal %s; shutting down gracefully", signum)
        runtime.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    runtime.run_forever()
    return 0


# -- Query actions ---------------------------------------------------------------


def _open_db(config: RuntimeConfig):
    database = Database(config.db_path)
    database.initialize()
    return database


def _active_signal(database) -> Any | None:
    return SignalRepository(database).get_active_signal()


def _cmd_status(config: RuntimeConfig, as_json: bool) -> int:
    database = _open_db(config)
    # health reads only runtime_state, so it never creates the demo account.
    runtime = Runtime(config, database=database)
    health = runtime.health()
    active = _active_signal(database)
    if as_json:
        payload = {**_health_json(health), "active": jsonable_signal(active) if active else None}
        print(json.dumps(payload, indent=2, default=_json_default))
        return 0
    print(render_status(health, active, db_path=config.db_path))
    return 0


def _cmd_signals(config: RuntimeConfig, as_json: bool) -> int:
    database = _open_db(config)
    signals = SignalRepository(database).list_signals(limit=50)
    if as_json:
        print(json.dumps([jsonable_signal(s) for s in signals], indent=2, default=_json_default))
        return 0
    print(render_signals(signals))
    return 0


def _cmd_active(config: RuntimeConfig, as_json: bool) -> int:
    database = _open_db(config)
    active = _active_signal(database)
    if as_json:
        print(json.dumps({"active": jsonable_signal(active) if active else None}, indent=2, default=_json_default))
        return 0
    print(render_active(active))
    return 0


def _cmd_demo(config: RuntimeConfig, as_json: bool) -> int:
    database = _open_db(config)
    repo = DemoRepository(database)
    account_row = repo.get_account("demo")
    account = DemoAccountRecord.from_row(account_row) if account_row is not None else None
    trades = (
        [DemoTrade.from_row(row) for row in repo.list_trades(account.id, limit=20)]
        if account is not None
        else []
    )
    stats = demo_statistics(account, trades) if account is not None else None
    if as_json:
        payload = {
            "account": jsonable_account(account) if account is not None else None,
            "trades": [jsonable_trade(t) for t in trades],
            "statistics": stats.as_dict() if stats is not None else None,
        }
        print(json.dumps(payload, indent=2, default=_json_default))
        return 0
    print(render_demo(account, trades, stats))
    return 0


def _health_running(health: dict[str, Any]) -> bool:
    return health.get("scheduler") == "RUNNING" and health.get("monitor") == "RUNNING"


def _health_json(health: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": health["state"],
        "pid": health.get("pid"),
        "started_at": health.get("started_at"),
        "scheduler": health["scheduler"],
        "scheduler_last_tick": health.get("scheduler_last_tick"),
        "monitor": health["monitor"],
        "monitor_last_poll": health.get("monitor_last_poll"),
        "last_error": health.get("last_error"),
        "running": _health_running(health),
    }


def _cmd_health(config: RuntimeConfig, as_json: bool) -> int:
    database = _open_db(config)
    runtime = Runtime(config, database=database)
    health = runtime.health()
    running = _health_running(health)
    if as_json:
        print(json.dumps(_health_json(health), indent=2))
    else:
        print(render_health(health))
    return 0 if running else 1


def _cmd_web(config: RuntimeConfig, host: str, port: int) -> int:
    from signal_engine.config import ConfigService
    from web.server import WebServer, build_app

    database = _open_db(config)
    app = build_app(database, config=config, config_service=ConfigService(database))
    server = WebServer(app, host=host, port=port)
    print(
        "BTCUSDT Web Dashboard: " + server.url + "\n"
        "Stop with Ctrl+C. Signal creation stays DEMO-only.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nShutting down the dashboard.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _config(args)

    if args.command == "run":
        if args.json:
            sys.stderr.write("--json applies to query commands only; ignoring.\n")
        return _run_daemon(config, once=args.once)

    if args.command == "web":
        host = args.host or os.environ.get("BTCUSDT_WEB_HOST") or "127.0.0.1"
        port = args.port or int(os.environ.get("BTCUSDT_WEB_PORT") or 8000)
        return _cmd_web(config, host, port)

    if args.once:
        sys.stderr.write("--once applies to the daemon only; ignoring it.\n")

    if args.command == "status":
        return _cmd_status(config, args.json)
    if args.command == "signals":
        return _cmd_signals(config, args.json)
    if args.command == "active":
        return _cmd_active(config, args.json)
    if args.command == "demo":
        return _cmd_demo(config, args.json)
    if args.command == "health":
        return _cmd_health(config, args.json)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
