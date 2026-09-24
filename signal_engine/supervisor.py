"""Multi-symbol supervisor (plan B', phase B3): one shared DB + N daemons.

Each daemon process owns exactly one symbol (``BTCUSDT_SYMBOL`` env) on the
shared account/DB; a single web dashboard serves one port. This module reads
the enabled combo from Settings (``app_settings.symbols``, falling back to
``["BTCUSDT"]``) and converges the process set: start missing daemons, stop
extra ones — but NEVER stop a symbol holding an OPEN position (its TP/SL
monitor would go blind) or start a duplicate (two daemons, one DB, would
race signals).

Stdlib only, no project imports: safe to drive from ``run-symbols.sh`` on any
machine with a Python. All side effects (spawn/kill/ps/file) funnel through
small injectable callables so pytest covers the planning without processes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal as _signal
import sqlite3
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

SUPPORTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "XAUUSDT")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_WEB_PORT = 8000

DAEMON_PID_TEMPLATE = "daemon-{symbol}.pid"
WEB_PID_FILE = "web.pid"


# -- Settings + ledger reads (stdlib sqlite3; never raises) ------------------


def read_enabled_symbols(db_path: str) -> list[str]:
    """Enabled combo from Settings; ``["BTCUSDT"]`` on any failure."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT value FROM app_settings WHERE key = 'symbols'"
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return [DEFAULT_SYMBOL]
        data = json.loads(row[0])
        raw = data.get("symbols") if isinstance(data, dict) else None
        symbols = []
        if isinstance(raw, list):
            for item in raw:
                normalized = str(item).strip().upper() if isinstance(item, str) else ""
                if normalized in SUPPORTED_SYMBOLS and normalized not in symbols:
                    symbols.append(normalized)
        return symbols or [DEFAULT_SYMBOL]
    except Exception as exc:
        logger.warning("[Supervisor] enabled-symbols read failed: %s", exc)
        return [DEFAULT_SYMBOL]


def open_symbols(db_path: str) -> set[str]:
    """Symbols currently holding an OPEN signal (empty set on any failure)."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM signals WHERE status = 'OPEN'"
            ).fetchall()
        finally:
            conn.close()
        return {str(row[0]).strip().upper() for row in rows if row[0]}
    except Exception as exc:
        logger.warning("[Supervisor] open-symbols read failed: %s", exc)
        return set()


# -- Process handling (injectable for tests) ---------------------------------


def is_alive(pid: int) -> bool:
    """True when a PID exists (never raises)."""
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, ValueError, OverflowError):
        return False
    except Exception:
        return False


def _pid_file(rundir: str, name: str) -> str:
    return os.path.join(rundir, name)


def read_pid(rundir: str, name: str) -> int | None:
    """PID from a pidfile, or None when missing/stale/unparseable."""
    try:
        with open(_pid_file(rundir, name), encoding="utf-8") as handle:
            pid = int(handle.read().strip().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return pid if is_alive(pid) else None


def _cmdline_matches(pid: int, *needles: str, ps_fn=None) -> bool:
    """Best-effort guard against stale pidfiles (PID reuse)."""
    try:
        if ps_fn is None:
            proc = subprocess.run(
                ["ps", "-o", "command=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            text = proc.stdout
        else:
            text = ps_fn(pid)
        return all(needle in text for needle in needles)
    except Exception:
        return False


def daemon_pids(rundir: str, ps_fn=None) -> dict[str, int]:
    """Live daemon PIDs keyed by symbol (stale pidfiles ignored)."""
    running: dict[str, int] = {}
    for symbol in SUPPORTED_SYMBOLS:
        pid = read_pid(rundir, DAEMON_PID_TEMPLATE.format(symbol=symbol))
        if pid is not None and _cmdline_matches(
            pid, "signal_engine", "daemon", ps_fn=ps_fn
        ):
            running[symbol] = pid
    return running


def web_pid(rundir: str, ps_fn=None) -> int | None:
    """Live web PID, or None."""
    pid = read_pid(rundir, WEB_PID_FILE)
    if pid is not None and _cmdline_matches(pid, "signal_engine", "web", ps_fn=ps_fn):
        return pid
    return None


# -- Planning (pure; the unit-tested core) ------------------------------------


def plan(
    desired: list[str], running: dict[str, int], blocked: set[str]
) -> dict[str, list[str]]:
    """Converge plan: start what's missing, stop what's extra-but-safe.

    Symbols holding an OPEN position are never stopped (reported in
    ``skipped_blocked``). Unknown desired symbols are ignored. Returns
    ``{"start": [...], "stop": [...], "skipped_blocked": [...]}`` in a
    deterministic (SUPPORTED_SYMBOLS) order.
    """
    wanted = [s for s in SUPPORTED_SYMBOLS if s in set(desired or [])]
    to_start = [s for s in wanted if s not in running]
    to_stop = [s for s in running if s not in wanted]
    skipped = sorted(s for s in to_stop if s in blocked)
    return {
        "start": to_start,
        "stop": sorted(s for s in to_stop if s not in blocked),
        "skipped_blocked": skipped,
    }


# -- Actions -------------------------------------------------------------------


def _spawn(
    python_bin: str,
    args: list[str],
    env: dict[str, str],
    cwd: str,
    log_path: str,
    spawn_fn=None,
) -> int:
    if spawn_fn is not None:
        return int(spawn_fn(python_bin, args, env, cwd, log_path))
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [python_bin, "-m", "signal_engine", *args],
            env=env,
            cwd=cwd or None,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return int(proc.pid)


def _terminate(pid: int, kill_fn=None, grace_s: float = 5.0) -> bool:
    if kill_fn is not None:
        kill_fn(pid)
        return True
    try:
        os.kill(pid, _signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return False
    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return True
        time.sleep(0.2)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, _signal.SIGKILL)
    return not is_alive(pid)


def _write_pid(rundir: str, name: str, pid: int) -> None:
    os.makedirs(rundir, exist_ok=True)
    with open(_pid_file(rundir, name), "w", encoding="utf-8") as handle:
        handle.write(str(pid))


def sync(
    *,
    db_path: str,
    rundir: str,
    web_port: int = DEFAULT_WEB_PORT,
    python_bin: str | None = None,
    cwd: str | None = None,
    spawn_fn=None,
    kill_fn=None,
    ps_fn=None,
) -> dict[str, object]:
    """Converge processes to the Settings combo. Returns a report dict.

    Starts missing daemons (+ the web dashboard when absent), stops extra
    daemons except those holding OPEN positions (reported, exit code 1).
    Never starts a duplicate, never stops a blinded position.
    """
    python_bin = python_bin or sys.executable
    cwd = cwd or os.getcwd()
    desired = read_enabled_symbols(db_path)
    running = daemon_pids(rundir, ps_fn=ps_fn)
    blocked = open_symbols(db_path) & set(running)
    actions = plan(desired, running, blocked)
    started: list[str] = []
    stopped: list[str] = []
    base_env = dict(os.environ)
    for symbol in actions["start"]:
        env = dict(base_env)
        env["BTCUSDT_SYMBOL"] = symbol
        env["BTCUSDT_DB_PATH"] = db_path
        pid = _spawn(
            python_bin,
            ["daemon"],
            env,
            cwd,
            os.path.join(rundir, f"{symbol.lower()}-daemon.log"),
            spawn_fn=spawn_fn,
        )
        _write_pid(rundir, DAEMON_PID_TEMPLATE.format(symbol=symbol), pid)
        started.append(symbol)
        logger.info("[Supervisor] started %s daemon (pid %s)", symbol, pid)
    for symbol in actions["stop"]:
        pid = running[symbol]
        _terminate(pid, kill_fn=kill_fn)
        with contextlib.suppress(OSError):
            os.remove(_pid_file(rundir, DAEMON_PID_TEMPLATE.format(symbol=symbol)))
        stopped.append(symbol)
        logger.info("[Supervisor] stopped %s daemon (pid %s)", symbol, pid)
    web_running = web_pid(rundir, ps_fn=ps_fn)
    web_started = False
    if web_running is None:
        env = dict(base_env)
        env["BTCUSDT_DB_PATH"] = db_path
        pid = _spawn(
            python_bin,
            ["web", "--host", "127.0.0.1", "--port", str(web_port)],
            env,
            cwd,
            os.path.join(rundir, "web.log"),
            spawn_fn=spawn_fn,
        )
        _write_pid(rundir, WEB_PID_FILE, pid)
        web_started = True
        logger.info("[Supervisor] started web dashboard (pid %s)", pid)
    return {
        "desired": desired,
        "started": started,
        "stopped": stopped,
        "skipped_blocked": actions["skipped_blocked"],
        "web_started": web_started,
        "exit_code": 1 if actions["skipped_blocked"] else 0,
    }


def stop_all(
    *,
    db_path: str,
    rundir: str,
    kill_fn=None,
    ps_fn=None,
) -> dict[str, object]:
    """Stop every daemon + web, except symbols holding OPEN positions."""
    running = daemon_pids(rundir, ps_fn=ps_fn)
    blocked = sorted(open_symbols(db_path) & set(running))
    stopped: list[str] = []
    for symbol in sorted(running):
        if symbol in blocked:
            continue
        _terminate(running[symbol], kill_fn=kill_fn)
        with contextlib.suppress(OSError):
            os.remove(_pid_file(rundir, DAEMON_PID_TEMPLATE.format(symbol=symbol)))
        stopped.append(symbol)
    web_stopped = False
    pid = web_pid(rundir, ps_fn=ps_fn)
    if pid is not None:
        _terminate(pid, kill_fn=kill_fn)
        with contextlib.suppress(OSError):
            os.remove(_pid_file(rundir, WEB_PID_FILE))
        web_stopped = True
    return {
        "stopped": stopped,
        "skipped_blocked": blocked,
        "web_stopped": web_stopped,
        "exit_code": 1 if blocked else 0,
    }


def status(*, db_path: str, rundir: str, ps_fn=None) -> dict[str, object]:
    """Point-in-time process + position table (read-only)."""
    running = daemon_pids(rundir, ps_fn=ps_fn)
    open_now = open_symbols(db_path)
    return {
        "desired": read_enabled_symbols(db_path),
        "daemons": {
            symbol: {"pid": running.get(symbol), "open": symbol in open_now}
            for symbol in SUPPORTED_SYMBOLS
        },
        "web_pid": web_pid(rundir, ps_fn=ps_fn),
        "open_symbols": sorted(open_now),
    }


# -- CLI -------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-symbols",
        description="Supervise per-symbol daemons on one shared DB (plan B').",
    )
    parser.add_argument(
        "command", choices=("sync", "stop", "status"), help="action to perform"
    )
    parser.add_argument(
        "--db",
        default=os.path.join("data", "btcusdt_signals.db"),
        help="shared SQLite path (default data/btcusdt_signals.db)",
    )
    parser.add_argument(
        "--rundir",
        default="logs",
        help="pidfiles + logs directory (default logs/)",
    )
    parser.add_argument(
        "--web-port", type=int, default=DEFAULT_WEB_PORT, help="dashboard port"
    )
    parser.add_argument(
        "--python",
        default=None,
        help="python binary for children (default: this interpreter)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "status":
        print(json.dumps(status(db_path=args.db, rundir=args.rundir), indent=2))
        return 0
    if args.command == "stop":
        report = stop_all(db_path=args.db, rundir=args.rundir)
    else:
        report = sync(
            db_path=args.db,
            rundir=args.rundir,
            web_port=args.web_port,
            python_bin=args.python,
        )
    print(json.dumps(report, indent=2, default=str))
    return int(report.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SYMBOL",
    "DEFAULT_WEB_PORT",
    "SUPPORTED_SYMBOLS",
    "daemon_pids",
    "is_alive",
    "open_symbols",
    "plan",
    "read_enabled_symbols",
    "read_pid",
    "status",
    "stop_all",
    "sync",
    "web_pid",
]
