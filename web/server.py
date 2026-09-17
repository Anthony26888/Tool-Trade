"""Phase 15: the Web Dashboard server (standard-library HTTP + JSON API).

Routes
------
Static SPA:
    GET  /                          -> index.html
    GET  /static/<file>             -> app.js / style.css

JSON API (all responses are ``application/json`` with ``ok`` envelope):
    GET  /api/health                -> runtime subsystem health (read-only)
    GET  /api/dashboard             -> account + stats + active signal + position + unrealized PnL + health
    GET  /api/chart?interval=1h     -> OHLCV candles + active-signal overlay (1m/5m/15m/1h/4h)
    GET  /api/signals?limit=50&status=&direction=&since=&until=
                                            -> signal ledger history (filters optional)
    GET  /api/trades?limit=50       -> demo trade ledger (immutable)
    GET  /api/statistics            -> win-rate / drawdown / fees / equity
    GET  /api/settings              -> masked settings + audit (NO secrets)
    PUT  /api/settings              -> update ai_provider / demo / telegram / symbol groups
    POST /api/settings/test-connection     -> live provider connectivity test
    GET  /api/settings/ollama/models?base_url=... -> installed Ollama models
    POST /api/settings/test-telegram        -> send a test Telegram message
    POST /api/demo/reset                    -> reset demo account to fresh start
    POST /api/runtime/clear-error           -> clear the stored last-error state
                                               (balance=equity=peak=initial_balance,
                                               clear positions/trades; 409 while active)
    DELETE /api/signals             -> delete signals by {"ids":[...]} or
                                       {"all":true}; OPEN signals never deleted;
                                       linked positions/trades cascade
    POST /api/signal-chat                   -> Stream (ndjson) chat about the
                                               active signal (read-only)

Security
--------
* API keys / bot tokens are never serialized into responses, errors, or logs.
* The demo/stats endpoints are read-only except one idempotent lazy bootstrap:
  they create the demo account row (``ensure_account``, which never resets an
  existing account) so the dashboard is usable before the daemon's first run.
  Settings mutations go through ``ConfigService`` locking and promotion rules.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.parse
from collections.abc import Iterator
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from binance.client import BinanceError
from binance.market_data import BinanceMarketData
from database.database import (
    Database,
    DemoRepository,
    RuntimeStateRepository,
    SignalRepository,
    SignalValidationError,
)
from demo.position import DemoAccountRecord, DemoPosition, DemoTrade
from demo.scenario import tpsl_outcomes, unrealized_pnl
from demo.statistics import demo_statistics
from signal_engine.config import (
    ConfigService,
    SettingsError,
    SettingsLockedError,
    SettingsTestError,
    SettingsValidationError,
)
from signal_engine.llm import LLMConfigError, build_llm_client
from signal_engine.runtime import (
    RUNTIME_KEY_LAST_ERROR,
    RUNTIME_KEY_LAST_ERROR_AT,
    RuntimeConfig,
    _jsonable,
    health_from_snapshot,
    runtime_config_from_env,
)
from web.signal_chat import (
    CANNED_NO_SIGNAL,
    MAX_QUESTION_CHARS,
    build_messages,
    stream_answer,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: Intervals the dashboard chart may display (default 1h = analysis timeframe).
CHART_INTERVALS = ("1m", "5m", "15m", "1h", "4h")
#: How long a fetched OHLCV snapshot stays cached. The SPA polls the dashboard
#: every 15s; this TTL keeps Binance traffic modest (AGENTS.md 21) while the
#: chart still tracks the forming candle closely.
CHART_CACHE_TTL = 25.0


# -- JSON helpers -----------------------------------------------------------------


def _signal_json(signal) -> dict[str, Any]:
    return {
        "id": signal.id,
        "symbol": signal.symbol,
        "timeframe": signal.timeframe,
        "direction": signal.direction,
        "status": signal.status,
        "entry": _jsonable(signal.entry),
        "stop_loss": _jsonable(signal.stop_loss),
        "take_profit": _jsonable(signal.take_profit),
        "confidence": signal.confidence,
        "provider": signal.provider,
        "model_name": signal.model_name,
        "temperature": signal.temperature,
        "analysis_timestamp": signal.analysis_timestamp,
        "market_timestamp": signal.market_timestamp,
        "created_at": signal.created_at,
        "opened_at": signal.opened_at,
        "closed_at": signal.closed_at,
        "close_price": _jsonable(signal.close_price),
        "close_reason": signal.close_reason,
        "result": signal.result,
    }


def _account_json(account) -> dict[str, Any]:
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
        "created_at": account.created_at,
        "updated_at": account.updated_at,
    }


def _position_json(position) -> dict[str, Any]:
    return {
        "id": position.id,
        "signal_id": position.signal_id,
        "symbol": position.symbol,
        "side": position.side,
        "entry_price": _jsonable(position.entry_price),
        "quantity": _jsonable(position.quantity),
        "position_size": _jsonable(position.position_size),
        "margin": _jsonable(position.margin),
        "leverage": position.leverage,
        "stop_loss": _jsonable(position.stop_loss),
        "take_profit": _jsonable(position.take_profit),
        "status": position.status,
        "opened_at": position.opened_at,
        "closed_at": position.closed_at,
    }


def _trade_json(trade) -> dict[str, Any]:
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
        "opened_at": trade.opened_at,
        "closed_at": trade.closed_at,
    }


def _decimal_stats(stats) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in stats.as_dict().items()}


class WebApplication:
    """Read-only dashboard data + Settings handled through ConfigService."""

    def __init__(
        self,
        database: Database,
        *,
        config: RuntimeConfig | None = None,
        config_service: ConfigService | None = None,
        market_data: BinanceMarketData | None = None,
        chat_responder=None,
    ) -> None:
        self.database = database
        self.config = config if config is not None else runtime_config_from_env()
        self.config_service = (
            config_service if config_service is not None else ConfigService(database)
        )
        self.market_data = (
            market_data if market_data is not None else BinanceMarketData()
        )
        self.chat_responder = (
            chat_responder if chat_responder is not None else stream_answer
        )
        self._signals = SignalRepository(database)
        self._demo = DemoRepository(database)
        self._chart_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._price_cache: tuple[str, float, Any] | None = None

    # -- Data ------------------------------------------------------------------

    def _demo_account(self) -> DemoAccountRecord:
        """The demo account row, created lazily when missing.

        Mirrors the daemon's account bootstrap so the dashboard is usable even
        before the daemon has produced its first signal. ``ensure_account`` is
        idempotent and never resets an existing row (AGENTS.md sections 17 and
        23), so a restored/backed-up account is left untouched.
        """
        row = self._demo.get_account("demo")
        if row is not None:
            return DemoAccountRecord.from_row(row)
        stored = self.config_service.demo_config_object()
        config = stored if stored is not None else self.config.demo
        row = self._demo.ensure_account(
            name="demo",
            initial_balance=config.initial_balance,
            margin_per_trade=config.margin_per_trade,
            leverage=config.leverage,
            risk_percent=config.risk_percent,
            fee_rate=config.fee_rate,
        )
        return DemoAccountRecord.from_row(row)

    def health(self) -> dict[str, Any]:
        snapshot = RuntimeStateRepository(self.database).snapshot()
        return health_from_snapshot(
            snapshot,
            # Once the daemon has heartbeated (scheduler/monitor last tick),
            # a running web process must not treat the timestamps as stale.
            now=datetime.now(timezone.utc),
            scheduler_poll=self.config.scheduler_poll,
            monitor_poll=self.config.monitor_poll,
        )

    def clear_last_error(self) -> dict[str, Any]:
        """Clear the stored ``last_error``/``last_error_at`` runtime state."""
        repo = RuntimeStateRepository(self.database)
        repo.set(RUNTIME_KEY_LAST_ERROR, "")
        repo.set(RUNTIME_KEY_LAST_ERROR_AT, "")
        return {"health": self.health()}

    def dashboard(self) -> dict[str, Any]:
        account = self._demo_account()
        trades = [DemoTrade.from_row(r) for r in self._demo.list_trades(account.id, limit=50)]
        stats = demo_statistics(account, trades)
        active = self._signals.get_active_signal()
        position = None
        if active is not None:
            pos_row = self._demo.get_position_for_signal(active.id)
            if pos_row is not None:
                position = DemoPosition.from_row(pos_row)
        position_outcomes = (
            tpsl_outcomes(account, position) if position is not None else None
        )
        return {
            "account": _account_json(account),
            "statistics": _decimal_stats(stats),
            "active_signal": _signal_json(active) if active is not None else None,
            "position": _position_json(position) if position is not None else None,
            "position_outcomes": position_outcomes,
            "unrealized": self._live_unrealized(position),
            "health": self.health(),
            "symbol": self.config_service.resolve_symbol(),
        }

    def _live_unrealized(self, position) -> dict[str, Any] | None:
        """Live PnL row (USDT + % margin) for an OPEN position, or None.

        Zero extra Binance calls when idle. When the price fetch fails the
        panel simply omits the row instead of breaking the dashboard.
        """
        if position is None:
            return None
        current = self._market_price(symbol=position.symbol)
        if current is None:
            return None
        payload = unrealized_pnl(position, current)
        return {key: _jsonable(value) for key, value in payload.items()}

    def _market_price(self, symbol: str | None = None) -> float | None:
        """Latest price (1m candle close) for ``symbol``, cached briefly.

        Live PnL always fetches the OPEN position's own symbol (never the
        changed Settings symbol), so a cross-symbol switch can't produce a
        misleading PnL. When ``symbol`` is omitted the configured analysis
        symbol is used. The monitor's 15s cadence matches this TTL, so the
        dashboard's live PnL row tracks the chart without hammering Binance
        (AGENTS.md 21). The cache is keyed per symbol so a Settings switch
        never shows stale data.
        """
        symbol = symbol or self.config_service.resolve_symbol()
        now = time.time()
        if (
            self._price_cache is not None
            and self._price_cache[0] == symbol
            and now - self._price_cache[1] < CHART_CACHE_TTL
        ):
            return self._price_cache[2]
        try:
            candles = self.market_data.fetch_klines(
                symbol=symbol, interval="1m", limit=1
            )
        except BinanceError:
            self._price_cache = (symbol, now, None)
            return None
        price = candles[-1].close if candles else None
        self._price_cache = (symbol, now, price)
        return price

    def signal_chat(self, question: str) -> Iterator[dict[str, Any]]:
        """Stream the Signal Chat drawer events for ``question``.

        Returns an iterator of JSON-serializable events:
        ``{"delta": "..."}`` per text chunk, ``{"text": "..."}`` for a canned
        reply, and ``{"error": "..."}`` when the LLM fails mid-stream.

        Validation/configuration errors raise synchronously (``SettingsError``)
        so the HTTP layer can reply 400/502 before the stream opens. This is
        strictly read-only: only database reads happen here (AGENTS.md 17, 27).
        """
        if not isinstance(question, str) or not question.strip():
            raise SettingsValidationError("question must be a non-empty string")
        if len(question) > MAX_QUESTION_CHARS:
            raise SettingsValidationError(
                f"question is too long (max {MAX_QUESTION_CHARS} characters)"
            )

        active = self._signals.get_active_signal()
        if active is None:
            return self._signal_chat_events([], None, canned=CANNED_NO_SIGNAL)

        llm_config = self.config_service.resolve_llm_config()
        try:
            build_llm_client(llm_config)
        except LLMConfigError as exc:
            raise SettingsTestError(
                "AI provider is not configured (missing API key or invalid model)"
            ) from exc

        context = self._signal_chat_context(active)
        messages = build_messages(question, context)
        return self._signal_chat_events(messages, llm_config, canned=None)

    def _signal_chat_context(self, signal) -> dict[str, Any]:
        """Read-only facts the chat may use, assembled fresh from the ledger."""
        account = self._demo_account()
        position = None
        pos_row = self._demo.get_position_for_signal(signal.id)
        if pos_row is not None:
            position = DemoPosition.from_row(pos_row)
        trades = [DemoTrade.from_row(r) for r in self._demo.list_trades(account.id)]
        stats = demo_statistics(account, trades)
        return {
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "signal": _signal_json(signal),
            "position": _position_json(position) if position is not None else None,
            "position_outcomes": (
                tpsl_outcomes(account, position) if position is not None else None
            ),
            "unrealized": self._live_unrealized(position),
            "account_risk": {
                "risk_percent": _jsonable(account.risk_percent),
                "fee_rate": _jsonable(account.fee_rate),
                "margin_per_trade": _jsonable(account.margin_per_trade),
                "leverage": account.leverage,
            },
            "statistics": _decimal_stats(stats),
        }

    def _signal_chat_events(
        self,
        messages: list[dict[str, str]],
        llm_config,
        *,
        canned: str | None,
    ) -> Iterator[dict[str, Any]]:
        def _gen():
            if canned is not None:
                yield {"text": canned}
                return
            try:
                for chunk in self.chat_responder(messages, llm_config):
                    if chunk:
                        yield {"delta": chunk}
            except Exception:
                logger.exception("[Web] signal chat LLM failed mid-stream")
                yield {"error": "AI unavailable — please try again"}

        return _gen()

    def signals(
        self,
        limit: int = 50,
        *,
        status: str | None = None,
        direction: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ledger read-out; optional filters are passed to the repository."""
        return [
            _signal_json(s)
            for s in self._signals.list_signals(
                limit=int(limit),
                status=status,
                direction=direction,
                created_since=since,
                created_until=until,
            )
        ]

    def delete_signals(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Delete signals (and their demo footprint) via explicit user action.

        Accepts ``{"ids": [...]}`` for a specific selection or ``{"all": true}``
        to purge every non-OPEN signal. OPEN signals are never touched and are
        reported in ``skipped_open``.
        """
        if not isinstance(payload, dict):
            raise SettingsValidationError("request body must be a JSON object")
        if payload.get("all") is True:
            counts = self._signals.delete_all_signals()
        else:
            ids = payload.get("ids")
            if not isinstance(ids, list) or not ids:
                raise SettingsValidationError(
                    "`ids` must be a non-empty array of signal ids (or `all: true`)"
                )
            counts = self._signals.delete_signals(ids)
        return {"counts": counts}

    def trades(self, limit: int = 50) -> dict[str, Any]:
        rows = self._demo.list_trades(limit=int(limit))
        trades = [_trade_json(DemoTrade.from_row(r)) for r in rows]
        return {"count": len(trades), "trades": trades}

    def chart(self, interval: str = "1h", limit: int = 160, symbol: str | None = None) -> dict[str, Any]:
        """OHLCV candles plus the active-signal overlay for the price chart.

        Data is served for the configured analysis symbol (Settings > DB > env
        > default), so switching the symbol in Settings immediately changes the
        chart without a daemon restart. The newest candle may still be forming;
        it is shown for reference only and is never used by the AI or the TP/SL
        monitor (AGENTS.md 5). Results are cached briefly (``CHART_CACHE_TTL``)
        so the 15s SPA poll does not hammer Binance. ``BinanceError`` propagates
        for a 502 mapping.
        """
        interval = interval.strip().lower()
        if interval not in CHART_INTERVALS:
            raise SettingsValidationError(
                f"unsupported chart interval {interval!r}; "
                f"choose one of {', '.join(CHART_INTERVALS)}"
            )
        limit = max(10, min(int(limit), 500))
        symbol = symbol or self.config_service.resolve_symbol()
        key = f"{symbol}:{interval}:{limit}"
        cached = self._chart_cache.get(key)
        if cached is not None and time.time() - cached[0] < CHART_CACHE_TTL:
            return cached[1]
        candles = self.market_data.fetch_klines(
            symbol=symbol, interval=interval, limit=limit
        )
        payload = {
            "candles": [
                {
                    "t": c.timestamp,
                    "o": c.open,
                    "h": c.high,
                    "l": c.low,
                    "c": c.close,
                    "v": c.volume,
                    "is_closed": c.is_closed,
                }
                for c in candles
            ],
            "overlay": self._signal_overlay(symbol),
        }
        self._chart_cache[key] = (time.time(), payload)
        return payload

    def _signal_overlay(self, symbol: str) -> dict[str, Any]:
        """Entry/TP/SL of the single active signal, or ``{}`` when idle.

        Only PENDING_ENTRY / OPEN signals carry validated entry, TP and SL, so
        an overlay is drawn exactly when the chart should show the levels. The
        overlay is served ONLY for the active signal's own symbol: while a
        position on BTCUSDT is open and the Settings symbol switches to
        ETHUSDT/XAUUSDT, the new symbol's chart must not stretch its price axis
        with unrelated BTC levels (which would render its candles invisible).
        """
        active = self._signals.get_active_signal()
        if active is None or active.symbol != symbol:
            return {}
        position = None
        pos_row = self._demo.get_position_for_signal(active.id)
        if pos_row is not None:
            position = DemoPosition.from_row(pos_row)
        return {
            "has_signal": True,
            "direction": active.direction,
            "status": active.status,
            "entry": _jsonable(active.entry),
            "take_profit": _jsonable(active.take_profit),
            "stop_loss": _jsonable(active.stop_loss),
            "model_name": active.model_name,
            "position_open": position is not None,
        }

    def statistics(self) -> dict[str, Any]:
        account = self._demo_account()
        trades = [DemoTrade.from_row(r) for r in self._demo.list_trades(account.id)]
        stats = demo_statistics(account, trades)
        return {"account": _account_json(account), "statistics": _decimal_stats(stats)}

    def settings(self) -> dict[str, Any]:
        return self.config_service.get_settings()

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise SettingsValidationError("settings body must be a JSON object")
        if "ai_provider" in payload:
            self.config_service.update_ai_provider(payload["ai_provider"])
        if "demo" in payload:
            self.config_service.update_demo(payload["demo"])
        if "telegram" in payload:
            self.config_service.update_telegram(payload["telegram"])
        if "symbol" in payload:
            self.config_service.update_symbol(payload["symbol"])
        return self.config_service.get_settings()


# -- HTTP plumbing ----------------------------------------------------------------


class _JsonHandler(BaseHTTPRequestHandler):
    """Small JSON/static router over the standard library HTTP server."""

    app: WebApplication
    server_version = "BTCUSDT-Web/1.0"

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: Any) -> None:
        payload = json.dumps(obj, default=_jsonable).encode("utf-8")
        self._send(status, payload, "application/json")

    def _ok(self, obj: Any) -> None:
        self._json(HTTPStatus.OK, {"ok": True, **obj})

    def _static(self, relative: str) -> None:
        root = STATIC_DIR.resolve()
        target = (root / relative).resolve()
        if root not in target.parents and target != root:
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        if not target.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".png": "image/png",
        }.get(target.suffix.lower(), "application/octet-stream")
        self._send(HTTPStatus.OK, target.read_bytes(), content_type)

    def _route_error(self, exception: Exception) -> None:
        if isinstance(exception, (SettingsValidationError, SignalValidationError)):
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exception)})
        elif isinstance(exception, SettingsLockedError):
            self._json(HTTPStatus.CONFLICT, {"ok": False, "error": str(exception)})
        elif isinstance(exception, SettingsTestError):
            self._json(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": str(exception)})
        else:
            logger.exception("[Web] unhandled route error")
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": "internal server error"},
            )

    # -- Route handlers ---------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            self._static("index.html")
            return
        if path.startswith("/static/"):
            self._static(path[len("/static/"):])
            return
        try:
            if path == "/api/health":
                self._ok({"health": self.app.health()})
            elif path == "/api/dashboard":
                self._ok(self.app.dashboard())
            elif path == "/api/chart":
                interval = query.get("interval", ["1h"])[0]
                limit = int(query.get("limit", ["160"])[0])
                self._ok({"chart": self.app.chart(interval=interval, limit=limit)})
            elif path == "/api/statistics":
                self._ok(self.app.statistics())
            elif path == "/api/signals":
                limit = int(query.get("limit", ["50"])[0])
                status = query.get("status", [""])[0] or None
                direction = query.get("direction", [""])[0] or None
                since = query.get("since", [""])[0] or None
                until = query.get("until", [""])[0] or None
                self._ok(
                    {
                        "signals": self.app.signals(
                            limit=limit,
                            status=status,
                            direction=direction,
                            since=since,
                            until=until,
                        )
                    }
                )
            elif path == "/api/trades":
                limit = int(query.get("limit", ["50"])[0])
                self._ok(self.app.trades(limit=limit))
            elif path == "/api/settings":
                self._ok({"settings": self.app.settings()})
            elif path == "/api/settings/ollama/models":
                base_url = query.get("base_url", [""])[0]
                self._ok({"models": self.app.config_service.list_ollama_models(base_url)})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
        except SettingsError as exc:
            self._route_error(exc)
        except BinanceError:
            self._json(
                HTTPStatus.BAD_GATEWAY,
                {"ok": False, "error": "market data unavailable"},
            )
        except Exception as exc:
            self._route_error(exc)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw) if raw else {}
        except (ValueError, UnicodeDecodeError) as exc:
            raise SettingsValidationError("request body must be a JSON object") from exc

    def do_PUT(self) -> None:  # noqa: N802 (stdlib API)
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/settings":
                self._ok({"settings": self.app.update_settings(self._read_body())})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
        except SettingsError as exc:
            self._route_error(exc)
        except Exception as exc:
            self._route_error(exc)

    def _stream_signal_chat(self, body: dict[str, Any]) -> None:
        """Send the chat reply as newline-delimited JSON, flushing per token."""
        question = body.get("question", "")
        try:
            events = self.app.signal_chat(question)
        except SettingsError as exc:
            self._route_error(exc)
            return
        except Exception as exc:
            self._route_error(exc)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for event in events:
                line = json.dumps(event).encode("utf-8")
                self.wfile.write(line + b"\n")
                self.wfile.flush()
        except OSError:
            pass  # client disconnected mid-stream
        finally:
            self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802 (stdlib API)
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/settings/test-connection":
                result = self.app.config_service.test_llm_connection(self._read_body())
                self._ok({"result": result})
            elif parsed.path == "/api/settings/test-telegram":
                self._ok({"result": self.app.config_service.test_telegram()})
            elif parsed.path == "/api/demo/reset":
                self._ok({"settings": self.app.config_service.reset_demo_account()})
            elif parsed.path == "/api/runtime/clear-error":
                self._ok(self.app.clear_last_error())
            elif parsed.path == "/api/signal-chat":
                self._stream_signal_chat(self._read_body())
            else:
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
        except SettingsError as exc:
            self._route_error(exc)
        except Exception as exc:
            self._route_error(exc)

    def do_DELETE(self) -> None:  # noqa: N802 (stdlib API)
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/signals":
                self._ok(self.app.delete_signals(self._read_body()))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
        except SettingsError as exc:
            self._route_error(exc)
        except Exception as exc:
            self._route_error(exc)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Never log query strings: a base_url could carry a key.
        logger.info("[Web] %s %s", self.address_string(), self.path.split("?", 1)[0])


def _handler_type(app: WebApplication) -> type[BaseHTTPRequestHandler]:
    def _init(self: _JsonHandler, *args: Any, **kwargs: Any) -> None:
        self.app = app
        super(_JsonHandler, self).__init__(*args, **kwargs)

    return type("Handler", (_JsonHandler,), {"__init__": _init})


class WebServer:
    """A daemon thread serving the Web Application on host:port."""

    def __init__(
        self,
        app: WebApplication,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        self.app = app
        self.host = host
        self.port = port
        self.httpd = ThreadingHTTPServer((host, port), _handler_type(app))
        self.httpd.daemon_threads = True
        self.httpd.allow_reuse_address = True
        self._thread: threading.Thread | None = None

    @property
    def bound_port(self) -> int:
        return int(self.httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.bound_port}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread = None


def build_app(
    database: Database,
    *,
    config: RuntimeConfig | None = None,
    config_service: ConfigService | None = None,
) -> WebApplication:
    return WebApplication(database, config=config, config_service=config_service)


def serve(config: RuntimeConfig, *, host: str = "127.0.0.1", port: int = 8000) -> WebServer:
    """Open the database, build the app, and serve until interrupted."""
    database = Database(config.db_path)
    database.initialize()
    config_service = ConfigService(database)
    app = build_app(database, config=config, config_service=config_service)
    server = WebServer(app, host=host, port=port)
    logger.info(
        "[Web] dashboard ready at http://%s:%s (demo account: %s, symbol %s)",
        host,
        server.bound_port,
        config.db_path,
        config.symbol,
    )
    server.serve_forever()
    return server
