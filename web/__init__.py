"""Phase 15: the BTCUSDT Web Dashboard + Settings UI.

A dependency-free web layer built on the standard-library
``http.server.ThreadingHTTPServer`` (no Flask/FastAPI/uvicorn needed). It
serves a small static single-page app and a read-only JSON API over the
SQLite ledger plus the :class:`signal_engine.config.ConfigService`.

The web layer NEVER writes trading state: signals, entry/SL/TP, positions,
trades, and history are immutable from here. Only Settings (AI provider /
demo / telegram) are writable, and always through the ConfigService safety
rules (AI provider changes apply only when idle; Demo changes are refused
while a signal is active).
"""

from __future__ import annotations

from .server import STATIC_DIR, WebApplication, WebServer, build_app, serve

__all__ = ["STATIC_DIR", "WebApplication", "WebServer", "build_app", "serve"]
