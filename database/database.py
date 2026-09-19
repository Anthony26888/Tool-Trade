"""SQLite persistence layer for the BTCUSDT signal engine (Phase 3).

Storage decisions
-----------------

- **Money** (entry, SL, TP, fees, PnL, balances, ...) is saved in ``TEXT``
  columns as exact :class:`decimal.Decimal` strings. SQLite has no exact
  decimal type, so binary floating point is never used for money. Ordering
  ``CHECK`` constraints are only a defense-in-depth backstop that tolerates a
  ``1e-10`` relative margin (``CAST(... AS NUMERIC)`` is binary64 and cannot
  distinguish orderings that differ only beyond the type's precision); the
  precise gate is Python ``Decimal`` validation in ``validate_signal_prices``.
- **Timestamps** are ISO-8601 UTC strings ending in ``Z``.
- **Schema** is created/opened idempotently (``CREATE ... IF NOT EXISTS``);
  opening or initializing a database never destroys existing data. Databases
  whose ``signals.status`` CHECK predates ``PENDING_ENTRY`` are rebuilt in
  place (data preserved) so the status column accepts the new lifecycle state.
- **Writes** run inside explicit transaction blocks; any failure rolls back,
  so a signal is never left partially written.

Phase 3 scope: schema + signal persistence primitives. The
``demo_accounts``/``demo_positions``/``demo_trades`` tables are created now so
later phases reuse this schema, but no repositories exist for them yet.
"""

from __future__ import annotations

import math
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

if os.name != "nt":  # pragma: no cover - cross-process advisory locking
    import fcntl

from binance.market_data import validate_interval, validate_symbol

from .models import (
    ALLOWED_TRANSITIONS,
    DIRECTION_LONG,
    DIRECTION_SHORT,
    DIRECTIONS,
    STATUS_CANCELLED,
    STATUS_OPEN,
    STATUS_PENDING_ENTRY,
    STATUS_SL_HIT,
    STATUS_TP_HIT,
    STATUSES,
    InvalidTransitionError,
    Signal,
    SignalExistsError,
    SignalNotFoundError,
    SignalValidationError,
)

DEFAULT_DB_PATH = os.path.join("data", "btcusdt_signals.db")

DECIMAL_COLUMNS: dict[str, frozenset[str]] = {
    "signals": frozenset(
        {
            "entry",
            "stop_loss",
            "take_profit",
            "close_price",
            "risk_reward",
            "candle_close_price",
        }
    ),
    "demo_accounts": frozenset(
        {
            "initial_balance",
            "balance",
            "equity",
            "margin_per_trade",
            "risk_percent",
            "fee_rate",
            "peak_equity",
        }
    ),
    "demo_positions": frozenset(
        {
            "entry_price",
            "quantity",
            "position_size",
            "margin",
            "stop_loss",
            "take_profit",
            "unrealized_pnl",
        }
    ),
    "demo_trades": frozenset(
        {
            "entry_price",
            "exit_price",
            "quantity",
            "margin",
            "position_size",
            "gross_pnl",
            "fee",
            "net_pnl",
            "pnl_percent",
        }
    ),
}

_SIGNALS_COLUMNS: tuple[str, ...] = (
    "id",
    "symbol",
    "timeframe",
    "direction",
    "status",
    "entry",
    "stop_loss",
    "take_profit",
    "confidence",
    "risk_reward",
    "rationale",
    "provider",
    "model_name",
    "temperature",
    "strategy_name",
    "strategy_version",
    "created_at",
    "opened_at",
    "closed_at",
    "close_price",
    "close_reason",
    "result",
    "analysis_timestamp",
    "market_timestamp",
    "candle_close_price",
)

_SIGNALS_TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        direction TEXT NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
        status TEXT NOT NULL CHECK (status IN ('PENDING_ENTRY', 'OPEN', 'TP_HIT', 'SL_HIT', 'CANCELLED')),
        entry TEXT NOT NULL,
        stop_loss TEXT NOT NULL,
        take_profit TEXT NOT NULL,
        confidence INTEGER CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 100),
        risk_reward TEXT,
        rationale TEXT,
        provider TEXT,
        model_name TEXT,
        temperature TEXT,
        strategy_name TEXT,
        strategy_version TEXT,
        created_at TEXT NOT NULL,
        opened_at TEXT,
        closed_at TEXT,
        close_price TEXT,
        close_reason TEXT,
        result TEXT CHECK (result IS NULL OR result IN ('WIN', 'LOSS')),
        analysis_timestamp TEXT,
        market_timestamp TEXT,
        candle_close_price TEXT,
        CHECK (close_price IS NULL OR close_reason IS NOT NULL),
        CHECK (
            (direction = 'LONG' AND
             CAST(take_profit AS NUMERIC) >= CAST(entry AS NUMERIC) * (1 - 1e-10)
             AND CAST(entry AS NUMERIC) >= CAST(stop_loss AS NUMERIC) * (1 - 1e-10))
            OR
            (direction = 'SHORT' AND
             CAST(entry AS NUMERIC) >= CAST(take_profit AS NUMERIC) * (1 - 1e-10)
             AND CAST(stop_loss AS NUMERIC) >= CAST(entry AS NUMERIC) * (1 - 1e-10))
        )
    )
    """

_SCHEMA_TABLES: tuple[str, ...] = (
    _SIGNALS_TABLE_DDL,
    """
    CREATE TABLE IF NOT EXISTS demo_accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        initial_balance TEXT NOT NULL,
        balance TEXT NOT NULL,
        equity TEXT,
        margin_per_trade TEXT NOT NULL,
        leverage INTEGER NOT NULL,
        risk_percent TEXT NOT NULL,
        fee_rate TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS demo_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id INTEGER NOT NULL REFERENCES demo_accounts(id),
        signal_id INTEGER REFERENCES signals(id),
        symbol TEXT NOT NULL,
        side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
        entry_price TEXT NOT NULL,
        quantity TEXT NOT NULL,
        position_size TEXT NOT NULL,
        margin TEXT NOT NULL,
        leverage INTEGER NOT NULL,
        stop_loss TEXT NOT NULL,
        take_profit TEXT NOT NULL,
        unrealized_pnl TEXT,
        status TEXT NOT NULL CHECK (status IN ('OPEN', 'CLOSED')),
        opened_at TEXT NOT NULL,
        closed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS demo_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id INTEGER NOT NULL REFERENCES demo_accounts(id),
        position_id INTEGER REFERENCES demo_positions(id),
        signal_id INTEGER REFERENCES signals(id),
        side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
        entry_price TEXT NOT NULL,
        exit_price TEXT NOT NULL,
        quantity TEXT NOT NULL,
        margin TEXT NOT NULL,
        position_size TEXT NOT NULL,
        leverage INTEGER NOT NULL,
        gross_pnl TEXT NOT NULL,
        fee TEXT NOT NULL,
        net_pnl TEXT NOT NULL,
        pnl_percent TEXT NOT NULL,
        result TEXT NOT NULL CHECK (result IN ('WIN', 'LOSS')),
        opened_at TEXT NOT NULL,
        closed_at TEXT NOT NULL
    )
    """,
    # Phase 10: one row per (symbol, timeframe) recording the most recent 1H
    # candle already consumed by the analysis scheduler. Persisting this marker
    # (rather than caching it in memory) makes the "AI runs once per closed
    # candle" rule survive restarts and prevents duplicate LLM calls on WAIT.
    """
    CREATE TABLE IF NOT EXISTS scheduler_state (
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        last_processed_candle INTEGER NOT NULL,
        processed_at TEXT NOT NULL,
        PRIMARY KEY (symbol, timeframe)
    )
    """,
    # Phase 14: a small runtime-metadata store the daemon heartbeats so the CLI
    # can report live status (scheduler/monitor last activity, last error, next
    # analysis). This is NOT trading state: signals/positions/trades/accounts
    # remain the source of truth in their own tables.
    """
    CREATE TABLE IF NOT EXISTS runtime_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # Phase 15: user-editable Settings (AI provider / demo / telegram). Values
    # are JSON documents; API keys and bot tokens are NEVER stored here — they
    # live in a 0600-permission secrets file off the database. State (applied vs
    # pending-until-idle) is carried in the ai_provider keys, extra fields, and
    # staged rows described in signal_engine/config.py.
    """
    CREATE TABLE IF NOT EXISTS app_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # Phase 15: a metadata-only audit trail of every Settings change (which
    # setting, which action, a short human summary). No secret is ever written.
    """
    CREATE TABLE IF NOT EXISTS config_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        namespace TEXT NOT NULL,
        action TEXT NOT NULL,
        summary TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    # Phase 16: one diagnostic row per analysed 1H candle, explaining why the
    # daemon produced LONG/SHORT/WAIT or skipped the candle (AI locked, data
    # unavailable, LLM error). Upserted per (symbol, timeframe, candle open
    # time), so each candle keeps exactly one row even when a tick retries.
    # Purely diagnostic: it never gates or feeds trading logic.
    """
    CREATE TABLE IF NOT EXISTS candle_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        candle_timestamp_ms INTEGER NOT NULL,
        closed_at TEXT,
        recorded_at TEXT NOT NULL,
        outcome TEXT NOT NULL,
        decision TEXT CHECK (decision IS NULL OR decision IN ('LONG', 'SHORT', 'WAIT', 'NONE')),
        confidence INTEGER CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 100),
        entry TEXT,
        stop_loss TEXT,
        take_profit TEXT,
        close_price TEXT,
        signal_id INTEGER REFERENCES signals(id),
        provider TEXT,
        model TEXT,
        temperature TEXT,
        reasoning TEXT,
        error_notes TEXT,
        indicators_json TEXT,
        llm_calls INTEGER,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        UNIQUE (symbol, timeframe, candle_timestamp_ms)
    )
    """,
)

_SCHEMA_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)",
    "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_signals_timeframe ON signals(timeframe)",
    "CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_signals_opened_at ON signals(opened_at)",
    "CREATE INDEX IF NOT EXISTS idx_signals_closed_at ON signals(closed_at)",
    "CREATE INDEX IF NOT EXISTS idx_demo_accounts_name ON demo_accounts(name)",
    "CREATE INDEX IF NOT EXISTS idx_demo_positions_status ON demo_positions(status)",
    "CREATE INDEX IF NOT EXISTS idx_demo_positions_account_id ON demo_positions(account_id)",
    "CREATE INDEX IF NOT EXISTS idx_demo_positions_signal_id ON demo_positions(signal_id)",
    "CREATE INDEX IF NOT EXISTS idx_demo_trades_account_id ON demo_trades(account_id)",
    "CREATE INDEX IF NOT EXISTS idx_demo_trades_position_id ON demo_trades(position_id)",
    "CREATE INDEX IF NOT EXISTS idx_demo_trades_signal_id ON demo_trades(signal_id)",
    # Phase 14: at most one demo position and at most one demo trade ever derive
    # from a signal. No pre-Phase-14 code wrote these tables (no repository
    # existed), so a UNIQUE index cannot collide with legacy rows.
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_demo_positions_signal ON demo_positions(signal_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_demo_trades_signal ON demo_trades(signal_id)",
    "CREATE INDEX IF NOT EXISTS idx_candle_log_candle ON candle_log(symbol, timeframe, candle_timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_candle_log_recorded_at ON candle_log(recorded_at)",
)

_SCHEMA_STATEMENTS: tuple[str, ...] = (*_SCHEMA_TABLES, *_SCHEMA_INDEXES)

# Additive, data-preserving migrations for databases created before a column
# existed. ``CREATE TABLE IF NOT EXISTS`` cannot add columns to an existing
# table, so opening an older database must ALTER the table instead. All entries
# are nullable columns, so existing rows are never rewritten.
_ADDITIVE_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("signals", "analysis_timestamp", "TEXT"),
    ("signals", "market_timestamp", "TEXT"),
    ("signals", "candle_close_price", "TEXT"),
    # Phase 14: the demo account row tracks its peak equity so drawdown is
    # derivable and the value survives restart (SQLite stays the source of truth).
    ("demo_accounts", "peak_equity", "TEXT"),
    # Phase 15: the sampling temperature used to produce a signal (metadata).
    ("signals", "temperature", "TEXT"),
    # Quota audit: per-candle LLM usage (calls always recorded; token counts
    # best-effort and NULL when the provider path drops the metadata).
    ("candle_log", "llm_calls", "INTEGER"),
    ("candle_log", "prompt_tokens", "INTEGER"),
    ("candle_log", "completion_tokens", "INTEGER"),
    ("candle_log", "total_tokens", "INTEGER"),
)

_TRANSITION_COLUMNS = ("created_at", "opened_at", "closed_at")
_ORDER_COLUMNS = frozenset(_TRANSITION_COLUMNS) | {"id"}

#: Phase 16: how many diagnostic candle-log rows to keep (one row per analysed
#: 1H candle ≈ 24/day). Older rows are pruned after each write.
CANDLE_LOG_RETENTION = 2000


def iso_utc_now() -> str:
    """Current UTC time as an ISO-8601 string with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def normalize_iso_utc(value: str) -> str:
    """Normalize any ISO-8601 timestamp to the storage ``...Z`` ms form.

    Accepts naive or ``+00:00``/``Z``-suffixed timestamps (including date-only
    boundaries), so callers can pass either "2026-09-13T17:00" or a full
    timestamp. Anything that cannot be parsed raises :class:`SignalValidationError`.
    """
    try:
        text = str(value).strip()
        if not text:
            raise ValueError("empty timestamp")
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SignalValidationError(f"invalid ISO timestamp: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def to_decimal(value: Any, field: str) -> Decimal:
    """Parse ``value`` into a positive finite :class:`Decimal` or raise."""
    if isinstance(value, Decimal):
        decimal_value = value
    else:
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise SignalValidationError(f"{field} must be a decimal number") from exc
    if not decimal_value.is_finite() or decimal_value <= 0:
        raise SignalValidationError(f"{field} must be a positive finite number")
    return decimal_value


def to_decimal_signed(value: Any, field: str) -> Decimal:
    """Parse ``value`` into any finite :class:`Decimal` (positive, negative, zero).

    Used for signed money values such as gross/net PnL and PnL percents, where
    a loss is legitimately negative.
    """
    if isinstance(value, Decimal):
        decimal_value = value
    else:
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise SignalValidationError(f"{field} must be a decimal number") from exc
    if not decimal_value.is_finite():
        raise SignalValidationError(f"{field} must be a finite decimal number")
    return decimal_value


#: Phase 15: accepted sampling-temperature range for the AI provider settings.
_TEMPERATURE_MIN = 0.0
_TEMPERATURE_MAX = 5.0


def _temperature_to_text(value: Any) -> str | None:
    """Normalize a sampling ``temperature`` to its TEXT form, or ``None``.

    ``temperature`` is plain metadata on a signal (not money), so a float is
    acceptable; validation only rejects non-finite or out-of-range values.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise SignalValidationError("temperature must be a number")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise SignalValidationError(
            "temperature must be a number between 0.0 and 5.0"
        ) from exc
    if not math.isfinite(number):
        raise SignalValidationError("temperature must be a finite number")
    if number < _TEMPERATURE_MIN or number > _TEMPERATURE_MAX:
        raise SignalValidationError("temperature must be between 0.0 and 5.0")
    return repr(number)


def validate_signal_prices(
    direction: str, entry: Decimal, stop_loss: Decimal, take_profit: Decimal
) -> None:
    """Enforce LONG ``SL < entry < TP`` and SHORT ``TP < entry < SL``."""
    if direction == DIRECTION_LONG:
        if not (stop_loss < entry < take_profit):
            raise SignalValidationError("LONG requires stop_loss < entry < take_profit")
    elif direction == DIRECTION_SHORT:
        if not (take_profit < entry < stop_loss):
            raise SignalValidationError("SHORT requires take_profit < entry < stop_loss")
    else:
        raise SignalValidationError(f"direction must be one of {sorted(DIRECTIONS)}")


class Database:
    """Owns the SQLite file, idempotent schema, and transaction primitives."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = str(path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.isolation_level = None
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def initialize(self) -> None:
        """Create/upgrade the schema. Idempotent and non-destructive.

        If an older database exists whose ``signals.status`` CHECK does not
        accept ``PENDING_ENTRY``, the ``signals`` table is rebuilt in place
        (data preserved) so the new lifecycle state can be persisted. Foreign
        keys are disabled only for the rebuild transaction; every later
        operation runs with them enabled again.

        The container runs the web server and the daemon as separate processes
        sharing one database file, so schema setup is serialised with an
        advisory ``flock``: only one process performs the WAL/schema dance and
        the other waits deterministically instead of hitting a ``journal_mode``
        write race ("database is locked").
        """
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        lock_path = self.path + ".lock"
        with open(lock_path, "a+b") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                self._initialize()
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    def _initialize(self) -> None:
        """Schema migration body, wrapped by :meth:`initialize` in the lock."""
        conn = self.connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            # foreign_keys cannot be changed inside a transaction; the rebuild
            # below temporarily drops/recreates the parent signals table.
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("BEGIN")
            try:
                if self._signals_needs_rebuild(conn):
                    self._rebuild_signals_table(conn)
                for statement in _SCHEMA_STATEMENTS:
                    conn.execute(statement)
                for table, column, column_type in _ADDITIVE_MIGRATIONS:
                    columns = {
                        row[1] for row in conn.execute(f"PRAGMA table_info({table})")
                    }
                    if column not in columns:
                        conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
                        )
            except sqlite3.Error:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")
        finally:
            conn.close()

    @staticmethod
    def _signals_needs_rebuild(conn: sqlite3.Connection) -> bool:
        """True when an existing re-signals table rejects ``PENDING_ENTRY``.

        SQLite stores each table's full ``CREATE`` statement in
        ``sqlite_master``; a pre-PENDING_ENTRY schema omits the new status from
        the column CHECK. A fresh database (no table yet) needs no rebuild.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'signals'"
        ).fetchone()
        if row is None or row["sql"] is None:
            return False
        return "PENDING_ENTRY" not in row["sql"]

    @staticmethod
    def _rebuild_signals_table(conn: sqlite3.Connection) -> None:
        """Rebuild ``signals`` with the PENDING_ENTRY-aware CHECK, preserving rows.

        SQLite cannot modify a CHECK constraint, so the standard in-place
        migration applies: create a mirror table with the new DDL, copy the
        (intersecting superset of) columns, drop the old table, and rename.
        ``id`` values (and therefore AUTOINCREMENT ordering) are preserved;
        additive nullable columns absent from very old schemas become NULL.
        """
        old_columns = [
            row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()
        ]
        present = [column for column in _SIGNALS_COLUMNS if column in old_columns]
        target_columns = ", ".join(present)
        conn.execute("DROP TABLE IF EXISTS signals_new")
        conn.execute(_SIGNALS_TABLE_DDL.replace("signals", "signals_new", 1))
        conn.execute(
            f"INSERT INTO signals_new ({target_columns}) "
            f"SELECT {target_columns} FROM signals"
        )
        conn.execute("DROP TABLE signals")
        conn.execute("ALTER TABLE signals_new RENAME TO signals")
        # sqlite_sequence keeps AUTOINCREMENT monotonicity across the copy.
        if "id" in old_columns:
            conn.execute(
                "UPDATE sqlite_sequence SET seq = (SELECT MAX(id) FROM signals) "
                "WHERE name = 'signals'"
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Write transaction: commits on success, rolls back on any failure."""
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
        except BaseException:
            with suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")
        finally:
            conn.close()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """Read-only connection with no active transaction."""
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()


class SignalRepository:
    """Signal persistence API used by later phases (engine, monitor, recovery).

    Enforces the critical trading invariants:

    - at most one active signal globally (``PENDING_ENTRY`` or ``OPEN``);
    - entry/SL/TP are immutable after creation;
    - LONG requires ``SL < entry < TP``, SHORT requires ``TP < entry < SL``;
    - only the allowed status transitions may be applied;
    - a closed signal can never reopen;
    - ``PENDING_ENTRY`` signals never close straight to TP/SL: the entry must
      first be touched (``PENDING_ENTRY -> OPEN``), then exit monitoring applies.
    """

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_signal(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        entry: Any,
        stop_loss: Any,
        take_profit: Any,
        *,
        confidence: int | None = None,
        risk_reward: Any | None = None,
        rationale: str | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        temperature: Any | None = None,
        strategy_name: str | None = None,
        strategy_version: str | None = None,
        created_at: str | None = None,
        opened_at: str | None = None,
        analysis_timestamp: str | None = None,
        market_timestamp: str | None = None,
        candle_close_price: Any | None = None,
    ) -> Signal:
        """Create and persist a new PENDING_ENTRY signal.

        The AI never opens a position directly: a validated LONG/SHORT is
        persisted as ``PENDING_ENTRY`` (awaiting its Entry trigger). Creation
        is idempotently rejected when any active signal (PENDING_ENTRY or
        OPEN) already exists.
        """
        symbol = validate_symbol(symbol)
        timeframe = validate_interval(timeframe)
        if direction not in DIRECTIONS:
            raise SignalValidationError(f"direction must be one of {sorted(DIRECTIONS)}")
        entry = to_decimal(entry, "entry")
        stop_loss = to_decimal(stop_loss, "stop_loss")
        take_profit = to_decimal(take_profit, "take_profit")
        validate_signal_prices(direction, entry, stop_loss, take_profit)
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, int):
                raise SignalValidationError("confidence must be an integer")
            if not 0 <= confidence <= 100:
                raise SignalValidationError("confidence must be between 0 and 100")
        temperature_value = _temperature_to_text(temperature)
        risk = to_decimal(risk_reward, "risk_reward") if risk_reward is not None else None
        if rationale is not None and not rationale.strip():
            raise SignalValidationError("rationale must be a non-empty string")
        for label, value in (
            ("analysis_timestamp", analysis_timestamp),
            ("market_timestamp", market_timestamp),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise SignalValidationError(f"{label} must be a non-empty string")
        candle_close = (
            to_decimal(candle_close_price, "candle_close_price")
            if candle_close_price is not None
            else None
        )

        created = created_at if created_at is not None else iso_utc_now()
        # A PENDING_ENTRY signal is not yet open; opened_at is recorded by the
        # PENDING_ENTRY -> OPEN transition.
        opened = opened_at
        with self.database.transaction() as conn:
            existing = conn.execute(
                "SELECT 1 FROM signals WHERE status IN (?, ?) LIMIT 1",
                (STATUS_PENDING_ENTRY, STATUS_OPEN),
            ).fetchone()
            if existing is not None:
                raise SignalExistsError(
                    "an active signal (PENDING_ENTRY or OPEN) already exists; "
                    "only one signal may be pending/open at any time"
                )
            cursor = conn.execute(
                """
                INSERT INTO signals (
                    symbol, timeframe, direction, status, entry, stop_loss,
                    take_profit, confidence, risk_reward, rationale, provider,
                    model_name, temperature, strategy_name, strategy_version,
                    created_at, opened_at, analysis_timestamp, market_timestamp,
                    candle_close_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    timeframe,
                    direction,
                    STATUS_PENDING_ENTRY,
                    str(entry),
                    str(stop_loss),
                    str(take_profit),
                    confidence,
                    str(risk) if risk is not None else None,
                    rationale,
                    provider,
                    model_name,
                    temperature_value,
                    strategy_name,
                    strategy_version,
                    created,
                    opened,
                    analysis_timestamp,
                    market_timestamp,
                    str(candle_close) if candle_close is not None else None,
                ),
            )
            signal_id = int(cursor.lastrowid)
        return self.get_signal(signal_id)

    def get_signal(self, signal_id: int) -> Signal:
        """Load a signal by id, raising :class:`SignalNotFoundError` when absent."""
        with self.database.read() as conn:
            row = conn.execute(
                "SELECT * FROM signals WHERE id = ?", (signal_id,)
            ).fetchone()
        if row is None:
            raise SignalNotFoundError(f"no signal with id {signal_id}")
        return Signal.from_row(row)

    def get_open_signal(self) -> Signal | None:
        """Return the current OPEN signal, or None when idle."""
        with self.database.read() as conn:
            row = conn.execute(
                "SELECT * FROM signals WHERE status = ? ORDER BY id LIMIT 1",
                (STATUS_OPEN,),
            ).fetchone()
        return Signal.from_row(row) if row is not None else None

    def get_active_signal(self) -> Signal | None:
        """Return the single active signal (PENDING_ENTRY or OPEN), or None.

        Active signals occupy the one-active-slot and lock the AI; this is the
        query recovery and the monitor use to decide what to resume.
        """
        with self.database.read() as conn:
            row = conn.execute(
                "SELECT * FROM signals WHERE status IN (?, ?) ORDER BY id LIMIT 1",
                (STATUS_PENDING_ENTRY, STATUS_OPEN),
            ).fetchone()
        return Signal.from_row(row) if row is not None else None

    def list_signals(
        self,
        *,
        status: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        direction: str | None = None,
        created_since: str | None = None,
        created_until: str | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "id",
        desc: bool = True,
    ) -> list[Signal]:
        """Filtered, ordered read-out of signals for later phases.

        ``created_since`` / ``created_until`` are ISO-8601 UTC boundaries
        compared against the stored ``created_at`` (lexicographic on the
        normalized ``...Z`` storage form), inclusive on both ends.
        """
        if order_by not in _ORDER_COLUMNS:
            raise SignalValidationError(
                f"order_by must be one of {sorted(_ORDER_COLUMNS)}"
            )
        if limit < 0 or offset < 0:
            raise SignalValidationError("limit and offset must be non-negative")
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(self._normalize_status(status))
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(validate_symbol(symbol))
        if timeframe is not None:
            clauses.append("timeframe = ?")
            params.append(validate_interval(timeframe))
        if direction is not None:
            if direction not in DIRECTIONS:
                raise SignalValidationError(
                    f"direction must be one of {sorted(DIRECTIONS)}"
                )
            clauses.append("direction = ?")
            params.append(direction)
        if created_since is not None:
            clauses.append("created_at >= ?")
            params.append(normalize_iso_utc(created_since))
        if created_until is not None:
            clauses.append("created_at <= ?")
            params.append(normalize_iso_utc(created_until))
        where = " AND ".join(clauses) if clauses else "1"
        ordering = "DESC" if desc else "ASC"
        params.extend([limit, offset])
        with self.database.read() as conn:
            rows = conn.execute(
                f"SELECT * FROM signals WHERE {where} "
                f"ORDER BY {order_by} {ordering} LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [Signal.from_row(row) for row in rows]

    def delete_signals(
        self, ids: list[int], *, cascade: bool = True
    ) -> dict[str, list[int]]:
        """Atomically delete signals (and their demo footprint).

        OPEN signals are never deleted and are reported in ``skipped_open``.
        With ``cascade`` the linked ``demo_positions`` and ``demo_trades`` are
        deleted in the same transaction (FK-safe order). Without cascade, any
        signal referenced by a position/trade is kept and reported in
        ``skipped_linked``. Deletion is an explicit user action; no lifecycle
        code calls it automatically.
        """
        if not isinstance(ids, list) or not ids:
            raise SignalValidationError("ids must be a non-empty list of integers")
        normalized: list[int] = []
        for raw in ids:
            try:
                normalized.append(int(raw))
            except (TypeError, ValueError) as exc:
                raise SignalValidationError(f"invalid signal id: {raw!r}") from exc
        unique = list(dict.fromkeys(normalized))
        deleted: list[int] = []
        skipped_linked: list[int] = []
        with self.database.transaction() as conn:
            ph = ",".join("?" * len(unique))
            rows = conn.execute(
                f"SELECT id, status FROM signals WHERE id IN ({ph})", unique
            ).fetchall()
            existing = {row["id"]: row["status"] for row in rows}
            missing = sorted(id_ for id_ in unique if id_ not in existing)
            locked = sorted(
                id_ for id_, status in existing.items() if status == STATUS_OPEN
            )
            deletable = sorted(set(unique) - set(locked) - set(missing))
            if deletable:
                drop = ",".join("?" * len(deletable))
                if not cascade:
                    linked = {
                        row["signal_id"]
                        for row in conn.execute(
                            "SELECT DISTINCT signal_id FROM demo_positions "
                            f"WHERE signal_id IN ({drop}) "
                            "UNION SELECT DISTINCT signal_id FROM demo_trades "
                            f"WHERE signal_id IN ({drop})",
                            deletable + deletable,
                        )
                    }
                    for id_ in deletable:
                        if id_ in linked:
                            skipped_linked.append(id_)
                    deletable = sorted(set(deletable) - set(linked))
                if deletable:
                    drop = ",".join("?" * len(deletable))
                    conn.execute(
                        f"DELETE FROM demo_trades WHERE signal_id IN ({drop})",
                        deletable,
                    )
                    conn.execute(
                        f"DELETE FROM demo_positions WHERE signal_id IN ({drop})",
                        deletable,
                    )
                    conn.execute(
                        f"DELETE FROM signals WHERE id IN ({drop})", deletable
                    )
                    conn.execute(
                        "INSERT INTO config_audit "
                        "(namespace, action, summary, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            "signals",
                            "delete",
                            f"Deleted {len(deletable)} signal(s); skipped OPEN "
                            f"{len(locked)}, missing {len(missing)}, "
                            f"linked {len(skipped_linked)}.",
                            iso_utc_now(),
                        ),
                    )
                    deleted = deletable
        return {
            "deleted": deleted,
            "skipped_open": locked,
            "skipped_linked": skipped_linked,
            "missing": missing,
        }

    def delete_all_signals(
        self, *, cascade: bool = True
    ) -> dict[str, list[int]]:
        """Delete every non-OPEN signal (and its demo footprint) atomically."""
        with self.database.read() as conn:
            rows = conn.execute("SELECT id, status FROM signals").fetchall()
        open_ids = sorted(r["id"] for r in rows if r["status"] == STATUS_OPEN)
        deletable = sorted(r["id"] for r in rows if r["status"] != STATUS_OPEN)
        if not deletable:
            return {
                "deleted": [],
                "skipped_open": open_ids,
                "skipped_linked": [],
                "missing": [],
            }
        result = self.delete_signals(deletable, cascade=cascade)
        result["skipped_open"] = open_ids
        return result

    def transition_signal(
        self,
        signal_id: int,
        new_status: str,
        *,
        close_price: Any | None = None,
        close_reason: str | None = None,
        result: str | None = None,
        closed_at: str | None = None,
        opened_at: str | None = None,
    ) -> Signal:
        """Apply an allowed state transition.

        - ``PENDING_ENTRY -> OPEN``: records ``opened_at`` (the moment the Entry
          trigger confirmed) and sets no close fields.
        - ``PENDING_ENTRY -> CANCELLED`` / ``OPEN -> CANCELLED``: no close price.
        - ``OPEN -> TP_HIT`` / ``OPEN -> SL_HIT``: requires a ``close_price``.

        Entry/SL/TP are never modified. A signal that already reached a
        terminal state can never transition again.
        """
        new_status = self._normalize_status(new_status)
        close_value = (
            to_decimal(close_price, "close_price") if close_price is not None else None
        )
        if result is not None and result not in ("WIN", "LOSS"):
            raise SignalValidationError("result must be WIN, LOSS, or None")
        with self.database.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM signals WHERE id = ?", (signal_id,)
            ).fetchone()
            if row is None:
                raise SignalNotFoundError(f"no signal with id {signal_id}")
            current = Signal.from_row(row)
            if new_status not in ALLOWED_TRANSITIONS[current.status]:
                raise InvalidTransitionError(
                    f"cannot transition signal {signal_id} from "
                    f"{current.status!r} to {new_status!r}"
                )
            if new_status in (STATUS_TP_HIT, STATUS_SL_HIT) and close_value is None:
                raise SignalValidationError(
                    "TP_HIT and SL_HIT transitions require a close_price"
                )
            if new_status == STATUS_OPEN:
                # PENDING_ENTRY -> OPEN: record the open moment; never invent
                # or clear close fields on an opening position.
                opened = opened_at if opened_at is not None else iso_utc_now()
                conn.execute(
                    """
                    UPDATE signals
                    SET status = ?, opened_at = ?, closed_at = NULL,
                        close_price = NULL, close_reason = NULL, result = NULL
                    WHERE id = ?
                    """,
                    (new_status, opened, signal_id),
                )
            else:
                closed = closed_at if closed_at is not None else iso_utc_now()
                conn.execute(
                    """
                    UPDATE signals
                    SET status = ?, closed_at = ?, close_price = ?, close_reason = ?,
                        result = ?
                    WHERE id = ?
                    """,
                    (
                        new_status,
                        closed,
                        str(close_value) if close_value is not None else None,
                        close_reason if close_reason is not None else new_status,
                        result,
                        signal_id,
                    ),
                )
            updated = conn.execute(
                "SELECT * FROM signals WHERE id = ?", (signal_id,)
            ).fetchone()
        return Signal.from_row(updated)

    def cancel_signal(self, signal_id: int, *, close_reason: str | None = None) -> Signal:
        """Cancel the active signal (PENDING_ENTRY or OPEN -> CANCELLED)."""
        return self.transition_signal(
            signal_id, STATUS_CANCELLED, close_reason=close_reason
        )

    @staticmethod
    def _normalize_status(status: str) -> str:
        if status not in STATUSES:
            raise SignalValidationError(f"unknown status {status!r}")
        return status


class SchedulerStateRepository:
    """Persisted "last analyzed 1H candle" marker for the analysis scheduler.

    Phase 10: the scheduler may run the AI at most once per closed 1H candle.
    This marker row is the SQLite-backed source of truth for that rule, so the
    guarantee survives restarts and is not held in process memory. A WAIT
    decision produces no signal, so the signal table alone cannot record that a
    candle was already analyzed; the scheduler uses this store for that case
    (and to keep candle processing idempotent even when a signal is created).
    """

    def __init__(self, database: Database) -> None:
        self.database = database

    def last_processed_candle(self, symbol: str, timeframe: str) -> int | None:
        """Return the open-time (epoch ms) of the last processed candle, or None.

        ``symbol`` and ``timeframe`` identify one scheduler's candle stream.
        """
        symbol = validate_symbol(symbol)
        timeframe = validate_interval(timeframe)
        with self.database.read() as conn:
            row = conn.execute(
                "SELECT last_processed_candle FROM scheduler_state "
                "WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            ).fetchone()
        return int(row["last_processed_candle"]) if row is not None else None

    def record_processed_candle(
        self, symbol: str, timeframe: str, candle_ts: int, processed_at: str
    ) -> None:
        """Upsert the most recently processed candle for a stream.

        ``candle_ts`` is the candle open time in epoch milliseconds (UTC).
        ``processed_at`` is the ISO-8601 UTC moment the candle was consumed.
        """
        symbol = validate_symbol(symbol)
        timeframe = validate_interval(timeframe)
        if isinstance(candle_ts, bool) or not isinstance(candle_ts, int):
            raise SignalValidationError("candle_ts must be an integer")
        if candle_ts <= 0:
            raise SignalValidationError("candle_ts must be positive")
        if not isinstance(processed_at, str) or not processed_at.strip():
            raise SignalValidationError("processed_at must be a non-empty string")
        with self.database.transaction() as conn:
            conn.execute(
                """
                INSERT INTO scheduler_state (symbol, timeframe, last_processed_candle, processed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe)
                DO UPDATE SET last_processed_candle = excluded.last_processed_candle,
                              processed_at = excluded.processed_at
                """,
                (symbol, timeframe, candle_ts, processed_at),
            )


class DemoError(Exception):
    """Base class for demo account/position/trade persistence errors."""


class DemoValidationError(ValueError, DemoError):
    """A demo persistence input is invalid."""


class DemoNotFoundError(DemoError):
    """The requested demo account, position, or trade does not exist."""


class DemoDuplicateError(DemoError):
    """A demo position or trade already exists for the same signal."""


class DemoRepository:
    """SQLite persistence for the DEMO account, positions, and trades (Phase 14).

    The account row carries the configurable money/risk configuration plus the
    current ``balance``/``equity`` and the running ``peak_equity``. Opening a
    position reserves nothing in the balance (the Phase 8 accounting model moves
    the balance only at close), so ``equity`` equals ``balance`` at rest.

    Atomicity guarantees
    --------------------
    - ``ensure_account`` checks-then-inserts inside ``BEGIN IMMEDIATE``, so even
      two concurrent runtimes observe one account row.
    - ``create_position`` is idempotent per signal inside one transaction.
    - ``record_trade`` closes the position, inserts the trade, and updates the
      account balance/equity/peak equity in ONE transaction: a crash can never
      leave a trade without its balance update vice versa. The UNIQUE index on
      ``demo_trades.signal_id`` turns a double close into ``DemoDuplicateError``.
    """

    #: Database column names used for exact Decimal round-trips.
    _TRADE_COLUMNS: tuple[str, ...] = (
        "id", "account_id", "position_id", "signal_id", "side",
        "entry_price", "exit_price", "quantity", "margin", "position_size",
        "leverage", "gross_pnl", "fee", "net_pnl", "pnl_percent", "result",
        "opened_at", "closed_at",
    )

    def __init__(self, database: Database) -> None:
        self.database = database

    # -- Account --------------------------------------------------------------

    def ensure_account(
        self,
        *,
        name: str = "demo",
        initial_balance: Any = None,
        margin_per_trade: Any = None,
        leverage: int = 10,
        risk_percent: Any = None,
        fee_rate: Any = None,
        created_at: str | None = None,
    ) -> sqlite3.Row:
        """Return the existing ``name`` account or create it at the given config.

        Creating never resets an existing account: the stored balance/equity/
        peak_equity survive restarts (AGENTS.md sections 17 and 23).
        """
        if not isinstance(name, str) or not name.strip():
            raise DemoValidationError("account name must be a non-empty string")
        initial = (
            str(to_decimal(initial_balance, "initial_balance"))
            if initial_balance is not None
            else None
        )
        margin = (
            str(to_decimal(margin_per_trade, "margin_per_trade"))
            if margin_per_trade is not None
            else None
        )
        risk = (
            str(to_decimal(risk_percent, "risk_percent"))
            if risk_percent is not None
            else None
        )
        fee = str(to_decimal(fee_rate, "fee_rate")) if fee_rate is not None else None
        if leverage is None:
            raise DemoValidationError("leverage must be provided")
        if isinstance(leverage, bool) or not isinstance(leverage, int) or leverage < 1:
            raise DemoValidationError("leverage must be an integer >= 1")
        created = created_at if created_at is not None else iso_utc_now()

        with self.database.transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM demo_accounts WHERE name = ? LIMIT 1", (name,)
            ).fetchone()
            if existing is not None:
                return existing
            cursor = conn.execute(
                """
                INSERT INTO demo_accounts (
                    name, initial_balance, balance, equity, margin_per_trade,
                    leverage, risk_percent, fee_rate, created_at, updated_at,
                    peak_equity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    initial,
                    initial,
                    initial,
                    margin,
                    leverage,
                    risk,
                    fee,
                    created,
                    None,
                    initial,
                ),
            )
            return conn.execute(
                "SELECT * FROM demo_accounts WHERE id = ?", (int(cursor.lastrowid),)
            ).fetchone()

    def get_account(self, name: str = "demo") -> sqlite3.Row | None:
        """Return the named account row, or None."""
        with self.database.read() as conn:
            return conn.execute(
                "SELECT * FROM demo_accounts WHERE name = ? LIMIT 1", (name,)
            ).fetchone()

    def update_balance(
        self,
        account_id: int,
        *,
        balance: Any,
        equity: Any,
        peak_equity: Any,
        updated_at: str | None = None,
    ) -> sqlite3.Row:
        """Write the post-close balance/equity/peak equity atomically."""
        balance_s = str(to_decimal(balance, "balance"))
        equity_s = str(to_decimal(equity, "equity"))
        peak_s = str(to_decimal(peak_equity, "peak_equity"))
        updated = updated_at if updated_at is not None else iso_utc_now()
        with self.database.transaction() as conn:
            cur = conn.execute(
                """
                UPDATE demo_accounts
                SET balance = ?, equity = ?, peak_equity = ?, updated_at = ?
                WHERE id = ?
                """,
                (balance_s, equity_s, peak_s, updated, account_id),
            )
            if cur.rowcount == 0:
                raise DemoNotFoundError(f"no demo account with id {account_id}")
            return conn.execute(
                "SELECT * FROM demo_accounts WHERE id = ?", (account_id,)
            ).fetchone()

    def update_account_config(
        self,
        account_id: int,
        *,
        margin_per_trade: Any = None,
        leverage: Any = None,
        risk_percent: Any = None,
        fee_rate: Any = None,
        updated_at: str | None = None,
    ) -> sqlite3.Row:
        """Update the configurable account fields that apply to FUTURE trades.

        ``initial_balance``/``balance``/``peak_equity`` are never touched here:
        a Settings change must never rewrite the trading ledger (AGENTS.md
        section 23). The caller enforces the trading lock (no changes while a
        signal is activE); this method only persists the account row values.
        """
        margin = (
            str(to_decimal(margin_per_trade, "margin_per_trade"))
            if margin_per_trade is not None
            else None
        )
        risk = (
            str(to_decimal(risk_percent, "risk_percent"))
            if risk_percent is not None
            else None
        )
        fee = str(to_decimal(fee_rate, "fee_rate")) if fee_rate is not None else None
        if leverage is None:
            raise DemoValidationError("leverage must be provided")
        if isinstance(leverage, bool) or not isinstance(leverage, int) or leverage < 1:
            raise DemoValidationError("leverage must be an integer >= 1")
        if margin is None and risk is None and fee is None:
            raise DemoValidationError("at least one configurable field is required")
        updated = updated_at if updated_at is not None else iso_utc_now()
        with self.database.transaction() as conn:
            cur = conn.execute(
                """
                UPDATE demo_accounts
                SET margin_per_trade = COALESCE(?, margin_per_trade),
                    leverage = COALESCE(?, leverage),
                    risk_percent = COALESCE(?, risk_percent),
                    fee_rate = COALESCE(?, fee_rate),
                    updated_at = ?
                WHERE id = ?
                """,
                (margin, leverage, risk, fee, updated, account_id),
            )
            if cur.rowcount == 0:
                raise DemoNotFoundError(f"no demo account with id {account_id}")
            return conn.execute(
                "SELECT * FROM demo_accounts WHERE id = ?", (account_id,)
            ).fetchone()

    def reset_account(self, account_id: int, initial_balance: Any) -> sqlite3.Row:
        """Reset an account to a clean demo state in ONE transaction.

        Sets ``initial_balance``/``balance``/``equity``/``peak_equity`` to the
        given ``initial_balance`` and deletes all demo positions and demo trades
        for the account (full fresh start). The signal ledger is NEVER touched:
        it is immutable research history (AGENTS.md sections 17 and 23).

        A crash between steps can never leave a half-reset account because the
        whole reset is atomic. The caller enforces the trading lock (reset must
        be refused while a signal is active) and resolves the target balance.
        """
        initial = str(to_decimal(initial_balance, "initial_balance"))
        updated = iso_utc_now()
        with self.database.transaction() as conn:
            conn.execute(
                "DELETE FROM demo_trades WHERE account_id = ?", (account_id,)
            )
            conn.execute(
                "DELETE FROM demo_positions WHERE account_id = ?", (account_id,)
            )
            cur = conn.execute(
                """
                UPDATE demo_accounts
                SET initial_balance = ?,
                    balance = ?,
                    equity = ?,
                    peak_equity = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (initial, initial, initial, initial, updated, account_id),
            )
            if cur.rowcount == 0:
                raise DemoNotFoundError(f"no demo account with id {account_id}")
            return conn.execute(
                "SELECT * FROM demo_accounts WHERE id = ?", (account_id,)
            ).fetchone()

    # -- Positions ------------------------------------------------------------

    def create_position(
        self,
        *,
        account_id: int,
        signal_id: int,
        symbol: str,
        side: str,
        entry_price: Any,
        quantity: Any,
        position_size: Any,
        margin: Any,
        leverage: int,
        stop_loss: Any,
        take_profit: Any,
        opened_at: str | None = None,
    ) -> sqlite3.Row:
        """Insert the OPEN position for a signal, idempotently, in one transaction.

        A duplicate OPEN position for the same signal returns the existing row;
        a second trade lifecycle for the same signal is refused.
        """
        symbol = validate_symbol(symbol)
        if side not in DIRECTIONS:
            raise DemoValidationError(f"side must be one of {sorted(DIRECTIONS)}")
        if isinstance(leverage, bool) or not isinstance(leverage, int) or leverage < 1:
            raise DemoValidationError("leverage must be an integer >= 1")
        opened = opened_at if opened_at is not None else iso_utc_now()
        with self.database.transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM demo_positions WHERE signal_id = ?", (signal_id,)
            ).fetchone()
            if existing is not None:
                return existing
            cursor = conn.execute(
                """
                INSERT INTO demo_positions (
                    account_id, signal_id, symbol, side, entry_price, quantity,
                    position_size, margin, leverage, stop_loss, take_profit,
                    unrealized_pnl, status, opened_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
                """,
                (
                    account_id,
                    signal_id,
                    symbol,
                    side,
                    str(to_decimal(entry_price, "entry_price")),
                    str(to_decimal(quantity, "quantity")),
                    str(to_decimal(position_size, "position_size")),
                    str(to_decimal(margin, "margin")),
                    leverage,
                    str(to_decimal(stop_loss, "stop_loss")),
                    str(to_decimal(take_profit, "take_profit")),
                    None,
                    opened,
                ),
            )
            return conn.execute(
                "SELECT * FROM demo_positions WHERE id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()

    def get_position_for_signal(self, signal_id: int) -> sqlite3.Row | None:
        """Return the demo position tied to ``signal_id``, or None."""
        with self.database.read() as conn:
            return conn.execute(
                "SELECT * FROM demo_positions WHERE signal_id = ?", (signal_id,)
            ).fetchone()

    def list_positions(
        self, account_id: int | None = None, *, status: str | None = None
    ) -> list[sqlite3.Row]:
        clauses = ["1"]
        params: list[Any] = []
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = " AND ".join(clauses)
        with self.database.read() as conn:
            rows = conn.execute(
                f"SELECT * FROM demo_positions WHERE {where} ORDER BY id",
                params,
            ).fetchall()
        return list(rows)

    # -- Trades (atomic close: position + trade + account) ----------------------

    def record_trade(
        self,
        *,
        position_id: int,
        signal_id: int,
        account_id: int,
        side: str,
        entry_price: Any,
        exit_price: Any,
        quantity: Any,
        margin: Any,
        position_size: Any,
        leverage: int,
        gross_pnl: Any,
        fee: Any,
        net_pnl: Any,
        pnl_percent: Any,
        result: str,
        closed_at: str | None = None,
        next_balance: Any,
        next_equity: Any,
        next_peak_equity: Any,
        updated_at: str | None = None,
    ) -> sqlite3.Row:
        """Close a position, insert its trade, and update the account balance.

        All reads/writes happen inside a single ``BEGIN IMMEDIATE`` transaction.
        ``next_balance`` is the Phase 8 ``balance_after_close`` result
        (``current_balance + net_pnl``) and ``next_peak_equity`` the Phase 8
        ``update_peak_equity`` result; the repo stores exactly what the formulas
        return and never re-derives strategy numbers.
        """
        if side not in DIRECTIONS:
            raise DemoValidationError(f"side must be one of {sorted(DIRECTIONS)}")
        if isinstance(leverage, bool) or not isinstance(leverage, int) or leverage < 1:
            raise DemoValidationError("leverage must be an integer >= 1")
        if result not in ("WIN", "LOSS"):
            raise DemoValidationError("result must be WIN or LOSS")
        closed = closed_at if closed_at is not None else iso_utc_now()
        updated = updated_at if updated_at is not None else iso_utc_now()
        with self.database.transaction() as conn:
            position = conn.execute(
                "SELECT * FROM demo_positions WHERE id = ?", (position_id,)
            ).fetchone()
            if position is None:
                raise DemoNotFoundError(f"no demo position with id {position_id}")
            duplicate = conn.execute(
                "SELECT 1 FROM demo_trades WHERE signal_id = ? LIMIT 1", (signal_id,)
            ).fetchone()
            if duplicate is not None:
                raise DemoDuplicateError(
                    f"a demo trade for signal {signal_id} already exists"
                )
            conn.execute(
                "UPDATE demo_positions SET status = 'CLOSED', closed_at = ? WHERE id = ?",
                (closed, position_id),
            )
            cursor = conn.execute(
                """
                INSERT INTO demo_trades (
                    account_id, position_id, signal_id, side, entry_price,
                    exit_price, quantity, margin, position_size, leverage,
                    gross_pnl, fee, net_pnl, pnl_percent, result,
                    opened_at, closed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    position_id,
                    signal_id,
                    side,
                    str(to_decimal(entry_price, "entry_price")),
                    str(to_decimal(exit_price, "exit_price")),
                    str(to_decimal(quantity, "quantity")),
                    str(to_decimal(margin, "margin")),
                    str(to_decimal(position_size, "position_size")),
                    leverage,
                    str(to_decimal_signed(gross_pnl, "gross_pnl")),
                    str(to_decimal(fee, "fee")),
                    str(to_decimal_signed(net_pnl, "net_pnl")),
                    str(to_decimal_signed(pnl_percent, "pnl_percent")),
                    result,
                    str(position["opened_at"]),
                    closed,
                ),
            )
            account = conn.execute(
                "SELECT * FROM demo_accounts WHERE id = ?", (account_id,)
            ).fetchone()
            if account is None:
                raise DemoNotFoundError(f"no demo account with id {account_id}")
            conn.execute(
                """
                UPDATE demo_accounts
                SET balance = ?, equity = ?, peak_equity = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    str(to_decimal(next_balance, "balance")),
                    str(to_decimal(next_equity, "equity")),
                    str(to_decimal(next_peak_equity, "peak_equity")),
                    updated,
                    account_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM demo_trades WHERE id = ?", (int(cursor.lastrowid),)
            ).fetchone()
        return row

    def get_trade_for_signal(self, signal_id: int) -> sqlite3.Row | None:
        """Return the demo trade tied to ``signal_id``, or None."""
        with self.database.read() as conn:
            return conn.execute(
                "SELECT * FROM demo_trades WHERE signal_id = ?", (signal_id,)
            ).fetchone()

    def list_trades(
        self, account_id: int | None = None, *, limit: int = 100
    ) -> list[sqlite3.Row]:
        """Recent trades (newest first)."""
        if limit < 0:
            raise DemoValidationError("limit must be non-negative")
        with self.database.read() as conn:
            if account_id is not None:
                rows = conn.execute(
                    "SELECT * FROM demo_trades WHERE account_id = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (account_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM demo_trades ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        return list(rows)

    def count_trades(self, account_id: int | None = None) -> int:
        with self.database.read() as conn:
            if account_id is not None:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM demo_trades WHERE account_id = ?",
                    (account_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS n FROM demo_trades").fetchone()
        return int(row["n"])


class CandleLogEntry:
    """One diagnostic row of the per-candle daemon activity log (Phase 16)."""

    __slots__ = (
        "id",
        "symbol",
        "timeframe",
        "candle_timestamp_ms",
        "closed_at",
        "recorded_at",
        "outcome",
        "decision",
        "confidence",
        "entry",
        "stop_loss",
        "take_profit",
        "close_price",
        "signal_id",
        "provider",
        "model",
        "temperature",
        "reasoning",
        "error_notes",
        "indicators_json",
        "llm_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )

    def __init__(
        self,
        *,
        id: int | None,
        symbol: str,
        timeframe: str,
        candle_timestamp_ms: int,
        closed_at: str | None,
        recorded_at: str,
        outcome: str,
        decision: str | None,
        confidence: int | None,
        entry: str | None,
        stop_loss: str | None,
        take_profit: str | None,
        close_price: str | None,
        signal_id: int | None,
        provider: str | None,
        model: str | None,
        temperature: str | None,
        reasoning: str | None,
        error_notes: str | None,
        indicators_json: str | None,
        llm_calls: int | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
    ) -> None:
        self.id = id
        self.symbol = symbol
        self.timeframe = timeframe
        self.candle_timestamp_ms = candle_timestamp_ms
        self.closed_at = closed_at
        self.recorded_at = recorded_at
        self.outcome = outcome
        self.decision = decision
        self.confidence = confidence
        self.entry = entry
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.close_price = close_price
        self.signal_id = signal_id
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.reasoning = reasoning
        self.error_notes = error_notes
        self.indicators_json = indicators_json
        self.llm_calls = llm_calls
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> CandleLogEntry:
        return cls(
            id=int(row["id"]),
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            candle_timestamp_ms=int(row["candle_timestamp_ms"]),
            closed_at=row["closed_at"],
            recorded_at=row["recorded_at"],
            outcome=row["outcome"],
            decision=row["decision"],
            confidence=row["confidence"],
            entry=row["entry"],
            stop_loss=row["stop_loss"],
            take_profit=row["take_profit"],
            close_price=row["close_price"],
            signal_id=row["signal_id"],
            provider=row["provider"],
            model=row["model"],
            temperature=row["temperature"],
            reasoning=row["reasoning"],
            error_notes=row["error_notes"],
            indicators_json=row["indicators_json"],
            llm_calls=row["llm_calls"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
        )


class CandleLogRepository:
    """Per-candle daemon activity log (Phase 16).

    Stores one upserted row per ``(symbol, timeframe, candle open time)`` so
    every analysed 1H candle keeps a single, stable entry even when a tick
    retries the same candle (errors) or the AI stays locked across polls. The
    log is diagnostic only: it never influences signals, positions, or trades.
    """

    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert(
        self,
        *,
        symbol: str,
        timeframe: str,
        candle_timestamp_ms: int,
        outcome: str,
        recorded_at: str | None = None,
        closed_at: str | None = None,
        decision: str | None = None,
        confidence: int | None = None,
        entry: Any = None,
        stop_loss: Any = None,
        take_profit: Any = None,
        close_price: Any = None,
        signal_id: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: Any = None,
        reasoning: str | None = None,
        error_notes: str | None = None,
        indicators_json: str | None = None,
        llm_calls: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Insert or replace the diagnostic row for one candle.

        Prices/temperature are normalized to their TEXT form (like signals);
        ``decision`` is one of LONG/SHORT/WAIT/NONE for validated rows.
        Quota counters (``llm_calls`` etc.) must be non-negative integers;
        token counts are best-effort and stay NULL when the provider path
        drops the usage metadata.
        """
        if not isinstance(candle_timestamp_ms, int) or candle_timestamp_ms <= 0:
            raise SignalValidationError(
                "candle_timestamp_ms must be a positive integer"
            )
        if not outcome:
            raise SignalValidationError("outcome must be a non-empty string")
        confidence_int = None
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, int):
                raise SignalValidationError("confidence must be an integer")
            if not 0 <= confidence <= 100:
                raise SignalValidationError("confidence must be between 0 and 100")
            confidence_int = confidence
        quota: dict[str, int | None] = {}
        for field, value in (
            ("llm_calls", llm_calls),
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
            ("total_tokens", total_tokens),
        ):
            if value is None:
                quota[field] = None
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SignalValidationError(
                    f"{field} must be a non-negative integer"
                )
            else:
                quota[field] = value
        temperature_text = _temperature_to_text(temperature) if temperature is not None else None
        entry_text = _optional_price(str(entry), entry)
        stop_text = _optional_price(str(stop_loss), stop_loss)
        take_text = _optional_price(str(take_profit), take_profit)
        close_text = _optional_price(str(close_price), close_price)
        recorded = recorded_at if recorded_at is not None else iso_utc_now()
        with self.database.transaction() as conn:
            conn.execute(
                """
                INSERT INTO candle_log (
                    symbol, timeframe, candle_timestamp_ms, closed_at,
                    recorded_at, outcome, decision, confidence, entry,
                    stop_loss, take_profit, close_price, signal_id, provider,
                    model, temperature, reasoning, error_notes, indicators_json,
                    llm_calls, prompt_tokens, completion_tokens, total_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, candle_timestamp_ms)
                DO UPDATE SET
                    closed_at = excluded.closed_at,
                    recorded_at = excluded.recorded_at,
                    outcome = excluded.outcome,
                    decision = excluded.decision,
                    confidence = excluded.confidence,
                    entry = excluded.entry,
                    stop_loss = excluded.stop_loss,
                    take_profit = excluded.take_profit,
                    close_price = excluded.close_price,
                    signal_id = excluded.signal_id,
                    provider = excluded.provider,
                    model = excluded.model,
                    temperature = excluded.temperature,
                    reasoning = excluded.reasoning,
                    error_notes = excluded.error_notes,
                    indicators_json = excluded.indicators_json,
                    llm_calls = excluded.llm_calls,
                    prompt_tokens = excluded.prompt_tokens,
                    completion_tokens = excluded.completion_tokens,
                    total_tokens = excluded.total_tokens
                """,
                (
                    validate_symbol(symbol),
                    validate_interval(timeframe),
                    int(candle_timestamp_ms),
                    closed_at,
                    recorded,
                    outcome,
                    decision,
                    confidence_int,
                    entry_text,
                    stop_text,
                    take_text,
                    close_text,
                    signal_id,
                    provider,
                    model,
                    temperature_text,
                    reasoning,
                    error_notes,
                    indicators_json,
                    quota["llm_calls"],
                    quota["prompt_tokens"],
                    quota["completion_tokens"],
                    quota["total_tokens"],
                ),
            )
            conn.execute(
                "DELETE FROM candle_log WHERE id NOT IN "
                "(SELECT id FROM candle_log ORDER BY id DESC LIMIT ?)",
                (int(CANDLE_LOG_RETENTION),),
            )

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        decision: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> list[CandleLogEntry]:
        """Read the log newest-first; optional ``decision``/symbol/timeframe filters."""
        if limit < 0 or offset < 0:
            raise SignalValidationError("limit and offset must be non-negative")
        clauses: list[str] = []
        params: list[Any] = []
        if decision is not None:
            if decision not in ("LONG", "SHORT", "WAIT", "NONE"):
                raise SignalValidationError("decision must be one of LONG/SHORT/WAIT/NONE")
            clauses.append("decision = ?")
            params.append(decision)
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(validate_symbol(symbol))
        if timeframe is not None:
            clauses.append("timeframe = ?")
            params.append(validate_interval(timeframe))
        where = " AND ".join(clauses) if clauses else "1"
        params.extend([limit, offset])
        with self.database.read() as conn:
            rows = conn.execute(
                f"SELECT * FROM candle_log WHERE {where} "
                "ORDER BY id DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [CandleLogEntry.from_row(row) for row in rows]

    def clear(self) -> int:
        """Delete every candle-log row; returns how many were removed."""
        with self.database.transaction() as conn:
            cursor = conn.execute("DELETE FROM candle_log")
        return int(cursor.rowcount)


def _optional_price(value: Any, raw: Any) -> str | None:
    """Normalize an optional candle-log price to its TEXT form, or None."""
    if raw is None or str(raw).strip() in ("", "None"):
        return None
    try:
        decimal_value = to_decimal(raw, "price")
    except SignalValidationError:
        return str(raw)
    return str(decimal_value)


class RuntimeStateRepository:
    """Persisted key/value heartbeat store for runtime status (Phase 14).

    This store records operational metadata only (last scheduler/monitor
    activity, last error, the running flag). It is never used by the trading
    logic; signals/positions/trades/accounts decide trading behavior.
    """

    def __init__(self, database: Database) -> None:
        self.database = database

    def set(self, key: str, value: str, *, updated_at: str | None = None) -> None:
        if not isinstance(key, str) or not key.strip():
            raise SignalValidationError("runtime_state key must be a non-empty string")
        if not isinstance(value, str):
            raise SignalValidationError("runtime_state value must be a string")
        updated = updated_at if updated_at is not None else iso_utc_now()
        with self.database.transaction() as conn:
            conn.execute(
                """
                INSERT INTO runtime_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                               updated_at = excluded.updated_at
                """,
                (key, value, updated),
            )

    def get(self, key: str) -> str | None:
        with self.database.read() as conn:
            row = conn.execute(
                "SELECT value FROM runtime_state WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row is not None else None

    def snapshot(self) -> dict[str, str]:
        """Return every heartbeat key -> value (latest timestamps preserved)."""
        with self.database.read() as conn:
            rows = conn.execute("SELECT key, value FROM runtime_state").fetchall()
        return {row["key"]: row["value"] for row in rows}


class ConfigRepository:
    """SQLite persistence for the Phase 15 Settings and their audit trail.

    ``app_settings`` stores JSON documents keyed by namespace (``ai_provider``,
    ``demo``, ``telegram``, ``runtime``). Secrets are never written to these
    tables; only masked/configured flags live here. ``promote_setting`` moves a
    staged key onto the live key atomically (see ``signal_engine/config.py``).
    """

    def __init__(self, database: Database) -> None:
        self.database = database

    def get(self, key: str) -> str | None:
        with self.database.read() as conn:
            row = conn.execute(
                "SELECT value FROM app_settings WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row is not None else None

    def set(
        self,
        key: str,
        value: str,
        *,
        updated_at: str | None = None,
    ) -> None:
        if not isinstance(key, str) or not key.strip():
            raise DemoValidationError("app_settings key must be a non-empty string")
        if not isinstance(value, str):
            raise DemoValidationError("app_settings value must be a string")
        updated = updated_at if updated_at is not None else iso_utc_now()
        with self.database.transaction() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                updated_at = excluded.updated_at
                """,
                (key, value, updated),
            )

    def promote(self, staged_key: str, live_key: str) -> None:
        """Atomically copy a staged key onto the live key."""
        with self.database.transaction() as conn:
            row = conn.execute(
                "SELECT value, updated_at FROM app_settings WHERE key = ?",
                (staged_key,),
            ).fetchone()
            if row is None:
                return
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                updated_at = excluded.updated_at
                """,
                (live_key, row["value"], row["updated_at"] or iso_utc_now()),
            )

    def delete(self, key: str) -> None:
        with self.database.transaction() as conn:
            conn.execute("DELETE FROM app_settings WHERE key = ?", (key,))

    def list(self) -> list[tuple[str, str]]:
        with self.database.read() as conn:
            rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
        return [(row["key"], row["value"]) for row in rows]

    # -- Audit trail ----------------------------------------------------------

    def append_audit(
        self,
        namespace: str,
        action: str,
        summary: str,
        *,
        created_at: str | None = None,
    ) -> None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise DemoValidationError("audit namespace must be a non-empty string")
        if not isinstance(action, str) or not action.strip():
            raise DemoValidationError("audit action must be a non-empty string")
        if not isinstance(summary, str) or not summary.strip():
            raise DemoValidationError("audit summary must be a non-empty string")
        created = created_at if created_at is not None else iso_utc_now()
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO config_audit (namespace, action, summary, created_at) "
                "VALUES (?, ?, ?, ?)",
                (namespace, action, summary, created),
            )

    def list_audit(self, limit: int = 100) -> list[tuple[str, str, str, str]]:
        row_limit = int(limit)
        if row_limit < 0:
            raise DemoValidationError("limit must be non-negative")
        with self.database.read() as conn:
            rows = conn.execute(
                "SELECT namespace, action, summary, created_at FROM config_audit "
                "ORDER BY id DESC LIMIT ?",
                (row_limit,),
            ).fetchall()
        return [(r["namespace"], r["action"], r["summary"], r["created_at"]) for r in rows]
