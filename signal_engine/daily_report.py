"""Daily Telegram report (Phase P6a): one morning summary on the reports channel.

Sent once per Vietnam-calendar day at/after 07:00 (+07) from the daemon loop:
yesterday's signals, demo PnL, candle-outcome mix (incl. vetoes), balance and
the last error. Pure functions for testability; the runtime only gathers rows
and persists the ``report.last_daily`` flag (restart-safe, exactly-once).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

logger = logging.getLogger(__name__)

#: Vietnam wall-clock hour the report goes out (fixed UTC+7, no DST).
REPORT_HOUR_VN = 7
#: Persisted exactly-once flag (value: ``YYYY-MM-DD`` of the reported VN day).
REPORT_LAST_DAILY_KEY = "report.last_daily"


def vn_now(now_utc: datetime) -> datetime:
    """Shift a UTC instant to Vietnam wall-clock (naive-safe: assumes UTC)."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    return now_utc.astimezone(timezone.utc) + timedelta(hours=7)


def report_window(now_utc: datetime) -> tuple[str, str, str]:
    """UTC ISO bounds + label of the VN day being reported.

    At/after 07:00 VN on day D this covers VN day D-1 (00:00-24:00 +07),
    returned as inclusive ``...Z`` bounds comparable with stored timestamps.
    """
    today_vn = vn_now(now_utc).date()
    target = today_vn - timedelta(days=1)
    start_vn = datetime(target.year, target.month, target.day, tzinfo=timezone.utc) - timedelta(hours=7)
    end_vn = start_vn + timedelta(days=1)
    label = target.strftime("%d/%m")
    return (
        start_vn.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_vn.strftime("%Y-%m-%dT%H:%M:%SZ"),
        label,
    )


def should_send_daily(last_sent: str | None, now_utc: datetime) -> bool:
    """True once the 07:00 VN threshold for a new VN day is reached."""
    today_vn = vn_now(now_utc).date()
    if (last_sent or "") == today_vn.strftime("%Y-%m-%d"):
        return False
    return vn_now(now_utc).hour >= REPORT_HOUR_VN


def _money(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError, AttributeError):
        return Decimal("0")


def _fmt_usd(value: Decimal) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}${value:,.2f}"


def build_daily_report(
    *,
    signals: list[Any],
    trades: list[Any],
    candle_rows: list[Any],
    balance: Decimal | None,
    initial_balance: Decimal | None,
    last_error: str | None,
    day_label: str,
) -> str:
    """Compose the report text from already-filtered window rows (never raises)."""
    try:
        return _build(
            signals=signals,
            trades=trades,
            candle_rows=candle_rows,
            balance=balance,
            initial_balance=initial_balance,
            last_error=last_error,
            day_label=day_label,
        )
    except Exception as exc:  # never break the daemon loop
        logger.warning("[Report] build failed: %s", exc)
        return f"📊 Báo cáo ngày {day_label} (+07): không tổng hợp được ({exc})."


def _build(
    *,
    signals: list[Any],
    trades: list[Any],
    candle_rows: list[Any],
    balance: Decimal | None,
    initial_balance: Decimal | None,
    last_error: str | None,
    day_label: str,
) -> str:
    longs = shorts = waits = 0
    for signal in signals or []:
        decision = getattr(signal, "decision", None)
        if isinstance(decision, str):
            name = decision.upper()
        else:
            name = str(getattr(decision, "value", getattr(decision, "name", ""))).upper()
        if name == "LONG":
            longs += 1
        elif name == "SHORT":
            shorts += 1
        else:
            waits += 1
    net = Decimal("0")
    wins = losses = 0
    for trade in trades or []:
        pnl = _money(getattr(trade, "net_pnl", 0))
        net += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
    outcomes: dict[str, int] = {}
    for row in candle_rows or []:
        outcome = str(getattr(row, "outcome", "?"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    total_candles = sum(outcomes.values())
    veto_bits = []
    for key in ("EVENT_BLACKOUT", "REJECTED", "BLOCKED_ACTIVE_SIGNAL", "RATE_LIMITED"):
        if outcomes.get(key):
            veto_bits.append(f"{key} {outcomes[key]}")
    lines = [
        f"📊 Báo cáo ngày {day_label} (+07)",
        f"• Signal: {longs + shorts + waits} (LONG {longs} / SHORT {shorts} / WAIT {waits})",
        f"• Trade: {wins + losses} (thắng {wins} / thua {losses}, net {_fmt_usd(net)})",
    ]
    if balance is not None:
        base = f" (vốn {_fmt_usd(initial_balance)})" if initial_balance is not None else ""
        lines.append(f"• Balance: {_fmt_usd(balance)}{base}")
    if total_candles:
        lines.append(
            f"• Nến phân tích: {total_candles} "
            f"(WAIT {outcomes.get('WAIT', 0)} · CREATED {outcomes.get('CREATED', 0)}"
            + (f" · {' · '.join(veto_bits)}" if veto_bits else "")
            + ")"
        )
    lines.append(f"• Lỗi: {last_error}" if last_error else "• Lỗi: không")
    return "\n".join(lines)


__all__ = [
    "REPORT_HOUR_VN",
    "REPORT_LAST_DAILY_KEY",
    "build_daily_report",
    "report_window",
    "should_send_daily",
    "vn_now",
]
