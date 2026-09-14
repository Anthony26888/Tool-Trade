"""Validation boundary between Phase 5 AI output and Phase 6 persistence.

The Signal Engine is the authoritative layer, so the AI's structured analysis
(which is still float-typed in Phase 5) is only trusted here after full
validation. Every failure raises the Phase 3 ``SignalValidationError`` so the
engine can translate it into a REJECTED result without leaking raw exceptions.

Rules
-----
- Prices are ``Decimal`` only; no binary float comparisons anywhere.
- No rounding or quantization of AI prices. ``str()`` of the provided value is
  converted to an exact ``Decimal`` (via the Phase 3 ``to_decimal`` helper,
  which rejects zero, negative, NaN, +/- infinity, and malformed Decimals).
- LONG requires ``stop_loss < entry < take_profit``.
- SHORT requires ``take_profit < entry < stop_loss``.
- LONG/SHORT require every price level; a missing level rejects the analysis.
- Confidence must be a finite number within [0, 100] (the Phase 5 contract).
  It is normalized deterministically (half-up) to the INTEGER database column.
- An invalid LONG/SHORT is never downgraded to WAIT.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from database.database import to_decimal, validate_signal_prices
from database.models import SignalValidationError

from .analysis import DECISIONS, SignalAnalysis

_CONFIDENCE_MIN = 0
_CONFIDENCE_MAX = 100


@dataclass(frozen=True)
class SignalCandidate:
    """A fully validated, persistence-ready LONG/SHORT signal (Decimals only)."""

    decision: str
    entry: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    confidence: int
    reasoning: str
    provider: str | None
    model_name: str | None
    temperature: float | None
    symbol: str
    timeframe: str
    analysis_timestamp: str
    market_timestamp: str
    candle_close_price: Decimal | None


def validate_decision(decision: str) -> str:
    """Return ``decision`` when it is one of LONG/SHORT/WAIT, else raise."""
    if not isinstance(decision, str) or decision not in DECISIONS:
        raise SignalValidationError(
            f"decision must be one of {', '.join(DECISIONS)}"
        )
    return decision


def validate_confidence(value: float) -> int:
    """Validate the Phase 5 confidence contract and normalize to an integer.

    The database stores ``confidence`` in an INTEGER column, so the value is
    normalized deterministically with half-up rounding to the nearest integer.
    Any value outside [0, 100], non-finite, or non-numeric is rejected.
    """
    if isinstance(value, bool):
        raise SignalValidationError("confidence must be a number")
    if value is None:
        raise SignalValidationError("confidence is required")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise SignalValidationError("confidence must be a number") from exc
    if not decimal_value.is_finite():
        raise SignalValidationError("confidence must be finite")
    if decimal_value < _CONFIDENCE_MIN or decimal_value > _CONFIDENCE_MAX:
        raise SignalValidationError(
            f"confidence must be between {_CONFIDENCE_MIN} and {_CONFIDENCE_MAX}"
        )
    return int(decimal_value.to_integral_value(rounding=ROUND_HALF_UP))


def to_price(value, field: str) -> Decimal:
    """Exact, unambiguous Decimal conversion of a trading price.

    Delegates to the Phase 3 ``to_decimal`` rule (positive, finite, exact); the
    value is never rounded or quantized here.
    """
    return to_decimal(value, field)


def validate_analysis(analysis: SignalAnalysis) -> SignalCandidate:
    """Validate a Phase 5 analysis and return an exact, persistence-ready candidate.

    Raises:
        SignalValidationError: for any structural, numeric, or ordering problem.
    """
    if not isinstance(analysis, SignalAnalysis):
        raise SignalValidationError(
            "analysis must be a SignalAnalysis result from Phase 5"
        )
    decision = validate_decision(analysis.decision)
    if decision == "WAIT":
        raise SignalValidationError("WAIT cannot be validated as a tradable signal")

    missing = [
        name
        for name, value in (
            ("entry_price", analysis.entry_price),
            ("stop_loss", analysis.stop_loss),
            ("take_profit", analysis.take_profit),
        )
        if value is None
    ]
    if missing:
        raise SignalValidationError(
            f"{decision} requires {', '.join(sorted(missing))}"
        )

    entry = to_price(analysis.entry_price, "entry_price")
    stop_loss = to_price(analysis.stop_loss, "stop_loss")
    take_profit = to_price(analysis.take_profit, "take_profit")
    validate_signal_prices(decision, entry, stop_loss, take_profit)

    confidence = validate_confidence(analysis.confidence)
    candle_close = (
        to_price(analysis.candle_close_price, "candle_close_price")
        if analysis.candle_close_price is not None
        else None
    )

    return SignalCandidate(
        decision=decision,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        reasoning=analysis.reasoning,
        provider=analysis.provider,
        model_name=analysis.model,
        temperature=analysis.temperature,
        symbol=analysis.symbol,
        timeframe=analysis.timeframe,
        analysis_timestamp=analysis.analysis_timestamp,
        market_timestamp=analysis.market_timestamp,
        candle_close_price=candle_close,
    )
