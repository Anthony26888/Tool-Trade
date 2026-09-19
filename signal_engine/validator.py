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
- Phase B trade guardrails (policy, tunable via ``GuardrailConfig``/env):
  confidence must reach ``BTCUSDT_MIN_CONFIDENCE``; risk-reward must reach
  ``BTCUSDT_MIN_RISK_REWARD``; the take-profit distance must cover round-trip
  fees (``BTCUSDT_FEE_RATE``); when the analysis carries the ATR snapshot,
  stop/take distances and entry proximity must sit inside sane ATR multiples.
- An invalid LONG/SHORT is never downgraded to WAIT.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from database.database import to_decimal, validate_signal_prices
from database.models import SignalValidationError

from .analysis import DECISIONS, SignalAnalysis

_CONFIDENCE_MIN = 0
_CONFIDENCE_MAX = 100

#: Env tunables for the Phase B trade guardrails (0 disables that check,
#: which is also how Phase C replays the unguarded baseline for free).
ENV_MIN_CONFIDENCE = "BTCUSDT_MIN_CONFIDENCE"
ENV_MIN_RISK_REWARD = "BTCUSDT_MIN_RISK_REWARD"
ENV_FEE_RATE = "BTCUSDT_FEE_RATE"
ENV_PENDING_EXPIRY_HOURS = "BTCUSDT_PENDING_EXPIRY_HOURS"
ENV_MAX_FUNDING_RATE = "BTCUSDT_MAX_FUNDING_RATE"
ENV_HTF_BIAS = "BTCUSDT_HTF_BIAS"

DEFAULT_MIN_CONFIDENCE = 60
DEFAULT_MIN_RISK_REWARD = Decimal("1.2")
DEFAULT_FEE_RATE = Decimal("0.0004")
DEFAULT_PENDING_EXPIRY_HOURS = 6.0
#: Crowded-side funding cap (fraction per 8h): a LONG is rejected while
#: funding exceeds +cap (overcrowded longs), a SHORT while funding is below
#: -cap. 0 disables. Typical BTC funding is ~0.0001; 0.0005+ is crowded.
DEFAULT_MAX_FUNDING_RATE = Decimal("0.0005")
#: 4H regime veto (1 = on, 0 = prompt-only): LONG is rejected while the 4H
#: regime reads DOWN, SHORT while it reads UP. Unclear/unavailable never vetoes.
DEFAULT_HTF_BIAS = 1

#: Structural ATR multiples (constants; rarely need tuning).
_ATR_SL_MIN = Decimal("0.5")
_ATR_SL_MAX = Decimal("5")
_ATR_TP_MIN = Decimal("0.5")
_ATR_TP_MAX = Decimal("8")
_ENTRY_PROXIMITY_ATR = Decimal("1.5")
#: Take profit must cover round-trip fees (both sides). This is the exact
#: minimum (no buffer): the trade must at least pay for itself.
_FEE_COVER_MULT = Decimal("2")


def _env_int(source: Mapping[str, Any], name: str, default: int) -> int:
    try:
        return int(str(source.get(name, default)).strip())
    except (TypeError, ValueError, AttributeError):
        return default


def _env_decimal(source: Mapping[str, Any], name: str, default: Decimal) -> Decimal:
    try:
        value = Decimal(str(source.get(name, default)).strip())
    except (InvalidOperation, ValueError, TypeError, AttributeError):
        return default
    return value if value.is_finite() else default


def _env_float(source: Mapping[str, Any], name: str, default: float) -> float:
    try:
        value = float(str(source.get(name, default)).strip())
    except (TypeError, ValueError, AttributeError):
        return default
    return value if value == value and abs(value) != float("inf") else default


@dataclass(frozen=True)
class GuardrailConfig:
    """Tunable Phase B trade guardrails. Zero disables a check.

    ``min_confidence`` is compared against the half-up integer confidence;
    ``min_risk_reward`` against take-profit/stop-loss distance ratio;
    ``fee_rate`` against the take-profit distance covering round-trip fees;
    ``pending_expiry_hours`` caps how long a PENDING_ENTRY may wait (0 keeps
    it forever, as before Phase B); ``max_funding_rate`` rejects entries
    leaning into an overcrowded side (0 disables, None funding skips);
    ``htf_bias`` vetoes counter-regime entries on a clear 4H regime
    (0 = prompt-only, unclear regime never vetoes).
    """

    min_confidence: int = DEFAULT_MIN_CONFIDENCE
    min_risk_reward: Decimal = DEFAULT_MIN_RISK_REWARD
    fee_rate: Decimal = DEFAULT_FEE_RATE
    pending_expiry_hours: float = DEFAULT_PENDING_EXPIRY_HOURS
    max_funding_rate: Decimal = DEFAULT_MAX_FUNDING_RATE
    htf_bias: int = DEFAULT_HTF_BIAS

    @classmethod
    def from_env(cls, env: Mapping[str, Any] | None = None) -> GuardrailConfig:
        """Build from ``BTCUSDT_*`` env vars (defaults keep live behavior safe)."""
        source = os.environ if env is None else env
        return cls(
            min_confidence=_env_int(source, ENV_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE),
            min_risk_reward=_env_decimal(source, ENV_MIN_RISK_REWARD, DEFAULT_MIN_RISK_REWARD),
            fee_rate=_env_decimal(source, ENV_FEE_RATE, DEFAULT_FEE_RATE),
            pending_expiry_hours=_env_float(
                source, ENV_PENDING_EXPIRY_HOURS, DEFAULT_PENDING_EXPIRY_HOURS
            ),
            max_funding_rate=_env_decimal(
                source, ENV_MAX_FUNDING_RATE, DEFAULT_MAX_FUNDING_RATE
            ),
            htf_bias=_env_int(source, ENV_HTF_BIAS, DEFAULT_HTF_BIAS),
        )


def default_guardrails(env: Mapping[str, Any] | None = None) -> GuardrailConfig:
    """Resolve the active guardrails (env vars, read per call, no caching)."""
    return GuardrailConfig.from_env(env)


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


def validate_funding_crowd(
    decision: str, funding_rate: float | None, guards: GuardrailConfig
) -> None:
    """Reject entries leaning into an overcrowded side (Phase P1).

    Positive funding means longs pay shorts (overcrowded longs): LONG is
    rejected while funding exceeds +``max_funding_rate``; negative funding
    mirrors for SHORT. A zero cap disables; ``None`` funding (feed
    unavailable) skips the check so a dead feed never blocks trading.
    """
    cap = guards.max_funding_rate
    if cap is None or cap <= 0 or funding_rate is None:
        return
    try:
        rate = Decimal(str(funding_rate))
    except (InvalidOperation, ValueError, TypeError):
        return
    if not rate.is_finite():
        return
    if decision == "LONG" and rate > cap:
        raise SignalValidationError(
            f"LONG rejected: funding rate {rate} exceeds crowded-long cap {cap}"
        )
    if decision == "SHORT" and rate < -cap:
        raise SignalValidationError(
            f"SHORT rejected: funding rate {rate} below crowded-short cap -{cap}"
        )


def validate_htf_bias(
    decision: str, regime: str | None, guards: GuardrailConfig
) -> None:
    """Veto entries against a clear 4H regime (Phase P2, soft version).

    LONG is rejected while the regime reads DOWN, SHORT while it reads UP.
    A disabled knob (``htf_bias = 0``) or an unclear/unavailable regime
    (anything but UP/DOWN) never vetoes: structure and ADX stay AI context.
    """
    if not guards.htf_bias or regime not in ("UP", "DOWN"):
        return
    if decision == "LONG" and regime == "DOWN":
        raise SignalValidationError(
            "LONG rejected: 4H regime is DOWN (counter-trend entry)"
        )
    if decision == "SHORT" and regime == "UP":
        raise SignalValidationError(
            "SHORT rejected: 4H regime is UP (counter-trend entry)"
        )


def to_price(value, field: str) -> Decimal:
    """Exact, unambiguous Decimal conversion of a trading price.

    Delegates to the Phase 3 ``to_decimal`` rule (positive, finite, exact); the
    value is never rounded or quantized here.
    """
    return to_decimal(value, field)


def validate_trade_levels(
    decision: str,
    entry: Decimal,
    stop_loss: Decimal,
    take_profit: Decimal,
    *,
    atr: Decimal | None = None,
    close: Decimal | None = None,
    guardrails: GuardrailConfig | None = None,
) -> None:
    """Enforce Phase B trade-structure guardrails on validated Decimals.

    Ordering must already hold (strict inequalities, so both distances are
    positive). ATR-bound checks are skipped when no ATR snapshot is attached;
    every other check always applies. Raises ``SignalValidationError``.
    """
    guards = guardrails if guardrails is not None else default_guardrails()
    sl_dist = abs(entry - stop_loss)
    tp_dist = abs(take_profit - entry)
    if guards.min_risk_reward > 0:
        risk_reward = tp_dist / sl_dist
        if risk_reward < guards.min_risk_reward:
            raise SignalValidationError(
                f"risk-reward {risk_reward:.2f} below minimum {guards.min_risk_reward}"
            )
    if guards.fee_rate > 0:
        min_tp_dist = entry * guards.fee_rate * _FEE_COVER_MULT
        if tp_dist < min_tp_dist:
            raise SignalValidationError(
                f"take-profit distance {tp_dist} does not cover round-trip "
                f"fees (minimum {min_tp_dist})"
            )
    if atr is not None and atr > 0:
        if not (_ATR_SL_MIN * atr <= sl_dist <= _ATR_SL_MAX * atr):
            raise SignalValidationError(
                f"stop-loss distance {sl_dist} outside "
                f"[{_ATR_SL_MIN}*ATR, {_ATR_SL_MAX}*ATR] (ATR={atr})"
            )
        if not (_ATR_TP_MIN * atr <= tp_dist <= _ATR_TP_MAX * atr):
            raise SignalValidationError(
                f"take-profit distance {tp_dist} outside "
                f"[{_ATR_TP_MIN}*ATR, {_ATR_TP_MAX}*ATR] (ATR={atr})"
            )
        if close is not None and abs(entry - close) > _ENTRY_PROXIMITY_ATR * atr:
            raise SignalValidationError(
                f"entry {entry} is stale: more than {_ENTRY_PROXIMITY_ATR}*ATR "
                f"from last close {close} (ATR={atr})"
            )


def validate_analysis(
    analysis: SignalAnalysis, *, guardrails: GuardrailConfig | None = None
) -> SignalCandidate:
    """Validate a Phase 5 analysis and return an exact, persistence-ready candidate.

    Raises:
        SignalValidationError: for any structural, numeric, ordering, or
        guardrail problem.
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
    guards = guardrails if guardrails is not None else default_guardrails()
    if confidence < guards.min_confidence:
        raise SignalValidationError(
            f"confidence {confidence} below minimum {guards.min_confidence}"
        )
    validate_funding_crowd(decision, analysis.funding_rate, guards)
    validate_htf_bias(decision, analysis.regime, guards)
    candle_close = (
        to_price(analysis.candle_close_price, "candle_close_price")
        if analysis.candle_close_price is not None
        else None
    )
    atr: Decimal | None = None
    if analysis.atr is not None:
        try:
            candidate_atr = Decimal(str(analysis.atr))
        except (InvalidOperation, ValueError, TypeError):
            candidate_atr = None
        if candidate_atr is not None and candidate_atr.is_finite() and candidate_atr > 0:
            atr = candidate_atr
    validate_trade_levels(
        decision,
        entry,
        stop_loss,
        take_profit,
        atr=atr,
        close=candle_close,
        guardrails=guards,
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
