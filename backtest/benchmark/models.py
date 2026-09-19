"""Shared value types for the Phase 13 local-model benchmark.

A :class:`ModelDecision` is the authoritative per-candle record of what a local
LLM produced for one eligible candle: status, direction, prices, latency, and
cache metadata. It is serialized one line per candle into ``decisions.jsonl``,
which doubles as the replay/cache store: a re-run with the same configuration
scope replays recorded decisions instead of re-calling the LLM.

Status vocabulary
-----------------
- ``OK``      the model returned a structurally valid LONG/SHORT/WAIT.
- ``INVALID`` the model returned a rule-invalid LONG/SHORT (e.g. ``entry`` on
  the wrong side of ``stop_loss``); the backtest engine rejects it and no
  position is created. An ``INVALID`` output is never downgraded to WAIT.
- ``FAILED``  the LLM invocation/parse failed. The benchmark provider substitutes
  a deliberately invalid LONG (``entry == stop``) so the engine records an
  explicit rejected event and keeps running; a failure is never reported as the
  model saying WAIT, and never becomes a position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Bump when the decision context/prompt changes so stale cached decisions are
#: never reused. Keys are scoped to this version.
#: v2: Phase N appends the scheduled-event note section to the context.
CONTEXT_VERSION = 2

#: Cap the reasoning text persisted per decision so files stay readable.
_REASONING_CAP = 4000


class DecisionStatus:
    """Accepted ``ModelDecision.status`` values."""

    OK = "OK"
    INVALID = "INVALID"
    FAILED = "FAILED"


#: Short display labels for the two local benchmark models.
MODEL_LABELS: dict[str, str] = {
    "qwen3:4b": "Qwen3 4B",
    "gemma3:4b": "Gemma 4B",
}


def model_label(model: str) -> str:
    """Human-readable label for a model id, defaulting to a raw id."""
    return MODEL_LABELS.get(model, model)


def model_slug(model: str) -> str:
    """Filesystem-safe slug for a model id (e.g. ``qwen3:4b`` -> ``qwen3_4b``)."""
    return "".join(ch if ch.isalnum() else "_" for ch in model).strip("_") or "model"


#: Confidence buckets for the signal-quality distribution.
CONFIDENCE_BUCKETS = ((0, 40), (40, 70), (70, 90), (90, 101))


def bucket_for_confidence(confidence: float) -> str:
    """Return the bucket label a confidence value falls into."""
    if confidence < 0 or confidence > 100:  # pragma: no cover - defensive
        raise ValueError(f"confidence {confidence} out of range [0, 100]")
    for low, high in CONFIDENCE_BUCKETS:
        if low <= confidence < high:
            return f"{low}-{high-1}"
    return "90-100"  # pragma: no cover - defensive


@dataclass(frozen=True)
class ModelDecision:
    """One recorded decision from a benchmark model run."""

    model: str
    provider: str
    candle_ts: int
    candle_close: float
    status: str
    decision: str | None
    confidence: float | None
    reasoning: str
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    latency_sec: float
    cached: bool
    scope: str
    context_version: int
    analysis_timestamp: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if len(self.reasoning) > _REASONING_CAP:
            reasoning = self.reasoning[:_REASONING_CAP]
        else:
            reasoning = self.reasoning
        return {
            "context_version": self.context_version,
            "scope": self.scope,
            "model": self.model,
            "provider": self.provider,
            "candle_ts": self.candle_ts,
            "candle_close": self.candle_close,
            "analysis_timestamp": self.analysis_timestamp,
            "status": self.status,
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": reasoning,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "latency_sec": self.latency_sec,
            "cached": self.cached,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> ModelDecision:
        return cls(
            model=str(mapping["model"]),
            provider=str(mapping["provider"]),
            candle_ts=int(mapping["candle_ts"]),
            candle_close=float(mapping["candle_close"]),
            status=str(mapping["status"]),
            decision=_nullable_str(mapping.get("decision")),
            confidence=_nullable_float(mapping.get("confidence")),
            reasoning=str(mapping.get("reasoning") or ""),
            entry_price=_nullable_float(mapping.get("entry_price")),
            stop_loss=_nullable_float(mapping.get("stop_loss")),
            take_profit=_nullable_float(mapping.get("take_profit")),
            latency_sec=float(mapping.get("latency_sec") or 0.0),
            cached=bool(mapping.get("cached", False)),
            scope=str(mapping.get("scope") or ""),
            context_version=int(mapping.get("context_version") or CONTEXT_VERSION),
            analysis_timestamp=str(mapping.get("analysis_timestamp") or ""),
            error=mapping.get("error"),
        )


def _nullable_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _nullable_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass(frozen=True)
class InferenceStats:
    """Latency statistics for the fresh (non-cached) LLM calls of one run."""

    total_sec: float = 0.0
    min_sec: float | None = None
    max_sec: float | None = None
    n_calls: int = 0
    n_cached: int = 0
    n_failed: int = 0
    load_time_sec: float = 0.0
    median_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_time_sec": round(self.load_time_sec, 6),
            "fresh_calls": self.n_calls,
            "cached_hits": self.n_cached,
            "failed_calls": self.n_failed,
            "total_sec": round(self.total_sec, 6),
            "median_sec": None if self.median_sec is None else round(self.median_sec, 6),
            "min_sec": None if self.min_sec is None else round(self.min_sec, 6),
            "max_sec": None if self.max_sec is None else round(self.max_sec, 6),
            "avg_sec": (
                round(self.total_sec / self.n_calls, 6) if self.n_calls else None
            ),
        }


class RunningInferenceStats:
    """Mutable accumulator producing an :class:`InferenceStats` snapshot."""

    def __init__(self) -> None:
        self._latencies: list[float] = []
        self._cached = 0
        self._failed = 0
        self._load_time = 0.0

    def add(self, latency_sec: float, *, cached: bool, failed: bool) -> None:
        if cached:
            self._cached += 1
            return
        self._latencies.append(latency_sec)
        if failed:
            self._failed += 1

    def set_load_time(self, value: float) -> None:
        self._load_time = value

    def snapshot(self) -> InferenceStats:
        latencies = sorted(self._latencies)
        n = len(latencies)
        median = latencies[n // 2] if n else None
        return InferenceStats(
            total_sec=sum(latencies),
            min_sec=latencies[0] if n else None,
            max_sec=latencies[-1] if n else None,
            n_calls=n,
            n_cached=self._cached,
            n_failed=self._failed,
            load_time_sec=self._load_time,
            median_sec=median,
        )


__all__ = [
    "CONFIDENCE_BUCKETS",
    "CONTEXT_VERSION",
    "DecisionStatus",
    "InferenceStats",
    "ModelDecision",
    "MODEL_LABELS",
    "RunningInferenceStats",
    "bucket_for_confidence",
    "model_label",
    "model_slug",
]
