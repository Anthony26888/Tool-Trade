"""Decision cache / replay store for the Phase 13 benchmark.

One JSON-lines file per model holds every recorded :class:`ModelDecision`,
ordered by decision candle. The file is consumed as a cache (a re-run with the
same configuration scope reuses records instead of re-calling the LLM) and, at
the same time, is the exported ``decisions.jsonl`` artifact.

Scope rules (no cross-model leakage)
------------------------------------
Each record's ``scope`` is a deterministic hash of the model, its resolved LLM
configuration, the prompt-affecting context inputs (symbol, timeframe,
``max_candles``), and ``context_version``. The cache key is the scope plus the
candle open time, so:

- a decision produced by ``qwen3:4b`` can never be replayed for ``gemma3:4b``;
- a decision produced with ``max_candles=40`` is never reused when
  ``max_candles=80`` (the prompt would differ);
- bumping ``CONTEXT_VERSION`` invalidates all previously cached prompts.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from typing import Any

from .models import CONTEXT_VERSION, ModelDecision

_SCOPE_SEPARATOR = "|"


def scope_hash(parts: Iterable[Any]) -> str:
    """Deterministic hex digest of the parts that define a decision's prompt."""
    joined = _SCOPE_SEPARATOR.join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def compute_scope(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    temperature: float | None,
    timeout: float | None,
    max_retries: int | None,
    max_tokens: int | None,
    symbol: str,
    timeframe: str,
    max_candles: int,
    context_version: int = CONTEXT_VERSION,
    dataset_id: str | None = None,
) -> str:
    """Compute the cache scope for a resolved model configuration.

    ``dataset_id`` is an optional caller-supplied discriminant (e.g. the data
    file path or run label); when present it is part of the scope so identical
    models/configs over different datasets never share cached decisions.
    """
    return scope_hash(
        (
            provider.lower(),
            model,
            base_url,
            temperature,
            timeout,
            max_retries,
            max_tokens,
            symbol.upper(),
            timeframe,
            int(max_candles),
            int(context_version),
            dataset_id,
        )
    )


def cache_key(scope: str, candle_ts: int) -> str:
    """The exact lookup key for one candle within a scope."""
    return f"{scope}::{candle_ts}"


class DecisionCache:
    """Append-only JSONL cache of model decisions for one benchmark run.

    Thread-safety is not a concern: the Phase 12 engine drives decisions
    sequentially. The file is only opened for the duration of each append, so a
    crash mid-run never corrupts already-recorded decisions.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = os.fspath(path)
        self._records: dict[str, ModelDecision] = {}

    @property
    def path(self) -> str:
        return self._path

    def load(self) -> None:
        """Load existing records into memory (missing/corrupt file => empty)."""
        self._records = {}
        try:
            with open(self._path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except FileNotFoundError:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = ModelDecision.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # A malformed trailing line must not invalidate the run; it
                # is ignored so prior good records stay usable.
                continue
            key = cache_key(record.scope, record.candle_ts)
            self._records[key] = record

    def __len__(self) -> int:
        return len(self._records)

    def get(self, scope: str, candle_ts: int) -> ModelDecision | None:
        return self._records.get(cache_key(scope, candle_ts))

    def put(self, record: ModelDecision) -> None:
        """Append a record to the file and keep it in memory."""
        self._records[cache_key(record.scope, record.candle_ts)] = record
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False))
            handle.write("\n")

    def records(self) -> tuple[ModelDecision, ...]:
        """All records in insertion order (decision candle order)."""
        return tuple(self._records.values())

    def _prepare_fresh(self) -> None:
        """Start a fresh run: truncate the file and reset memory state."""
        self._records = {}
        with open(self._path, "w", encoding="utf-8") as handle:
            handle.write("")


__all__ = [
    "DecisionCache",
    "cache_key",
    "compute_scope",
    "scope_hash",
]
