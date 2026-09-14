"""Phase 13: sequential local-model benchmark for the BTCUSDT Signal Engine.

One benchmark run uses ONE local model (e.g. ``qwen3:4b`` or ``gemma3:4b``) over
a single dataset/configuration, strictly sequentially, reusing the identical
closed-candle prompt/context for every model. The comparison step only lays the
metrics side by side for a human decision — no automatic winner is ever
selected, and production configuration is never modified.
"""

from __future__ import annotations

from .cache import DecisionCache, cache_key, compute_scope, scope_hash
from .compare import ComparisonError, compare_summaries, write_comparison
from .models import (
    CONFIDENCE_BUCKETS,
    CONTEXT_VERSION,
    MODEL_LABELS,
    DecisionStatus,
    InferenceStats,
    ModelDecision,
    RunningInferenceStats,
    bucket_for_confidence,
    model_label,
    model_slug,
)
from .provider import (
    AnalyzeFn,
    BenchmarkConfigError,
    LLMDecisionProvider,
)
from .runner import (
    BenchmarkError,
    RunSummary,
    build_llm_decision_provider,
    build_signal_quality,
    build_summary,
    run_benchmark,
    verify_llm_temperature,
)

__all__ = [
    "AnalyzeFn",
    "BenchmarkConfigError",
    "BenchmarkError",
    "CONFIDENCE_BUCKETS",
    "CONTEXT_VERSION",
    "ComparisonError",
    "DecisionCache",
    "DecisionStatus",
    "InferenceStats",
    "LLMDecisionProvider",
    "MODEL_LABELS",
    "ModelDecision",
    "RunSummary",
    "RunningInferenceStats",
    "bucket_for_confidence",
    "build_llm_decision_provider",
    "build_signal_quality",
    "build_summary",
    "cache_key",
    "compare_summaries",
    "compute_scope",
    "model_label",
    "model_slug",
    "run_benchmark",
    "scope_hash",
    "verify_llm_temperature",
    "write_comparison",
]
