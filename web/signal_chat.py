"""Signal Chat: read-only Q&A about the currently active BTCUSDT signal.

The dashboard drawer lets the user ask the configured LLM why the active
signal was chosen, what the strategy is, and what the risk looks like. This
module is strictly display-only: it never writes to the database and never
touches trading state, the AI lock, or the TP/SL monitor (AGENTS.md sections
3-9, 27).

Answers are produced as an iterator of text chunks so the web layer can stream
them to the drawer (newline-delimited JSON) instead of waiting for a full
response — a local 4B model on CPU can take many seconds to finish a reply.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from signal_engine.llm import LLMConfig, build_llm_client

SYSTEM_PROMPT = """\
You are Signal Chat inside the NEXTRA AI BTCUSDT signal-engine dashboard. A \
user is asking about the CURRENTLY ACTIVE trading signal that the engine \
produced on a closed 1H candle. Act as a clear, honest technical analyst \
explaining things to a non-expert trader: what the strategy is, why this \
signal was chosen, and what the risk is.

STRICT RULES:
- Use ONLY the facts in the provided SIGNAL CONTEXT. Never invent prices, \
indicators, news, timeframes, or numbers that are not in the context.
- If you lack the information to answer, say so explicitly.
- This is display-only: never advise placing real orders, moving entry/TP/SL, \
or changing the running position.
- Reply in the SAME LANGUAGE as the user question (a Vietnamese question must \
be answered in Vietnamese).
- Be concise (a few sentences) unless the user asks for more detail.
"""

CANNED_NO_SIGNAL = (
    "There is no active signal to explain yet. The daemon analyses BTCUSDT "
    "on each closed 1H candle and publishes a signal here when one is created."
)

MAX_QUESTION_CHARS = 1000


def _json_default(value: Any) -> Any:
    """Keep Decimals/quantized values machine-readable in the prompt."""
    return str(value)


def build_messages(question: str, context: dict[str, Any]) -> list[dict[str, str]]:
    """Build the system + user prompt pair for the signal chat.

    The full serialized context is embedded verbatim so the model can only see
    facts that actually exist in the ledger (AGENTS.md sections 5, 7, 27).
    """
    payload = json.dumps(context, default=_json_default, sort_keys=True)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"QUESTION: {question}\n\nSIGNAL CONTEXT (JSON):\n{payload}"},
    ]


def iter_text_chunks(chunks: Iterator[Any]) -> Iterator[str]:
    """Normalize langchain stream chunks into plain text pieces."""
    for chunk in chunks or []:
        content = getattr(chunk, "content", None)
        if content is None:
            continue
        if isinstance(content, str):
            if content:
                yield content
        else:
            yield str(content)


def stream_answer(messages: list[dict[str, str]], config: LLMConfig) -> Iterator[str]:
    """Stream the LLM answer as text chunks, falling back to a full reply.

    ``ChatOpenAI`` supports ``.stream``; if a provider client does not expose
    it (``AttributeError``/``TypeError``) or rejects it, fall back to a single
    non-streaming ``invoke`` so the endpoint still works everywhere.
    """
    llm = build_llm_client(config).get_llm()
    try:
        stream = llm.stream(messages)
        yield from iter_text_chunks(stream)
    except (AttributeError, TypeError, NotImplementedError):
        response = llm.invoke(messages)
        content = getattr(response, "content", "") or ""
        if content:
            yield str(content)


__all__ = [
    "CANNED_NO_SIGNAL",
    "MAX_QUESTION_CHARS",
    "SYSTEM_PROMPT",
    "build_messages",
    "iter_text_chunks",
    "stream_answer",
]
