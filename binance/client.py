"""Binance USDT-M Futures public market-data HTTP client.

Phase 1 scope: PUBLIC market data only. No order execution and no API
credentials are implemented here. This module is kept isolated from the
TradingAgents vendor/yfinance dataflows.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import requests

logger = logging.getLogger(__name__)

# https://developers.binance.com/docs/derivatives/usds-margined-futures
DEFAULT_BASE_URL = "https://fapi.binance.com"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 0.5

_BACKOFF_MULTIPLIER = 2.0
_MAX_BACKOFF_SECONDS = 10.0

# Only transient failures are retried: rate limit (429) and server errors (5xx).
_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


class BinanceError(Exception):
    """Base class for Binance market-data errors."""


class BinanceConnectionError(BinanceError):
    """Network or timeout failure while talking to the Binance API."""


class BinanceHTTPError(BinanceError):
    """The API returned a non-2xx, non-retryable HTTP status."""

    def __init__(self, status_code: int, body: str):
        self.status_code = int(status_code)
        self.body = body
        super().__init__(f"Binance API returned HTTP {self.status_code}: {body!r}")


class BinanceRateLimitError(BinanceHTTPError):
    """HTTP 429 — request rejected for rate limiting (retryable)."""


class BinanceServerError(BinanceHTTPError):
    """HTTP 5xx — transient Binance server failure (retryable)."""


def _body(response) -> str:
    text = getattr(response, "text", "") or ""
    return str(text)[:200]


class BinanceFuturesClient:
    """Low-level HTTP client for Binance USDT-M Futures public endpoints.

    Public market-data endpoints need no API key. Requests carry a timeout and
    are retried (with capped exponential backoff) on transient failures only;
    non-retryable 4xx responses raise immediately.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff: float = DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        """GET ``path`` with retries on transient failures; returns parsed JSON.

        Raises ``BinanceConnectionError``, ``BinanceRateLimitError``,
        ``BinanceServerError``, or ``BinanceHTTPError`` depending on the failure,
        after any configured retries are exhausted.
        """
        url = f"{self.base_url}{path}"
        last_exc: BinanceError | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = BinanceConnectionError(str(exc))
                logger.warning(
                    "Binance request failed (attempt %d/%d): %s",
                    attempt + 1, attempts, last_exc,
                )
                self._backoff(attempt)
                continue

            if response.status_code in _RETRYABLE_STATUS_CODES:
                if response.status_code == 429:
                    last_exc = BinanceRateLimitError(response.status_code, _body(response))
                else:
                    last_exc = BinanceServerError(response.status_code, _body(response))
                logger.warning(
                    "Binance retryable HTTP %s (attempt %d/%d): %s",
                    response.status_code, attempt + 1, attempts, last_exc,
                )
                self._backoff(attempt)
                continue

            if response.status_code != 200:
                raise BinanceHTTPError(response.status_code, _body(response))

            try:
                return response.json()
            except ValueError as exc:
                raise BinanceError(
                    f"Binance returned a non-JSON body: {_body(response)!r}"
                ) from exc

        assert last_exc is not None
        raise last_exc

    def _backoff(self, attempt: int) -> None:
        delay = min(self.backoff * (_BACKOFF_MULTIPLIER ** attempt), _MAX_BACKOFF_SECONDS)
        time.sleep(delay)
