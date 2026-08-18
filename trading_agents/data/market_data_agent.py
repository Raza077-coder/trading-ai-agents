"""Market Data Agent — fetches prices, volumes and quotes from pluggable providers.

The agent abstracts away provider specifics behind a tiny interface, caches
responses, and provides both bulk (candles) and point-in-time (quote) access.
"""

from __future__ import annotations

import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from trading_agents.core.base import BaseAgent
from trading_agents.core.config import AgentConfig, DataConfig
from trading_agents.core.exceptions import DataFetchError, InvalidSymbolError
from trading_agents.core.models import Candle, Quote
from trading_agents.data.providers import BaseProvider, create_provider

# Symbols with letters/digits/hyphens/dots only (loose sanity check)
_VALID_SYMBOL_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-=^_")


class MarketDataAgent(BaseAgent):
    """Fetches market data from yfinance / Alpha Vantage / synthetic providers.

    Example:
        >>> with MarketDataAgent() as agent:
        ...     candles = agent.get_candles("AAPL", period="6mo")
        ...     quote = agent.get_quote("AAPL")
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        data_config: Optional[DataConfig] = None,
        provider: Optional[BaseProvider] = None,
    ) -> None:
        super().__init__("market_data", config)
        self.data_config = data_config or DataConfig()
        self.provider = provider or create_provider(
            self.data_config.provider,
            api_key=self.data_config.api_key,
            retries=self.data_config.retries,
            backoff_factor=self.data_config.backoff_factor,
        )
        self._cache: Dict[Tuple[str, str, str], Tuple[float, List[Candle]]] = {}
        self._quote_cache: Dict[str, Tuple[float, Quote]] = {}
        self._load_disk_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_candles(
        self,
        symbol: str,
        period: str | None = None,
        interval: str | None = None,
        use_cache: bool = True,
    ) -> List[Candle]:
        """Return OHLCV candles for a symbol.

        Args:
            symbol: Ticker symbol, e.g. ``AAPL``.
            period: Lookback period (``1mo``/``3mo``/``6mo``/``1y``/``max``).
            interval: Bar interval (``1d``/``1wk``/``5m``/...).
            use_cache: Use in-memory + disk cache (TTL from config).

        Raises:
            InvalidSymbolError: If the symbol looks malformed.
            DataFetchError: If the provider cannot return data.
        """
        self._guard_initialized()
        symbol = self._validate_symbol(symbol)
        period = period or self.data_config.default_period
        interval = interval or self.data_config.default_interval

        key = (symbol, period, interval)
        if use_cache and key in self._cache:
            fetched_at, candles = self._cache[key]
            if time.time() - fetched_at < self.config.cache_ttl_seconds:
                self.logger.debug("Cache hit for %s (%s/%s)", symbol, period, interval)
                return candles

        self.logger.info("Fetching candles for %s (%s/%s)", symbol, period, interval)
        candles = self.provider.get_candles(symbol, period=period, interval=interval)
        if len(candles) < 2:
            raise DataFetchError(f"Insufficient data returned for {symbol}")
        if use_cache:
            self._cache[key] = (time.time(), candles)
            self._save_disk_cache()
        return candles

    def get_quote(self, symbol: str, use_cache: bool = True) -> Quote:
        """Return the latest quote for a symbol."""
        self._guard_initialized()
        symbol = self._validate_symbol(symbol)
        if use_cache and symbol in self._quote_cache:
            fetched_at, quote = self._quote_cache[symbol]
            if time.time() - fetched_at < self.config.cache_ttl_seconds:
                return quote
        quote = self.provider.get_quote(symbol)
        if use_cache:
            self._quote_cache[symbol] = (time.time(), quote)
        return quote

    def get_many_quotes(self, symbols: List[str], use_cache: bool = True) -> Dict[str, Quote]:
        """Fetch quotes for several symbols, tolerating per-symbol failures."""
        results: Dict[str, Quote] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_quote(symbol, use_cache=use_cache)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Quote failed for %s: %s", symbol, exc)
        return results

    def health_check(self) -> bool:
        """True if the configured provider can serve a quote."""
        try:
            return self.provider.health_check()
        except Exception:  # noqa: BLE001
            return False

    def shutdown(self) -> None:
        self._save_disk_cache()
        super().shutdown()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_symbol(symbol: str) -> str:
        symbol = symbol.strip().upper()
        if not symbol or not all(ch in _VALID_SYMBOL_CHARS for ch in symbol):
            raise InvalidSymbolError(f"Invalid symbol: {symbol!r}")
        return symbol

    def _disk_cache_path(self) -> Path:
        return Path(self.data_config.cache_dir) / "market_data.pkl"

    def _load_disk_cache(self) -> None:
        try:
            path = self._disk_cache_path()
            if path.exists():
                with open(path, "rb") as fh:
                    data = pickle.load(fh)
                self._cache.update(data.get("candles", {}))
                self._quote_cache.update(data.get("quotes", {}))
                self.logger.debug("Loaded disk cache from %s", path)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Could not load disk cache: %s", exc)

    def _save_disk_cache(self) -> None:
        try:
            path = self._disk_cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump(
                    {"candles": self._cache, "quotes": self._quote_cache}, fh
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Could not save disk cache: %s", exc)
