"""Data provider abstraction.

Providers implement a small common interface so the :class:`MarketDataAgent`
can swap between live sources (yfinance, Alpha Vantage) and a deterministic
synthetic generator for demos and offline testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional
import random

from trading_agents.core.exceptions import DataFetchError, ProviderError
from trading_agents.core.models import Candle, Quote


class BaseProvider(ABC):
    """Interface every market data provider implements."""

    name: str = "base"

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> List[Candle]:
        """Return OHLCV candles for a symbol."""
        raise NotImplementedError

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Return a live quote for a symbol."""
        raise NotImplementedError

    def health_check(self) -> bool:
        """Best-effort provider health check."""
        try:
            self.get_quote("AAPL")
            return True
        except Exception:  # noqa: BLE001
            return False


# ---------------------------------------------------------------------------
# yfinance provider
# ---------------------------------------------------------------------------


class YFinanceProvider(BaseProvider):
    """Live market data via the free ``yfinance`` library."""

    name = "yfinance"

    def __init__(self, retries: int = 3, backoff_factor: float = 0.5) -> None:
        self.retries = retries
        self.backoff_factor = backoff_factor
        self._yf = None
        try:  # lazy import so the rest of the suite works without yfinance
            import yfinance as yf  # type: ignore

            self._yf = yf
        except ImportError:
            self._yf = None

    def _require_yf(self) -> None:
        if self._yf is None:
            raise ProviderError(
                "yfinance is not installed — pip install yfinance, "
                "or set TA_DATA_PROVIDER=alpha_vantage|synthetic"
            )

    def get_candles(self, symbol: str, period: str = "1y", interval: str = "1d") -> List[Candle]:
        self._require_yf()
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                ticker = self._yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval, auto_adjust=True)
                if df is None or df.empty:
                    raise DataFetchError(f"No data returned for {symbol}")
                candles: List[Candle] = []
                for idx, row in df.iterrows():
                    ts = idx.to_pydatetime()
                    if ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)
                    candles.append(
                        Candle(
                            timestamp=ts,
                            open=float(row["Open"]),
                            high=float(row["High"]),
                            low=float(row["Low"]),
                            close=float(row["Close"]),
                            volume=float(row.get("Volume", 0.0) or 0.0),
                        )
                    )
                if not candles:
                    raise DataFetchError(f"No candles for {symbol}")
                return candles
            except ProviderError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    import time

                    time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise DataFetchError(f"yfinance failed for {symbol}: {last_error}")

    def get_quote(self, symbol: str) -> Quote:
        self._require_yf()
        try:
            ticker = self._yf.Ticker(symbol)
            info = ticker.fast_info
            price = float(info.last_price)
            prev = float(info.previous_close) if info.previous_close else price
            change = price - prev
            change_pct = (change / prev * 100.0) if prev else 0.0
            return Quote(
                symbol=symbol,
                price=price,
                change=change,
                change_pct=change_pct,
                volume=float(info.last_volume) if info.last_volume else None,
            )
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(f"Quote fetch failed for {symbol}: {exc}")


# ---------------------------------------------------------------------------
# Alpha Vantage provider
# ---------------------------------------------------------------------------


class AlphaVantageProvider(BaseProvider):
    """Live market data via the Alpha Vantage REST API (needs API key)."""

    name = "alpha_vantage"

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, retries: int = 3, backoff_factor: float = 0.5) -> None:
        if not api_key:
            raise ProviderError("Alpha Vantage requires an API key (TA_ALPHAVANTAGE_KEY)")
        self.api_key = api_key
        self.retries = retries
        self.backoff_factor = backoff_factor
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _get(self, params: dict) -> dict:
        import time

        params = {**params, "apikey": self.api_key}
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._get_session().get(self.BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                if "Error Message" in data:
                    raise DataFetchError(data["Error Message"])
                if "Note" in data:  # rate-limit note
                    raise ProviderError(f"Alpha Vantage rate limit: {data['Note']}")
                return data
            except ProviderError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise DataFetchError(f"Alpha Vantage request failed: {last_error}")

    def get_candles(self, symbol: str, period: str = "1y", interval: str = "1d") -> List[Candle]:
        function = (
            "TIME_SERIES_DAILY_ADJUSTED"
            if interval in ("1d", "1wk", "1mo")
            else "TIME_SERIES_INTRADAY"
        )
        data = self._get({"function": function, "symbol": symbol, "outputsize": "full"})
        if function == "TIME_SERIES_DAILY_ADJUSTED":
            series = data.get("Time Series (Daily)", {})
            rows = sorted(series.items(), reverse=True)
            cutoff = 365 if period in ("1y", "ytd", "max") else 180
            rows = rows[:cutoff]
            return [
                Candle(
                    timestamp=datetime.strptime(day, "%Y-%m-%d"),
                    open=float(v["1. open"]),
                    high=float(v["2. high"]),
                    low=float(v["3. low"]),
                    close=float(v["4. close"]),
                    volume=float(v["6. volume"]),
                )
                for day, v in rows
            ]
        series = data.get("Time Series (5min)", {})
        rows = sorted(series.items(), reverse=True)[:390]
        return [
            Candle(
                timestamp=datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"),
                open=float(v["1. open"]),
                high=float(v["2. high"]),
                low=float(v["3. low"]),
                close=float(v["4. close"]),
                volume=float(v["5. volume"]),
            )
            for ts, v in rows
        ]

    def get_quote(self, symbol: str) -> Quote:
        data = self._get({"function": "GLOBAL_QUOTE", "symbol": symbol})
        q = data.get("Global Quote", {})
        if not q:
            raise DataFetchError(f"No quote for {symbol}")
        price = float(q.get("05. price", 0.0))
        change = float(q.get("09. change", 0.0))
        change_pct = float(q.get("10. change percent", "0%").rstrip("%"))
        return Quote(
            symbol=symbol,
            price=price,
            change=change,
            change_pct=change_pct,
            volume=float(q["06. volume"]) if q.get("06. volume") else None,
        )


# ---------------------------------------------------------------------------
# Synthetic provider (deterministic, offline-safe)
# ---------------------------------------------------------------------------


class SyntheticProvider(BaseProvider):
    """Deterministic random-walk generator — perfect for demos and CI."""

    name = "synthetic"

    def __init__(self, seed: int = 42, base_price: Optional[float] = None) -> None:
        self.seed = seed
        self.base_price = base_price
        self._rng = random.Random(seed)

    def _prices(self, symbol: str, n: int) -> List[float]:
        # Deterministic per-symbol drift derived from the symbol hash
        seed = self.seed + sum(ord(c) for c in symbol)
        rng = random.Random(seed)
        start = self.base_price or (50.0 + rng.uniform(0, 450))
        drift = rng.uniform(-0.0004, 0.0009)
        vol = rng.uniform(0.008, 0.025)
        prices = [start]
        for _ in range(n - 1):
            ret = drift + rng.gauss(0, vol)
            prices.append(max(1.0, prices[-1] * (1 + ret)))
        return prices

    def get_candles(self, symbol: str, period: str = "1y", interval: str = "1d") -> List[Candle]:
        n = 252 if period in ("1y", "ytd") else {"1mo": 21, "3mo": 63, "6mo": 126, "max": 756}.get(period, 252)
        prices = self._prices(symbol, n)
        end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        step = timedelta(days=1)
        candles: List[Candle] = []
        for i, close in enumerate(prices):
            day_open = prices[i - 1] if i else close
            hi = max(day_open, close) * (1 + self._rng.uniform(0.001, 0.015))
            lo = min(day_open, close) * (1 - self._rng.uniform(0.001, 0.015))
            vol = self._rng.uniform(1e5, 5e7)
            candles.append(
                Candle(
                    timestamp=end - step * (n - 1 - i),
                    open=round(day_open, 4),
                    high=round(hi, 4),
                    low=round(lo, 4),
                    close=round(close, 4),
                    volume=round(vol, 2),
                )
            )
        return candles

    def get_quote(self, symbol: str) -> Quote:
        candles = self.get_candles(symbol, period="1mo")
        price = candles[-1].close
        prev = candles[-2].close if len(candles) > 1 else price
        change = price - prev
        return Quote(
            symbol=symbol,
            price=round(price, 4),
            change=round(change, 4),
            change_pct=round(change / prev * 100.0, 4) if prev else 0.0,
            volume=candles[-1].volume,
            timestamp=datetime.utcnow(),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_provider(
    name: str,
    api_key: Optional[str] = None,
    retries: int = 3,
    backoff_factor: float = 0.5,
    seed: int = 42,
) -> BaseProvider:
    """Instantiate a provider by name.

    Args:
        name: One of ``yfinance``, ``alpha_vantage``, ``synthetic``.
        api_key: Required for Alpha Vantage.
        retries/backoff_factor: Network retry policy.
        seed: Random seed for the synthetic provider.

    Raises:
        ProviderError: On unknown provider names or missing API keys.
    """
    name = name.lower().replace("-", "_")
    if name in ("yfinance", "yf", "yahoo"):
        return YFinanceProvider(retries=retries, backoff_factor=backoff_factor)
    if name in ("alpha_vantage", "alphavantage", "av"):
        if not api_key:
            raise ProviderError(
                "Alpha Vantage provider requires TA_ALPHAVANTAGE_KEY to be set"
            )
        return AlphaVantageProvider(api_key=api_key, retries=retries, backoff_factor=backoff_factor)
    if name in ("synthetic", "sim", "mock"):
        return SyntheticProvider(seed=seed)
    raise ProviderError(f"Unknown data provider: {name!r}")
