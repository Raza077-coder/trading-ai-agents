"""Sentiment Analysis Agent — scores news headlines and social chatter.

Uses a finance-aware lexicon (Loughran-McDonald inspired word lists) with
negation handling and intensity weighting. Optionally plugs into an LLM for
deeper contextual scoring (``TA_SENTIMENT_USE_LLM=true``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from trading_agents.core.base import BaseAgent
from trading_agents.core.config import AgentConfig, SentimentConfig
from trading_agents.core.models import (
    NewsItem,
    SentimentLabel,
    SentimentScore,
)

# ---------------------------------------------------------------------------
# Finance-aware lexicon (Loughran-McDonald style, condensed)
# ---------------------------------------------------------------------------

_BULLISH = {
    "beat", "beats", "surge", "surges", "surged", "rally", "rallies", "rallied",
    "record", "records", "growth", "grow", "growing", "profit", "profits",
    "profitable", "gain", "gains", "gained", "upgrade", "upgrades", "upgraded",
    "outperform", "outperforms", "outperformed", "bullish", "buy", "strong",
    "stronger", "strength", "momentum", "breakout", "breakouts", "expansion",
    "expanding", "opportunity", "opportunities", "partnership", "partnerships",
    "launch", "launches", "launched", "approval", "approved", "win", "wins",
    "won", "soar", "soars", "soared", "jump", "jumps", "jumped", "climb",
    "climbs", "climbed", "positive", "optimistic", "outlook", "guidance",
    "raise", "raises", "raised", "upbeat", "breakthrough", "innovative",
    "exceed", "exceeds", "exceeded", "ahead", "tops", "topped", "hit", "hits",
}

_BEARISH = {
    "miss", "misses", "missed", "drop", "drops", "dropped", "plunge", "plunges",
    "plunged", "fall", "falls", "fell", "decline", "declines", "declined",
    "downgrade", "downgrades", "downgraded", "underperform", "underperforms",
    "bearish", "sell", "weak", "weaker", "weakness", "loss", "losses", "losing",
    "lawsuit", "lawsuits", "investigation", "investigations", "probe", "fraud",
    "scandal", "scandals", "layoff", "layoffs", "cut", "cuts", "cutting",
    "recession", "recessionary", "downturn", "slump", "slumps", "slumped",
    "tumble", "tumbles", "tumbled", "crash", "crashes", "crashed", "worst",
    "risk", "risks", "risky", "warning", "warnings", "warn", "warns", "warned",
    "negative", "pessimistic", "outlook", "guidance", "lower", "lowers",
    "lowered", "debt", "default", "defaults", "bankruptcy", "insolvency",
    "recall", "recalls", "recalled", "volatile", "volatility", "concern",
    "concerns", "uncertainty", "fear", "fears", "selloff", "sell-off",
    "correction", "bubble", "struggle", "struggles", "struggled",
}

_NEGATORS = {"not", "no", "never", "hardly", "barely", "without", "isn't", "aren't", "don't", "doesn't", "didn't", "won't", "can't", "cannot"}
_INTENSIFIERS = {"very", "extremely", "strongly", "highly", "significantly", "massively", "sharply", "clearly", "definitely", "greatly"}


@dataclass
class _Source:
    """A pluggable news source."""

    name: str
    url: str


class SentimentAnalysisAgent(BaseAgent):
    """Score market sentiment from headlines and optional social chatter.

    Example:
        >>> with SentimentAnalysisAgent() as agent:
        ...     score = agent.analyze_text("Apple beats earnings, stock surges")
        ...     agg = agent.aggregate([("AAPL", score)])
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        sentiment_config: Optional[SentimentConfig] = None,
    ) -> None:
        super().__init__("sentiment_analysis", config)
        self.sent_config = sentiment_config or SentimentConfig()
        self._sources: List[_Source] = [
            _Source(name="news_api", url="https://newsapi.org/v2/everything"),
        ]
        self._word_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_text(self, text: str, symbol: str = "GENERIC") -> SentimentScore:
        """Score a single headline/body of text in [-1, +1]."""
        score, pos, neg, neu = self._lexicon_score(text)
        label = (
            SentimentLabel.BULLISH
            if score > 0.15
            else SentimentLabel.BEARISH
            if score < -0.15
            else SentimentLabel.NEUTRAL
        )
        total = pos + neg + neu
        confidence = min(1.0, total / 12.0)  # saturating confidence
        return SentimentScore(
            symbol=symbol,
            score=round(score, 4),
            label=label,
            headline_count=1,
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            confidence=round(confidence, 3),
        )

    def analyze_news(self, items: List[NewsItem], symbol: str) -> SentimentScore:
        """Aggregate a list of :class:`NewsItem` into one sentiment score."""
        if not items:
            return SentimentScore(
                symbol=symbol, score=0.0, label=SentimentLabel.NEUTRAL, confidence=0.0
            )
        total = 0.0
        pos = neg = neu = 0
        for item in items:
            score, p, n, u = self._lexicon_score(f"{item.title} {item.body}")
            total += score
            pos += p
            neg += n
            neu += u
        mean = total / len(items)
        label = (
            SentimentLabel.BULLISH
            if mean > 0.15
            else SentimentLabel.BEARISH
            if mean < -0.15
            else SentimentLabel.NEUTRAL
        )
        confidence = min(1.0, len(items) / 15.0)
        return SentimentScore(
            symbol=symbol,
            score=round(mean, 4),
            label=label,
            headline_count=len(items),
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            sample=items[:5],
            confidence=round(confidence, 3),
        )

    def fetch_news(
        self,
        symbol: str,
        lookback_days: Optional[int] = None,
        limit: int = 25,
    ) -> List[NewsItem]:
        """Fetch recent headlines for a symbol (best-effort; returns [] if no API key)."""
        # Uses NEWS_API_KEY from env if present, otherwise returns empty list
        import os

        news_key = os.getenv("NEWS_API_KEY")
        if not news_key:
            self.logger.warning(
                "NEWS_API_KEY not set — returning empty news list (offline mode)"
            )
            return []
        lookback = lookback_days or self.sent_config.lookback_days
        from_date = (datetime.utcnow() - timedelta(days=lookback)).strftime("%Y-%m-%d")
        try:
            import requests

            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": symbol,
                    "from": from_date,
                    "sortBy": "publishedAt",
                    "pageSize": limit,
                    "language": "en",
                    "apiKey": news_key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            items: List[NewsItem] = []
            for art in data.get("articles", []):
                published = None
                if art.get("publishedAt"):
                    try:
                        published = datetime.fromisoformat(
                            art["publishedAt"].replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                    except ValueError:
                        published = None
                items.append(
                    NewsItem(
                        title=art.get("title") or "",
                        source=art.get("source", {}).get("name", "unknown")
                        if isinstance(art.get("source"), dict)
                        else "unknown",
                        url=art.get("url") or "",
                        published_at=published,
                        body=art.get("description") or "",
                    )
                )
            return items
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("News fetch failed for %s: %s", symbol, exc)
            return []

    def full_analysis(
        self, symbol: str, news_items: Optional[List[NewsItem]] = None
    ) -> SentimentScore:
        """One-stop: fetch news (if possible) and produce an aggregate score."""
        items = news_items if news_items is not None else self.fetch_news(symbol)
        if not items:
            # offline fallback — neutral with a note
            return SentimentScore(
                symbol=symbol,
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                details={"mode": "offline", "note": "No news sources configured"},
            )
        return self.analyze_news(items, symbol)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _lexicon_score(self, text: str):
        """Return (score, positive_count, negative_count, neutral_count)."""
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        if not tokens:
            return 0.0, 0, 0, 0
        score = 0.0
        pos = neg = neu = 0
        negation = False
        for i, token in enumerate(tokens):
            if token in _NEGATORS:
                negation = True
                continue
            weight = 1.0
            # look back for intensifier
            if i > 0 and tokens[i - 1] in _INTENSIFIERS:
                weight = 1.6
            if token in _BULLISH:
                w = weight * (0.6 if i < len(tokens) - 1 and tokens[i + 1] in _NEGATORS else 1.0)
                score += -w if negation else w
                pos += 1
                negation = False
            elif token in _BEARISH:
                w = weight * (0.6 if i < len(tokens) - 1 and tokens[i + 1] in _NEGATORS else 1.0)
                score += w if negation else -w
                neg += 1
                negation = False
            else:
                neu += 1
        # normalize to [-1, 1] with soft tanh
        import math

        norm = math.tanh(score / max(1, len(tokens) / 6))
        return norm, pos, neg, neu
