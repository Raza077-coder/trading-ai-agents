"""Base agent class — the foundation every trading agent inherits from."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from trading_agents.core.config import AgentConfig
from trading_agents.core.exceptions import TradingAgentError

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all trading agents.

    Provides a uniform lifecycle (``initialize`` → ``run`` → ``shutdown``),
    structured logging, and error handling so every agent in the suite behaves
    consistently and can be composed by the orchestrator.

    Attributes:
        name: Human-readable agent name.
        config: Agent-specific configuration.
        logger: Logger instance bound to the agent name.
    """

    def __init__(self, name: str, config: AgentConfig | None = None) -> None:
        self.name = name
        self.config = config or AgentConfig()
        self.logger = logging.getLogger(f"trading_agents.{self.name}")
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """One-time setup hook. Subclasses may override; call ``super()`` first."""
        if self._initialized:
            return
        self.logger.debug("Initializing agent '%s'", self.name)
        self._initialized = True

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's core logic.

        Subclasses override this or expose their own public methods
        (``analyze``, ``get_candles``, ...). The generic ``run`` defaults to
        raising :class:`NotImplementedError` so agents stay instantiable.
        """
        raise NotImplementedError(
            f"Agent '{self.name}' does not implement run(); "
            "use its specific public methods instead."
        )

    def shutdown(self) -> None:
        """Cleanup hook. Subclasses may override; call ``super()`` last."""
        self.logger.debug("Shutting down agent '%s'", self.name)
        self._initialized = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _guard_initialized(self) -> None:
        """Raise if the agent has not been initialized yet."""
        if not self._initialized:
            raise TradingAgentError(
                f"Agent '{self.name}' is not initialized — call initialize() first."
            )

    def __enter__(self) -> "BaseAgent":
        self.initialize()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    def __str__(self) -> str:
        return self.name
