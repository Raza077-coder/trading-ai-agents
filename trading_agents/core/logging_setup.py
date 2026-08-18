"""Structured logging setup for the trading agent suite."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Configure root logging for the suite.

    Args:
        level: Logging level name (DEBUG/INFO/WARNING/ERROR).
        log_file: Optional path to also write logs.
        verbose: If True, use DEBUG level regardless of ``level``.
    """
    effective = "DEBUG" if verbose else level.upper()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, effective, logging.INFO),
        format=_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
