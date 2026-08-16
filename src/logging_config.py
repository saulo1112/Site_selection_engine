"""Structured logging configuration for the pipeline.

All pipeline modules use `get_logger(__name__)` instead of print().
The format includes timestamp, level, module, and message for traceability
of downloads and transformations.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Returns a logger configured once for the whole process."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root = logging.getLogger()
        root.setLevel(level)
        root.addHandler(handler)
        # osmnx/urllib3 are noisy at INFO level; raise their threshold.
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("osmnx").setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)
