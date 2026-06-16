"""Configuracion de logging estructurado para el pipeline.

Todos los modulos del pipeline usan `get_logger(__name__)` en lugar de print().
El formato incluye timestamp, nivel, modulo y mensaje para trazabilidad de las
descargas y transformaciones.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Devuelve un logger configurado una sola vez para todo el proceso."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root = logging.getLogger()
        root.setLevel(level)
        root.addHandler(handler)
        # osmnx/urllib3 son ruidosos a nivel INFO; subir su umbral.
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("osmnx").setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)
