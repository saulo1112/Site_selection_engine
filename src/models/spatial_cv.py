"""Utilidades de validacion cruzada ESPACIAL (para v3).

El problema con un split aleatorio (v2): los hexagonos H3 vecinos estan espacialmente
autocorrelacionados, asi que repartir vecinos entre train y test filtra informacion ->
metricas infladas. Estas utilidades generan folds que separan train y test
GEOGRAFICAMENTE:

  1. Cada hexagono res-9 se asigna a un BLOQUE espacial = su padre H3 a una resolucion
     gruesa (`cell_to_parent`). Bloques enteros van juntos a train o test.
  2. `StratifiedGroupKFold` reparte los bloques en folds respetando los grupos (ningun
     bloque se parte) y balanceando la proporcion de positivos.
  3. BUFFER: para cada fold se excluyen del train los hexagonos a <=k anillos
     (`grid_disk`) de cualquier celda de test, eliminando la fuga en los bordes de bloque.

Funciones puras (sin I/O); las consume src/models/lookalike_v3.py.
"""

from __future__ import annotations

from collections.abc import Iterator

import h3
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedGroupKFold


def assign_spatial_blocks(h3_indices: list[str], coarse_res: int) -> npt.NDArray[np.str_]:
    """Bloque espacial de cada celda = su padre H3 a `coarse_res` (resolucion gruesa)."""
    return np.array([h3.cell_to_parent(c, coarse_res) for c in h3_indices])


def _buffer_exclusion_cells(test_cells: list[str], buffer_rings: int) -> set[str]:
    """Conjunto de celdas dentro de `buffer_rings` anillos de cualquier celda de test."""
    if buffer_rings <= 0:
        return set(test_cells)
    excluded: set[str] = set()
    for cell in test_cells:
        excluded.update(h3.grid_disk(cell, buffer_rings))
    return excluded


def buffered_spatial_folds(
    h3_indices: list[str],
    y: npt.ArrayLike,
    n_folds: int,
    coarse_res: int,
    buffer_rings: int,
    random_state: int,
) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], int]]:
    """Genera (train_idx, test_idx, n_buffer_removidas) por fold con separacion espacial.

    - Bloques = padre H3 a `coarse_res`; folds via StratifiedGroupKFold sobre los bloques.
    - Del train de cada fold se quitan las celdas dentro de `buffer_rings` anillos de
      alguna celda de test (zona de amortiguamiento), garantizando que train y test no
      compartan vecindario inmediato.
    """
    h3_arr = np.asarray(h3_indices)
    y_arr = np.asarray(y, dtype=np.int_)
    groups = assign_spatial_blocks(list(h3_indices), coarse_res)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in sgkf.split(h3_arr, y_arr, groups=groups):
        test_cells = h3_arr[test_idx].tolist()
        excluded = _buffer_exclusion_cells(test_cells, buffer_rings)

        # Mantener en train solo las celdas que NO caen en la zona de buffer del test.
        keep_mask = np.array([h3_arr[i] not in excluded for i in train_idx])
        kept_train_idx = train_idx[keep_mask]
        n_removed = int((~keep_mask).sum())

        yield kept_train_idx, test_idx, n_removed
