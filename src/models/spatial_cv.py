"""SPATIAL cross-validation utilities (for v3).

The problem with a random split (v2): neighboring H3 hexagons are spatially
autocorrelated, so scattering neighbors between train and test leaks information ->
inflated metrics. These utilities generate folds that separate train and test
GEOGRAPHICALLY:

  1. Each res-9 hexagon is assigned to a spatial BLOCK = its H3 parent at a coarse
     resolution (`cell_to_parent`). Whole blocks go together to train or test.
  2. `StratifiedGroupKFold` distributes the blocks into folds respecting the groups
     (no block is split) and balancing the proportion of positives.
  3. BUFFER: for each fold, hexagons within <=k rings (`grid_disk`) of any test cell
     are excluded from train, removing leakage at block borders.

Pure functions (no I/O); consumed by src/models/lookalike_v3.py.
"""

from __future__ import annotations

from collections.abc import Iterator

import h3
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedGroupKFold


def assign_spatial_blocks(h3_indices: list[str], coarse_res: int) -> npt.NDArray[np.str_]:
    """Spatial block of each cell = its H3 parent at `coarse_res` (coarse resolution)."""
    return np.array([h3.cell_to_parent(c, coarse_res) for c in h3_indices])


def _buffer_exclusion_cells(test_cells: list[str], buffer_rings: int) -> set[str]:
    """Set of cells within `buffer_rings` rings of any test cell."""
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
    """Generates (train_idx, test_idx, n_buffer_removed) per fold with spatial separation.

    - Blocks = H3 parent at `coarse_res`; folds via StratifiedGroupKFold over the blocks.
    - From each fold's train set, cells within `buffer_rings` rings of any test cell are
      removed (buffer zone), guaranteeing that train and test don't share an immediate
      neighborhood.
    """
    h3_arr = np.asarray(h3_indices)
    y_arr = np.asarray(y, dtype=np.int_)
    groups = assign_spatial_blocks(list(h3_indices), coarse_res)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in sgkf.split(h3_arr, y_arr, groups=groups):
        test_cells = h3_arr[test_idx].tolist()
        excluded = _buffer_exclusion_cells(test_cells, buffer_rings)

        # Keep in train only the cells that do NOT fall in the test buffer zone.
        keep_mask = np.array([h3_arr[i] not in excluded for i in train_idx])
        kept_train_idx = train_idx[keep_mask]
        n_removed = int((~keep_mask).sum())

        yield kept_train_idx, test_idx, n_removed
