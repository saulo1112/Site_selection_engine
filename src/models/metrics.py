"""Ranking metrics for evaluating location scores against the look-alike label.

Generic implementation over arrays (not tied to MCDA): reused by v1 (MCDA), v2 and v3.
They replicate the metrics from the reference paper (Lu et al., 2024; see docs/metodologia.md):

  - NDCG@K        : quality of the ordering in the top-K (rewards ranking positives higher).
  - top-K hitting : fraction of true positives captured in the top-K (recall@K).
  - top-K loss    : fraction of true positives left OUT of the top-K (1 - hitting).

Convention: higher `scores` = better candidate; `labels` binary (1 = positive).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


def _validate(scores: npt.ArrayLike, labels: npt.ArrayLike) -> tuple[FloatArray, IntArray]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int_)
    if s.shape != y.shape:
        raise ValueError(f"scores and labels must have the same shape: {s.shape} vs {y.shape}")
    if s.ndim != 1:
        raise ValueError(f"expected a 1D array, got ndim={s.ndim}")
    if s.size == 0:
        raise ValueError("scores/labels are empty")
    return s, y


def _topk_indices(scores: FloatArray, k: int) -> IntArray:
    """Indices of the k highest scores (descending order, stable under ties)."""
    k = min(k, scores.size)
    # stable argsort over the negative -> highest score first, ties by original order.
    return np.argsort(-scores, kind="stable")[:k]


def topk_hitting_rate(scores: npt.ArrayLike, labels: npt.ArrayLike, k: int) -> float:
    """Fraction of true positives captured in the top-K (recall@K).

    Denominator = total number of true positives (not K), so it measures how much of the
    "ground truth" we recover by exploring only K cells.
    """
    s, y = _validate(scores, labels)
    total_pos = int(y.sum())
    if total_pos == 0:
        return float("nan")
    top = _topk_indices(s, k)
    return float(y[top].sum()) / total_pos


def topk_loss(scores: npt.ArrayLike, labels: npt.ArrayLike, k: int) -> float:
    """Fraction of true positives left OUT of the top-K (1 - hitting@K)."""
    hit = topk_hitting_rate(scores, labels, k)
    return float("nan") if np.isnan(hit) else 1.0 - hit


def _dcg(relevances: FloatArray) -> float:
    """Discounted Cumulative Gain with log2(rank+1) discount (rank base 1)."""
    if relevances.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances * discounts))


def ndcg_at_k(scores: npt.ArrayLike, labels: npt.ArrayLike, k: int) -> float:
    """NDCG@K with binary relevance.

    DCG of the ordering induced by `scores` (top-K) normalized by the ideal DCG
    (all positives ranked first). Returns NaN if there are no positives.
    """
    s, y = _validate(scores, labels)
    k = min(k, s.size)
    top = _topk_indices(s, k)
    dcg = _dcg(y[top].astype(np.float64))

    n_pos = int(y.sum())
    ideal_rel = np.ones(min(n_pos, k), dtype=np.float64)
    idcg = _dcg(ideal_rel)
    if idcg == 0.0:
        return float("nan")
    return dcg / idcg


def ranking_report(scores: npt.ArrayLike, labels: npt.ArrayLike, k: int) -> dict[str, float]:
    """Computes all three ranking metrics in one go (for v1/v2/v3 reports)."""
    return {
        "ndcg_at_k": ndcg_at_k(scores, labels, k),
        "topk_hitting": topk_hitting_rate(scores, labels, k),
        "topk_loss": topk_loss(scores, labels, k),
        "k": float(k),
    }
