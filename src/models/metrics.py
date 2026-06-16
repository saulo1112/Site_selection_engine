"""Metricas de ranking para evaluar scores de ubicaciones contra la etiqueta look-alike.

Implementacion generica sobre arrays (no atada a MCDA): la reutilizan v1 (MCDA), v2 y v3.
Replican las metricas del paper de referencia (Lu et al., 2024; ver docs/metodologia.md):

  - NDCG@K        : calidad del orden en el top-K (premia poner positivos arriba).
  - top-K hitting : fraccion de positivos reales capturados en el top-K (recall@K).
  - top-K loss    : fraccion de positivos reales que quedaron FUERA del top-K (1 - hitting).

Convencion: `scores` mas alto = mejor candidato; `labels` binaria (1 = positivo).
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
        raise ValueError(f"scores y labels deben tener igual forma: {s.shape} vs {y.shape}")
    if s.ndim != 1:
        raise ValueError(f"se esperaba un arreglo 1D, se recibio ndim={s.ndim}")
    if s.size == 0:
        raise ValueError("scores/labels vacios")
    return s, y


def _topk_indices(scores: FloatArray, k: int) -> IntArray:
    """Indices de los k scores mas altos (orden descendente, estable ante empates)."""
    k = min(k, scores.size)
    # argsort estable sobre el negativo -> mayor score primero, empates por orden original.
    return np.argsort(-scores, kind="stable")[:k]


def topk_hitting_rate(scores: npt.ArrayLike, labels: npt.ArrayLike, k: int) -> float:
    """Fraccion de positivos reales capturados en el top-K (recall@K).

    Denominador = total de positivos reales (no K), para que mida cuanta de la
    "verdad" recuperamos al explorar solo K celdas.
    """
    s, y = _validate(scores, labels)
    total_pos = int(y.sum())
    if total_pos == 0:
        return float("nan")
    top = _topk_indices(s, k)
    return float(y[top].sum()) / total_pos


def topk_loss(scores: npt.ArrayLike, labels: npt.ArrayLike, k: int) -> float:
    """Fraccion de positivos reales que quedaron FUERA del top-K (1 - hitting@K)."""
    hit = topk_hitting_rate(scores, labels, k)
    return float("nan") if np.isnan(hit) else 1.0 - hit


def _dcg(relevances: FloatArray) -> float:
    """Discounted Cumulative Gain con descuento log2(rank+1) (rank base 1)."""
    if relevances.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances * discounts))


def ndcg_at_k(scores: npt.ArrayLike, labels: npt.ArrayLike, k: int) -> float:
    """NDCG@K con relevancia binaria.

    DCG del orden inducido por `scores` (top-K) normalizado por el DCG ideal
    (todos los positivos arriba). Devuelve NaN si no hay positivos.
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
    """Calcula las tres metricas de ranking de un tiron (para reportes de v1/v2/v3)."""
    return {
        "ndcg_at_k": ndcg_at_k(scores, labels, k),
        "topk_hitting": topk_hitting_rate(scores, labels, k),
        "topk_loss": topk_loss(scores, labels, k),
        "k": float(k),
    }
