"""Esquemas Pydantic de la API de serving."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HexScore(BaseModel):
    """Un hexagono rankeado (vista ligera para el mapa)."""
    h3_index: str
    lat_centroid: float
    lon_centroid: float
    score: float
    rank: int
    tiene_d1: int | None = None


class HexDetail(HexScore):
    """Detalle de un hexagono: score/rank + todas sus features (incl. demografia)."""
    features: dict[str, float | None]
    boundary: list[list[float]] = Field(
        default_factory=list,
        description="Anillo del hexagono como [[lon, lat], ...] (cerrado).",
    )


class HexesResponse(BaseModel):
    model: str
    score_col: str
    count: int
    items: list[HexScore]


class ScoreRequest(BaseModel):
    """Inferencia en vivo. Indica un h3_index (toma sus features del parquet) o
    pasa un vector de features explicito."""
    model: str | None = None
    h3_index: str | None = None
    features: dict[str, float] | None = None


class ScoreResponse(BaseModel):
    model: str
    h3_index: str | None = None
    score: float
    predictors: list[str]


class HealthResponse(BaseModel):
    status: str
    default_model: str
    available_models: list[str]
    n_hexes: int
