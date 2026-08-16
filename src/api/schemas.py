"""Pydantic schemas for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HexScore(BaseModel):
    """A ranked hexagon (lightweight view for the map)."""
    h3_index: str
    lat_centroid: float
    lon_centroid: float
    score: float
    rank: int
    tiene_d1: int | None = None


class HexDetail(HexScore):
    """Hexagon detail: score/rank + all its features (incl. demographics)."""
    features: dict[str, float | None]
    boundary: list[list[float]] = Field(
        default_factory=list,
        description="Hexagon ring as [[lon, lat], ...] (closed).",
    )


class HexesResponse(BaseModel):
    model: str
    score_col: str
    count: int
    items: list[HexScore]


class ScoreRequest(BaseModel):
    """Live inference. Provide an h3_index (takes its features from the parquet) or
    pass an explicit feature vector."""
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
