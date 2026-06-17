"""API de inferencia/serving del Site Selection Engine (FastAPI).

Sirve el ranking de hexagonos look-alike y la inferencia en vivo del modelo, leyendo
artefactos versionados (sin PostGIS en runtime). Pensada para una demo en vivo:
el frontend Streamlit (app/streamlit_app.py) la consume.

Ejecutar local:
    uv run uvicorn src.api.main:app --reload
Docs interactivas: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.api import service
from src.api.schemas import (
    HealthResponse,
    HexDetail,
    HexesResponse,
    HexScore,
    ScoreRequest,
    ScoreResponse,
)

app = FastAPI(
    title="Site Selection Engine API",
    version="0.1.0",
    description=(
        "Ranking de hexagonos H3 (Bogota) por similitud look-alike a Tiendas D1. "
        "El score es P(tipo-D1): prioridad de exploracion, no prediccion de ventas."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.SERVING_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _parse_bbox(bbox: str | None) -> tuple[float, float, float, float] | None:
    if not bbox:
        return None
    try:
        parts = [float(x) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        return parts[0], parts[1], parts[2], parts[3]
    except ValueError:
        raise HTTPException(422, "bbox debe ser 'minlon,minlat,maxlon,maxlat'.")


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"name": "Site Selection Engine API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    models = service.available_models()
    n = len(service.get_ranking(service.resolve_model(None))) if models else 0
    return HealthResponse(
        status="ok" if models else "no-artifacts",
        default_model=config.SERVING_MODEL,
        available_models=models,
        n_hexes=n,
    )


@app.get("/models")
def models() -> dict:
    return {"available": service.available_models(), "default": config.SERVING_MODEL}


@app.get("/hexes", response_model=HexesResponse)
def hexes(
    model: str | None = Query(None, description="mcda | v2 | v3 | v4"),
    top_k: int | None = Query(None, ge=1, le=5000),
    min_score: float | None = Query(None, ge=0.0, le=1.0),
    bbox: str | None = Query(None, description="minlon,minlat,maxlon,maxlat"),
) -> HexesResponse:
    m = service.resolve_model(model)
    df = service.query_hexes(m, top_k=top_k, min_score=min_score, bbox=_parse_bbox(bbox))
    items = [
        HexScore(
            h3_index=row.h3_index,
            lat_centroid=row.lat_centroid,
            lon_centroid=row.lon_centroid,
            score=row.score,
            rank=int(row.rank),
            tiene_d1=None if row.tiene_d1 is None else int(row.tiene_d1),
        )
        for row in df.itertuples(index=False)
    ]
    return HexesResponse(
        model=m, score_col=config.SERVING_SCORE_COL[m], count=len(items), items=items
    )


@app.get("/hex/{h3_index}", response_model=HexDetail)
def hex_detail(h3_index: str, model: str | None = Query(None)) -> HexDetail:
    m = service.resolve_model(model)
    try:
        return HexDetail(**service.hex_detail(m, h3_index))
    except KeyError:
        raise HTTPException(404, f"Hexagono '{h3_index}' no encontrado en el ranking '{m}'.")


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    m = service.resolve_model(req.model)
    if m not in config.SERVING_MODELS:
        raise HTTPException(422, f"El modelo '{m}' no admite inferencia en vivo (sin .joblib).")
    try:
        result = service.score_hex(m, req.h3_index, req.features)
    except KeyError:
        raise HTTPException(404, f"Hexagono '{req.h3_index}' no esta en la tabla de features.")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(422, str(exc))
    return ScoreResponse(**result)


@app.get("/pois/{name}")
def pois(name: str) -> dict:
    try:
        return service.poi_layer(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Capa POI '{name}' no disponible.")
