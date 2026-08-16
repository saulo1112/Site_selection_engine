"""Loading and querying of serving artifacts (no PostGIS at runtime).

Reads the precomputed rankings (parquet), the features table, the .joblib models and
the POI layers (GeoJSON). Caches everything in memory (loaded once per process).
"""

from __future__ import annotations

import json
from functools import lru_cache

import h3
import joblib
import pandas as pd

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)

_BASE_COLS = ["h3_index", "lat_centroid", "lon_centroid"]


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def available_models() -> list[str]:
    """Models whose precomputed ranking exists on disk."""
    return [m for m, p in config.SERVING_RANKINGS.items() if p.exists()]


def resolve_model(model: str | None) -> str:
    """Requested model if available; otherwise the default; otherwise the first one."""
    avail = available_models()
    if not avail:
        raise FileNotFoundError(
            "No precomputed rankings found. Run the models pipeline "
            "(uv run python -m src.models.lookalike_v3 / lookalike_v4)."
        )
    if model and model in avail:
        return model
    if config.SERVING_MODEL in avail:
        return config.SERVING_MODEL
    return avail[0]


# --------------------------------------------------------------------------- #
# Rankings (normalized to `score` + `rank` columns)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def get_ranking(model: str) -> pd.DataFrame:
    """Model ranking, normalized: h3_index, centroids, score, rank, tiene_d1."""
    path = config.SERVING_RANKINGS[model]
    score_col = config.SERVING_SCORE_COL[model]
    df = pd.read_parquet(path)
    if score_col not in df.columns:
        raise KeyError(f"'{score_col}' is not in {path.name} (columns: {list(df.columns)})")
    out = df[[c for c in _BASE_COLS if c in df.columns]].copy()
    out["score"] = df[score_col].astype(float)
    if "tiene_d1" in df.columns:
        out["tiene_d1"] = df["tiene_d1"].astype("Int64")
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    logger.info("Ranking '%s' loaded: %d hexagons", model, len(out))
    return out


@lru_cache(maxsize=1)
def get_features() -> pd.DataFrame:
    """Features table (for per-hexagon detail and inference by h3_index)."""
    return pd.read_parquet(config.FEATURES_PARQUET_PATH).set_index("h3_index", drop=False)


@lru_cache(maxsize=None)
def get_pipeline(model: str):
    """.joblib model (sklearn Pipeline) for live inference."""
    path = config.SERVING_MODELS.get(model)
    if path is None or not path.exists():
        raise FileNotFoundError(f"No .joblib model for '{model}' ({path}).")
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def query_hexes(
    model: str,
    top_k: int | None = None,
    min_score: float | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Filters the ranking by minimum score, bbox (minlon,minlat,maxlon,maxlat) and top_k."""
    df = get_ranking(model)
    if min_score is not None:
        df = df[df["score"] >= min_score]
    if bbox is not None:
        minlon, minlat, maxlon, maxlat = bbox
        df = df[
            df["lon_centroid"].between(minlon, maxlon)
            & df["lat_centroid"].between(minlat, maxlat)
        ]
    df = df.sort_values("rank")
    if top_k is not None:
        df = df.head(top_k)
    return df.reset_index(drop=True)


def hex_boundary(h3_index: str) -> list[list[float]]:
    """Hexagon ring as [[lon, lat], ...], closed (for GeoJSON/deck.gl)."""
    coords = [[lng, lat] for lat, lng in h3.cell_to_boundary(h3_index)]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def hex_detail(model: str, h3_index: str) -> dict:
    """Hexagon score/rank + all its features + ring geometry."""
    rank_df = get_ranking(model)
    row = rank_df[rank_df["h3_index"] == h3_index]
    if row.empty:
        raise KeyError(h3_index)
    r = row.iloc[0]

    feats = get_features()
    feat_cols = [c for c in feats.columns if c not in (*_BASE_COLS, "tiene_d1")]
    feature_values: dict[str, float | None] = {}
    if h3_index in feats.index:
        frow = feats.loc[h3_index]
        for c in feat_cols:
            v = frow[c]
            feature_values[c] = None if pd.isna(v) else float(v)

    return {
        "h3_index": h3_index,
        "lat_centroid": float(r["lat_centroid"]),
        "lon_centroid": float(r["lon_centroid"]),
        "score": float(r["score"]),
        "rank": int(r["rank"]),
        "tiene_d1": None if pd.isna(r.get("tiene_d1")) else int(r["tiene_d1"]),
        "features": feature_values,
        "boundary": hex_boundary(h3_index),
    }


def score_hex(model: str, h3_index: str | None, features: dict | None) -> dict:
    """Live inference with the .joblib model. Uses the features from the parquet if
    an h3_index is given, or an explicit feature vector."""
    pipe = get_pipeline(model)
    predictors = list(getattr(pipe, "feature_names_in_", []))
    if not predictors:
        raise RuntimeError(f"Model '{model}' does not expose feature_names_in_.")

    if h3_index is not None:
        feats = get_features()
        if h3_index not in feats.index:
            raise KeyError(h3_index)
        X = feats.loc[[h3_index], predictors]
    elif features is not None:
        X = pd.DataFrame([{c: features.get(c) for c in predictors}], columns=predictors)
    else:
        raise ValueError("Provide 'h3_index' or 'features'.")

    proba = float(pipe.predict_proba(X)[:, 1][0])
    return {"model": model, "h3_index": h3_index, "score": proba, "predictors": predictors}


@lru_cache(maxsize=None)
def poi_layer(name: str) -> dict:
    """Raw GeoJSON of a POI layer (d1/competitors/complementary) for overlays."""
    path = config.SERVING_POI_LAYERS.get(name)
    if path is None or not path.exists():
        raise FileNotFoundError(f"POI layer '{name}' not available ({path}).")
    return json.loads(path.read_text(encoding="utf-8"))
