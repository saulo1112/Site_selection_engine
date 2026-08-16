"""STAGE 3 — Load layers into PostGIS.

- Creates the `site_selection` database if it doesn't exist and enables PostGIS.
- Loads the layers (POIs, grid, street network and, if available, census blocks)
  with GIST spatial indexes.

The connection is read from the DATABASE_URL environment variable, with a fallback
defined in src/config.py (port 5433; see docker-compose.yml).

Run standalone:
    uv run python -m src.data.db
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import osmnx as ox
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)

GEOM_COL = "geom"


# --------------------------------------------------------------------------- #
# Connection / setup
# --------------------------------------------------------------------------- #
def ensure_database() -> None:
    """Creates the target database if it doesn't exist (connecting to `postgres`)."""
    url = make_url(config.DATABASE_URL)
    dbname = url.database
    admin_engine = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": dbname}
            ).scalar()
            if exists:
                logger.info("Database '%s' already exists", dbname)
            else:
                conn.execute(text(f'CREATE DATABASE "{dbname}"'))
                logger.info("Database '%s' created", dbname)
    finally:
        admin_engine.dispose()


def get_engine() -> Engine:
    return create_engine(config.DATABASE_URL)


def ensure_postgis(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    logger.info("PostGIS extension enabled")


# --------------------------------------------------------------------------- #
# Layer loading
# --------------------------------------------------------------------------- #
def load_layer(gdf: gpd.GeoDataFrame, table: str, engine: Engine) -> None:
    """Loads a GeoDataFrame into PostGIS (replacing) and creates a GIST index."""
    if gdf.crs is None:
        raise ValueError(f"GeoDataFrame for '{table}' has no CRS defined")
    gdf = gdf.to_crs("EPSG:4326").rename_geometry(GEOM_COL)
    gdf.to_postgis(table, engine, if_exists="replace", index=False)
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_geom "
            f"ON {table} USING GIST ({GEOM_COL})"
        ))
    logger.info("Table '%s' loaded: %d rows (+ GIST index)", table, len(gdf))


def load_geojson_layer(path: Path, table: str, engine: Engine) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run previous stages (download/grid)")
    load_layer(gpd.read_file(path), table, engine)


def load_streets(engine: Engine) -> None:
    """Loads the street network edges as a lines table."""
    if not config.STREETS_GRAPH_PATH.exists():
        logger.warning("%s does not exist; skipping the 'streets' table "
                       "(densidad_vial will remain null)", config.STREETS_GRAPH_PATH.name)
        return
    graph = ox.load_graphml(config.STREETS_GRAPH_PATH)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()
    keep = [c for c in ("osmid", "name", "highway", "length", "geometry") if c in edges.columns]
    edges = edges[keep]
    # osmid/highway/name may come as lists -> convert to str for PostGIS.
    for col in ("osmid", "highway", "name"):
        if col in edges.columns:
            edges[col] = edges[col].astype(str)
    load_layer(edges, config.TABLES["streets"], engine)


def load_censo(engine: Engine) -> None:
    """Loads census blocks if a local file is available; otherwise doesn't block."""
    censo_file = next((p for p in config.CENSO_PATH_CANDIDATES if p.exists()), None)
    if censo_file is None:
        logger.warning(
            "Census file not found (%s). Skipping 'manzanas_censo'. "
            "To enable demographic features, see src/data/load_censo.py.",
            ", ".join(p.name for p in config.CENSO_PATH_CANDIDATES),
        )
        return
    logger.info("Loading census data from %s", censo_file.name)
    load_layer(gpd.read_file(censo_file), config.TABLES["manzanas_censo"], engine)


def load_estrato(engine: Engine) -> None:
    """Loads blocks with socioeconomic stratum (IDECA) if a local file is available; otherwise doesn't block."""
    estrato_file = next((p for p in config.ESTRATO_PATH_CANDIDATES if p.exists()), None)
    if estrato_file is None:
        logger.warning(
            "Stratum file not found (%s). Skipping 'manzanas_estrato'. "
            "To enable the stratum feature, see src/data/load_estrato.py.",
            ", ".join(p.name for p in config.ESTRATO_PATH_CANDIDATES),
        )
        return
    logger.info("Loading stratum data from %s", estrato_file.name)
    load_layer(gpd.read_file(estrato_file), config.TABLES["manzanas_estrato"], engine)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    ensure_database()
    engine = get_engine()
    try:
        ensure_postgis(engine)
        load_geojson_layer(config.POIS_D1_PATH, config.TABLES["pois_d1"], engine)
        load_geojson_layer(config.POIS_COMPETIDORES_PATH, config.TABLES["pois_competidores"], engine)
        load_geojson_layer(config.POIS_COMPLEMENTARIOS_PATH, config.TABLES["pois_complementarios"], engine)
        load_geojson_layer(config.GRID_PATH, config.TABLES["grid"], engine)
        load_streets(engine)
        load_censo(engine)
        load_estrato(engine)
        logger.info("STAGE 3 complete.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
