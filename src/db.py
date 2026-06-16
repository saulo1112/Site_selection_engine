"""ETAPA 3 — Carga de capas a PostGIS.

- Crea la base de datos `site_selection` si no existe y habilita PostGIS.
- Carga las capas (POIs, grid, red vial y, si esta disponible, manzanas del censo)
  con indices espaciales GIST.

La conexion se lee de la variable de entorno DATABASE_URL, con fallback definido en
src/config.py (puerto 5433; ver docker-compose.yml).

Ejecutar de forma independiente:
    uv run python -m src.db
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
# Conexion / setup
# --------------------------------------------------------------------------- #
def ensure_database() -> None:
    """Crea la base de datos destino si no existe (conectando a `postgres`)."""
    url = make_url(config.DATABASE_URL)
    dbname = url.database
    admin_engine = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": dbname}
            ).scalar()
            if exists:
                logger.info("Base de datos '%s' ya existe", dbname)
            else:
                conn.execute(text(f'CREATE DATABASE "{dbname}"'))
                logger.info("Base de datos '%s' creada", dbname)
    finally:
        admin_engine.dispose()


def get_engine() -> Engine:
    return create_engine(config.DATABASE_URL)


def ensure_postgis(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    logger.info("Extension PostGIS habilitada")


# --------------------------------------------------------------------------- #
# Carga de capas
# --------------------------------------------------------------------------- #
def load_layer(gdf: gpd.GeoDataFrame, table: str, engine: Engine) -> None:
    """Carga un GeoDataFrame a PostGIS (reemplaza) y crea indice GIST."""
    if gdf.crs is None:
        raise ValueError(f"GeoDataFrame para '{table}' sin CRS definido")
    gdf = gdf.to_crs("EPSG:4326").rename_geometry(GEOM_COL)
    gdf.to_postgis(table, engine, if_exists="replace", index=False)
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_geom "
            f"ON {table} USING GIST ({GEOM_COL})"
        ))
    logger.info("Tabla '%s' cargada: %d filas (+ indice GIST)", table, len(gdf))


def load_geojson_layer(path: Path, table: str, engine: Engine) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}; corre etapas previas (download/grid)")
    load_layer(gpd.read_file(path), table, engine)


def load_streets(engine: Engine) -> None:
    """Carga las aristas de la red vial como tabla de lineas."""
    if not config.STREETS_GRAPH_PATH.exists():
        logger.warning("No existe %s; se omite la tabla 'streets' "
                       "(densidad_vial quedara nula)", config.STREETS_GRAPH_PATH.name)
        return
    graph = ox.load_graphml(config.STREETS_GRAPH_PATH)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()
    keep = [c for c in ("osmid", "name", "highway", "length", "geometry") if c in edges.columns]
    edges = edges[keep]
    # osmid/highway/name pueden venir como listas -> convertir a str para PostGIS.
    for col in ("osmid", "highway", "name"):
        if col in edges.columns:
            edges[col] = edges[col].astype(str)
    load_layer(edges, config.TABLES["streets"], engine)


def load_censo(engine: Engine) -> None:
    """Carga manzanas del censo si hay un archivo local; si no, no bloquea."""
    censo_file = next((p for p in config.CENSO_PATH_CANDIDATES if p.exists()), None)
    if censo_file is None:
        logger.warning(
            "No se encontro archivo de censo (%s). Se omite 'manzanas_censo'. "
            "Para habilitar features demograficas, ver src/load_censo.py.",
            ", ".join(p.name for p in config.CENSO_PATH_CANDIDATES),
        )
        return
    logger.info("Cargando censo desde %s", censo_file.name)
    load_layer(gpd.read_file(censo_file), config.TABLES["manzanas_censo"], engine)


# --------------------------------------------------------------------------- #
# Orquestacion
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
        logger.info("ETAPA 3 completa.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
