"""ETAPA 1 — Descarga de datos crudos (OSM/Overpass + red vial).

Descarga, de forma idempotente y con reintentos, las capas base del pipeline:
  1a. Limite administrativo de Bogota  -> data/raw/bogota_boundary.geojson
  1b. POIs (D1, competidores, complementarios) dentro del limite administrativo
  1c. Red vial (osmnx, network_type=drive) -> data/raw/bogota_streets.graphml

Ejecutar de forma independiente:
    uv run python -m src.download
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import geopandas as gpd
import osmnx as ox
import requests
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Overpass con reintentos y backoff exponencial
# --------------------------------------------------------------------------- #
def run_overpass(query_body: str, out_mode: str = "center") -> list[dict]:
    """Ejecuta una consulta Overpass y devuelve la lista de `elements`.

    out_mode: "center" (POIs como punto) o "geom" (geometria completa, p.ej.
    limites administrativos). Rota endpoints y reintenta con backoff.
    """
    full_query = (
        f"[out:json][timeout:{config.REQUEST_TIMEOUT}];"
        f"({query_body});"
        f"out {out_mode};"
    )
    headers = {"User-Agent": config.USER_AGENT}

    last_exc: Exception | None = None
    for attempt in range(config.MAX_RETRIES):
        endpoint = config.OVERPASS_ENDPOINTS[attempt % len(config.OVERPASS_ENDPOINTS)]
        try:
            resp = requests.post(
                endpoint,
                data={"data": full_query},
                headers=headers,
                timeout=config.REQUEST_TIMEOUT + 30,
            )
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            wait = config.BACKOFF_BASE * (2 ** attempt)
            logger.warning(
                "Overpass intento %d/%d fallo en %s (%s); reintento en %ds",
                attempt + 1, config.MAX_RETRIES, endpoint, type(exc).__name__, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Overpass fallo tras {config.MAX_RETRIES} intentos: {last_exc}")


# --------------------------------------------------------------------------- #
# Helpers de idempotencia y log de descarga
# --------------------------------------------------------------------------- #
def _load_download_log() -> dict:
    if config.DOWNLOAD_LOG_PATH.exists():
        return json.loads(config.DOWNLOAD_LOG_PATH.read_text(encoding="utf-8"))
    return {}


def _record_download(layer: str, path: Path, n_features: int, skipped: bool) -> None:
    log = _load_download_log()
    log[layer] = {
        "path": str(path.relative_to(config.PROJECT_ROOT)),
        "n_features": n_features,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "skipped_existing": skipped,
    }
    config.DOWNLOAD_LOG_PATH.write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _idempotent(layer: str, target: Path, builder: Callable[[], int]) -> None:
    """Salta la descarga si el archivo ya existe; si no, ejecuta `builder()`."""
    if target.exists():
        logger.info("[%s] %s ya existe -> se salta la descarga (idempotente)",
                    layer, target.name)
        _record_download(layer, target, _count_existing(target), skipped=True)
        return
    n = builder()
    logger.info("[%s] descargadas %d features -> %s", layer, n, target.name)
    _record_download(layer, target, n, skipped=False)


def _count_existing(path: Path) -> int:
    """Cuenta features de un archivo ya descargado (para el log)."""
    try:
        if path.suffix == ".graphml":
            g = ox.load_graphml(path)
            return g.number_of_edges()
        return len(gpd.read_file(path))
    except Exception:  # noqa: BLE001 - el conteo es informativo, no debe romper
        return -1


# --------------------------------------------------------------------------- #
# Parseo de elementos Overpass -> GeoDataFrame de puntos
# --------------------------------------------------------------------------- #
def overpass_to_point_gdf(elements: list[dict]) -> gpd.GeoDataFrame:
    """Convierte elementos Overpass (out center) en puntos con sus tags."""
    records = []
    for el in elements:
        et = el.get("type")
        if et == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:  # way / relation -> usar center
            center = el.get("center")
            if not center:
                continue
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue
        tags = el.get("tags", {})
        records.append({
            "osm_type": et,
            "osm_id": el.get("id"),
            "name": tags.get("name"),
            "brand": tags.get("brand"),
            "shop": tags.get("shop"),
            "amenity": tags.get("amenity"),
            "highway": tags.get("highway"),
            "geometry": Point(lon, lat),
        })
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    if not gdf.empty:
        gdf = gdf.drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(drop=True)
    return gdf


# --------------------------------------------------------------------------- #
# 1a. Limite administrativo
# --------------------------------------------------------------------------- #
def _assemble_boundary(elements: list[dict]) -> MultiPolygon | Polygon:
    """Ensambla el (multi)poligono de una relacion Overpass (out geom)."""
    outer_lines, inner_lines = [], []
    for el in elements:
        if el.get("type") != "relation":
            continue
        for member in el.get("members", []):
            geom = member.get("geometry")
            if member.get("type") != "way" or not geom:
                continue
            coords = [(pt["lon"], pt["lat"]) for pt in geom]
            if len(coords) >= 2:
                line = LineString(coords)
                (inner_lines if member.get("role") == "inner" else outer_lines).append(line)

    if not outer_lines:
        raise ValueError("La relacion no devolvio anillos exteriores via Overpass")

    outer_polys = list(polygonize(unary_union(outer_lines)))
    if not outer_polys:
        raise ValueError("No se pudo poligonizar el limite administrativo")
    boundary = unary_union(outer_polys)

    if inner_lines:
        inner_polys = list(polygonize(unary_union(inner_lines)))
        if inner_polys:
            boundary = boundary.difference(unary_union(inner_polys))
    return boundary


def download_boundary() -> int:
    """1a. Descarga y guarda el limite de Bogota (Overpass; fallback osmnx)."""
    try:
        elements = run_overpass(f"rel({config.STUDY_RELATION_ID});", out_mode="geom")
        geometry = _assemble_boundary(elements)
        source = "overpass"
    except Exception as exc:  # noqa: BLE001 - fallback explicito y logueado
        logger.warning("Ensamblado via Overpass fallo (%s); fallback a osmnx", exc)
        gdf_fb = ox.geocode_to_gdf(f"R{config.STUDY_RELATION_ID}", by_osmid=True)
        geometry = gdf_fb.geometry.iloc[0]
        source = "osmnx"

    gdf = gpd.GeoDataFrame(
        {"name": [config.STUDY_CITY], "relation_id": [config.STUDY_RELATION_ID],
         "source": [source]},
        geometry=[geometry], crs="EPSG:4326",
    )
    gdf.to_file(config.BOUNDARY_PATH, driver="GeoJSON")
    logger.info("Limite ensamblado via %s | area aprox %.1f km2",
                source, _approx_area_km2(geometry))
    return len(gdf)


def _approx_area_km2(geometry) -> float:
    return (
        gpd.GeoSeries([geometry], crs="EPSG:4326")
        .to_crs("EPSG:3116")  # MAGNA-SIRGAS / Colombia Bogota zone
        .area.iloc[0] / 1e6
    )


# --------------------------------------------------------------------------- #
# 1b. POIs
# --------------------------------------------------------------------------- #
def _assign_competidor_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    label = gdf["brand"].fillna(gdf["name"]).fillna("")
    gdf["marca"] = label
    gdf["es_d1"] = label.str.contains("D1", case=False, na=False).astype(int)
    return gdf


def _assign_complementario_categoria(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    def categorize(row) -> str | None:
        for categoria, key, valores in config.COMPLEMENTARIO_RULES:
            val = row.get(key)
            if val is not None and val in valores:
                return categoria
        return None
    gdf["categoria"] = gdf.apply(categorize, axis=1)
    n_sin = int(gdf["categoria"].isna().sum())
    if n_sin:
        logger.warning("%d POIs complementarios sin categoria asignada (descartados)", n_sin)
    return gdf[gdf["categoria"].notna()].reset_index(drop=True)


def download_pois_d1() -> int:
    elements = run_overpass(config.PIPELINE_QUERY_D1.format(area_id=config.STUDY_AREA_ID))
    gdf = overpass_to_point_gdf(elements)
    gdf.to_file(config.POIS_D1_PATH, driver="GeoJSON")
    return len(gdf)


def download_pois_competidores() -> int:
    elements = run_overpass(
        config.PIPELINE_QUERY_COMPETIDORES.format(area_id=config.STUDY_AREA_ID)
    )
    gdf = overpass_to_point_gdf(elements)
    gdf = _assign_competidor_fields(gdf)
    gdf.to_file(config.POIS_COMPETIDORES_PATH, driver="GeoJSON")
    logger.info("Competidores: %d D1 / %d total supermercados",
                int(gdf["es_d1"].sum()), len(gdf))
    return len(gdf)


def download_pois_complementarios() -> int:
    elements = run_overpass(
        config.PIPELINE_QUERY_COMPLEMENTARIOS.format(area_id=config.STUDY_AREA_ID)
    )
    gdf = overpass_to_point_gdf(elements)
    gdf = _assign_complementario_categoria(gdf)
    counts = gdf["categoria"].value_counts().to_dict()
    logger.info("Complementarios por categoria: %s", counts)
    gdf.to_file(config.POIS_COMPLEMENTARIOS_PATH, driver="GeoJSON")
    return len(gdf)


# --------------------------------------------------------------------------- #
# 1c. Red vial
# --------------------------------------------------------------------------- #
def download_streets() -> int:
    boundary = gpd.read_file(config.BOUNDARY_PATH)
    polygon = boundary.geometry.iloc[0]
    logger.info("Descargando red vial (network_type=%s)... puede tardar varios minutos",
                config.STREET_NETWORK_TYPE)
    graph = ox.graph_from_polygon(polygon, network_type=config.STREET_NETWORK_TYPE)
    ox.save_graphml(graph, config.STREETS_GRAPH_PATH)
    return graph.number_of_edges()


# --------------------------------------------------------------------------- #
# Orquestacion
# --------------------------------------------------------------------------- #
def main() -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    _idempotent("bogota_boundary", config.BOUNDARY_PATH, download_boundary)
    _idempotent("pois_d1", config.POIS_D1_PATH, download_pois_d1)
    _idempotent("pois_competidores", config.POIS_COMPETIDORES_PATH, download_pois_competidores)
    _idempotent("pois_complementarios", config.POIS_COMPLEMENTARIOS_PATH,
                download_pois_complementarios)
    _idempotent("bogota_streets", config.STREETS_GRAPH_PATH, download_streets)

    logger.info("ETAPA 1 completa. Log: %s", config.DOWNLOAD_LOG_PATH.name)


if __name__ == "__main__":
    main()
