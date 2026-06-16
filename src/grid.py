"""ETAPA 2 — Grid hexagonal H3 sobre Bogota.

Genera el grid H3 (resolucion configurable, por defecto 9 ~ 0.105 km2) que cubre
el limite administrativo de Bogota, conservando solo los hexagonos cuyo centroide
cae dentro del poligono. Guarda el grid y reporta cuantos hexagonos contienen al
menos un POI D1 (positivos disponibles antes de spatial CV).

Usa la API de h3 v4 (LatLngPoly, polygon_to_cells, cell_to_boundary, cell_to_latlng).

Ejecutar de forma independiente:
    uv run python -m src.grid
"""

from __future__ import annotations

import geopandas as gpd
import h3
from shapely.geometry import MultiPolygon, Point, Polygon

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)


def _polygon_to_cells(poly: Polygon, resolution: int) -> set[str]:
    """Devuelve las celdas H3 que cubren un Polygon shapely (coords lon/lat)."""
    # h3.LatLngPoly espera vertices en orden (lat, lng).
    outer = [(lat, lng) for lng, lat in poly.exterior.coords]
    holes = [
        [(lat, lng) for lng, lat in interior.coords]
        for interior in poly.interiors
    ]
    h3shape = h3.LatLngPoly(outer, *holes)
    return set(h3.polygon_to_cells(h3shape, resolution))


def _cells_covering(geometry, resolution: int) -> set[str]:
    """Celdas H3 que cubren un Polygon o MultiPolygon."""
    polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
    cells: set[str] = set()
    for poly in polygons:
        cells |= _polygon_to_cells(poly, resolution)
    return cells


def _cell_to_polygon(cell: str) -> Polygon:
    """Construye el poligono shapely (lon/lat) de una celda H3."""
    boundary = h3.cell_to_boundary(cell)  # secuencia de (lat, lng)
    return Polygon([(lng, lat) for lat, lng in boundary])


def build_grid() -> gpd.GeoDataFrame:
    """Construye el grid H3 filtrado por centroide dentro del limite."""
    boundary = gpd.read_file(config.BOUNDARY_PATH)
    geometry = boundary.geometry.iloc[0]

    cells = _cells_covering(geometry, config.H3_RESOLUTION)
    logger.info("Celdas H3 que cubren el bbox del limite: %d", len(cells))

    # Preparar el poligono unido para el test de contencion del centroide.
    boundary_union = boundary.geometry.union_all() if hasattr(
        boundary.geometry, "union_all"
    ) else boundary.geometry.unary_union

    records = []
    for cell in cells:
        lat, lng = h3.cell_to_latlng(cell)
        if not boundary_union.contains(Point(lng, lat)):
            continue
        records.append({
            "h3_index": cell,
            "lat_centroid": lat,
            "lon_centroid": lng,
            "geometry": _cell_to_polygon(cell),
        })

    grid = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    logger.info("Hexagonos con centroide dentro de Bogota: %d", len(grid))
    return grid


def count_positives(grid: gpd.GeoDataFrame) -> int:
    """Cuenta hexagonos que contienen al menos un POI D1 (positivos)."""
    if not config.POIS_D1_PATH.exists():
        logger.warning("No existe %s; no se cuentan positivos", config.POIS_D1_PATH.name)
        return -1
    d1 = gpd.read_file(config.POIS_D1_PATH)
    joined = gpd.sjoin(d1, grid[["h3_index", "geometry"]], predicate="within", how="inner")
    return joined["h3_index"].nunique()


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    grid = build_grid()
    grid.to_file(config.GRID_PATH, driver="GeoJSON")
    logger.info("Grid guardado -> %s", config.GRID_PATH.name)

    n_total = len(grid)
    n_pos = count_positives(grid)
    logger.info("=" * 50)
    logger.info("Total hexagonos:            %d", n_total)
    logger.info("Hexagonos con D1 (positivos antes de spatial CV): %d", n_pos)
    if n_pos > 0:
        logger.info("Ratio positivos:            %.3f%%", 100 * n_pos / n_total)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
