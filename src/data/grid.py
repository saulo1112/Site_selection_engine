"""STAGE 2 — H3 hexagonal grid over Bogota.

Generates the H3 grid (configurable resolution, default 9 ~ 0.105 km2) covering
Bogota's administrative boundary, keeping only the hexagons whose centroid
falls within the polygon. Saves the grid and reports how many hexagons contain
at least one D1 POI (positives available before spatial CV).

Uses the h3 v4 API (LatLngPoly, polygon_to_cells, cell_to_boundary, cell_to_latlng).

Run standalone:
    uv run python -m src.data.grid
"""

from __future__ import annotations

import geopandas as gpd
import h3
from shapely.geometry import MultiPolygon, Point, Polygon

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)


def _polygon_to_cells(poly: Polygon, resolution: int) -> set[str]:
    """Returns the H3 cells that cover a shapely Polygon (lon/lat coords)."""
    # h3.LatLngPoly expects vertices in (lat, lng) order.
    outer = [(lat, lng) for lng, lat in poly.exterior.coords]
    holes = [
        [(lat, lng) for lng, lat in interior.coords]
        for interior in poly.interiors
    ]
    h3shape = h3.LatLngPoly(outer, *holes)
    return set(h3.polygon_to_cells(h3shape, resolution))


def _cells_covering(geometry, resolution: int) -> set[str]:
    """H3 cells that cover a Polygon or MultiPolygon."""
    polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
    cells: set[str] = set()
    for poly in polygons:
        cells |= _polygon_to_cells(poly, resolution)
    return cells


def _cell_to_polygon(cell: str) -> Polygon:
    """Builds the shapely polygon (lon/lat) for an H3 cell."""
    boundary = h3.cell_to_boundary(cell)  # sequence of (lat, lng)
    return Polygon([(lng, lat) for lat, lng in boundary])


def build_grid() -> gpd.GeoDataFrame:
    """Builds the H3 grid filtered by centroid within the boundary."""
    boundary = gpd.read_file(config.BOUNDARY_PATH)
    geometry = boundary.geometry.iloc[0]

    cells = _cells_covering(geometry, config.H3_RESOLUTION)
    logger.info("H3 cells covering the boundary bbox: %d", len(cells))

    # Prepare the unioned polygon for the centroid containment test.
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
    logger.info("Hexagons with centroid inside Bogota: %d", len(grid))
    return grid


def count_positives(grid: gpd.GeoDataFrame) -> int:
    """Counts hexagons that contain at least one D1 POI (positives)."""
    if not config.POIS_D1_PATH.exists():
        logger.warning("%s does not exist; positives will not be counted", config.POIS_D1_PATH.name)
        return -1
    d1 = gpd.read_file(config.POIS_D1_PATH)
    joined = gpd.sjoin(d1, grid[["h3_index", "geometry"]], predicate="within", how="inner")
    return joined["h3_index"].nunique()


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    grid = build_grid()
    grid.to_file(config.GRID_PATH, driver="GeoJSON")
    logger.info("Grid saved -> %s", config.GRID_PATH.name)

    n_total = len(grid)
    n_pos = count_positives(grid)
    logger.info("=" * 50)
    logger.info("Total hexagons:            %d", n_total)
    logger.info("Hexagons with D1 (positives before spatial CV): %d", n_pos)
    if n_pos > 0:
        logger.info("Positive ratio:            %.3f%%", 100 * n_pos / n_total)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
