"""STAGE 3 (optional) — Acquisition of socioeconomic stratum per block (IDECA Bogota).

The stratum is NOT included in the DANE census (CNPV/MGN). It's a separate layer
published by the Secretaria Distrital de Planeacion / IDECA ("Manzana Estratificacion")
with the stratum (1-6) per urban block in Bogota. It's a highly relevant signal for
the look-alike: Tiendas D1 is a hard-discount chain focused on strata 1-3.

Like the census, it's distributed via a geoportal with a JavaScript interface / ArcGIS
services whose routes change, so a 100% programmatic download isn't reliable.
This script:

  1. Attempts a best-effort download from known candidate URLs/services.
  2. If it fails, prints MANUAL download instructions and exits without error
     (it does NOT block the pipeline: demographic features are optional).

Once you have the block layer with stratum locally, place it as one of:
    data/raw/estrato_bogota.gpkg       (recommended)
    data/raw/estrato_bogota.geojson
    data/raw/ManzanaEstratificacion.shp   (+ .dbf/.shx/.prj)
and rerun:          uv run python -m src.data.db       (loads the manzanas_estrato table)
                    uv run python -m src.data.features

Run:
    uv run python -m src.data.load_estrato
"""

from __future__ import annotations

import requests

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)

# Candidate services (may change; portals reorganize routes periodically).
# A GeoJSON query is attempted against an ArcGIS FeatureServer (stable format when
# the service exists). If the layer's id changes, use the manual download.
CANDIDATE_URLS = [
    # Datos Abiertos Bogota — GeoJSON export of the stratification dataset.
    "https://datosabiertos.bogota.gov.co/dataset/manzana-estratificacion",
]

MANUAL_INSTRUCTIONS = f"""
============== MANUAL STRATUM DOWNLOAD (IDECA / SDP Bogota) ==============
The programmatic download wasn't possible. Follow these steps (one time only):

1. Open the Bogota Open Data portal or the IDECA geoportal:
   - https://datosabiertos.bogota.gov.co/dataset/manzana-estratificacion
   - https://www.ideca.gov.co/  (search for "Manzana Estratificacion")

2. Download the "Manzana Estratificacion" layer (Shapefile / GeoPackage / GeoJSON).
   It must contain a stratum attribute per block (values 1-6; 0 usually means
   "no stratum"/non-residential — treated as null in features.py).

3. Unzip if needed.

4. Place the layer in data/raw/ with one of these names:
   {chr(10).join('     - ' + p.name for p in config.ESTRATO_PATH_CANDIDATES)}

5. Reload into PostGIS and recompute features:
     uv run python -m src.data.db
     uv run python -m src.data.features
=============================================================================
"""


def try_download() -> bool:
    """Attempts to download the stratum layer from the candidate URLs. True if it succeeds.

    Note: only accepted if the response looks like GeoJSON (FeatureCollection); portal
    HTML pages are discarded so we don't save junk.
    """
    target = config.DATA_RAW / "estrato_bogota.geojson"
    headers = {"User-Agent": config.USER_AGENT}
    for url in CANDIDATE_URLS:
        try:
            logger.info("Attempting download: %s", url)
            resp = requests.get(url, headers=headers, timeout=120)
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            text_head = resp.text[:200].lstrip()
            looks_geojson = "json" in ctype.lower() or text_head.startswith("{")
            if not looks_geojson or "FeatureCollection" not in resp.text[:2000]:
                logger.warning("Response from %s doesn't look like GeoJSON; discarding.", url)
                continue
            target.write_bytes(resp.content)
            logger.info("Downloaded -> %s", target.name)
            return True
        except requests.RequestException as exc:
            logger.warning("Download from %s failed (%s)", url, type(exc).__name__)
    return False


def main() -> None:
    existing = next((p for p in config.ESTRATO_PATH_CANDIDATES if p.exists()), None)
    if existing:
        logger.info("A local stratum file already exists: %s. Nothing to do.", existing.name)
        return
    if not try_download():
        logger.warning("Could not automatically download the stratum data.")
        print(MANUAL_INSTRUCTIONS)


if __name__ == "__main__":
    main()
