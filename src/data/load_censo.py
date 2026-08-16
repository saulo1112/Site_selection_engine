"""STAGE 3 (optional) — Acquisition of DANE census blocks (MGN + CNPV 2018).

The DANE Marco Geoestadistico Nacional (MGN) is distributed via a geoportal with a
JavaScript interface (not a stable REST API for direct download), so a 100%
programmatic download is not reliable. This script:

  1. Attempts a best-effort download from known candidate URLs.
  2. If it fails, prints clear MANUAL download instructions and exits without error
     (it does NOT block the rest of the pipeline: demographic features are optional).

Once you have the blocks file locally, place it as one of:
    data/raw/manzanas_censo.gpkg   (recommended)
    data/raw/manzanas_censo.geojson
    data/raw/MGN_ANM_MANZANA.shp   (+ .dbf/.shx/.prj)
and rerun:  uv run python -m src.data.db   (will load the manzanas_censo table)
                    uv run python -m src.data.features

Run:
    uv run python -m src.data.load_censo
"""

from __future__ import annotations

import requests

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)

# Candidate URLs (may change; the geoportal reorganizes routes periodically).
CANDIDATE_URLS = [
    # MGN 2018 at national level integrated with CNPV (compressed geopackage).
    "https://geoportal.dane.gov.co/descargas/mgn_2018/MGN2018_INTEGRADO.zip",
]

MANUAL_INSTRUCTIONS = f"""
================ MANUAL CENSUS DOWNLOAD (MGN + CNPV 2018) ================
The programmatic download was not possible. Follow these steps (one time only):

1. Open the DANE geostatistical downloads page:
   https://geoportal.dane.gov.co/servicios/descarga-y-metadatos/datos-geoestadisticos/

2. Select:
   - Product : Marco Geoestadistico Nacional (MGN) integrated with CNPV 2018
   - Level   : MANZANA (block)
   - Filter  : Departamento "11 - Bogota, D.C."
   (CNPV variables at block level: population, dwellings, households.
    NOTE: estrato (socioeconomic stratum) is NOT in the census; see docs/seleccion_area_estudio.md.)

3. Download the shapefile or geopackage and unzip it.

4. Place the blocks layer in data/raw/ under one of these names:
   {chr(10).join('     - ' + p.name for p in config.CENSO_PATH_CANDIDATES)}

5. Reload into PostGIS and recompute features:
     uv run python -m src.data.db
     uv run python -m src.data.features
============================================================================
"""


def try_download() -> bool:
    """Attempts to download the MGN from the candidate URLs. True if it succeeds."""
    target = config.DATA_RAW / "MGN2018_INTEGRADO.zip"
    headers = {"User-Agent": config.USER_AGENT}
    for url in CANDIDATE_URLS:
        try:
            logger.info("Attempting download: %s", url)
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(target, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            logger.info("Downloaded -> %s (unzip it and place the blocks layer)", target.name)
            return True
        except requests.RequestException as exc:
            logger.warning("Download failed for %s (%s)", url, type(exc).__name__)
    return False


def main() -> None:
    existing = next((p for p in config.CENSO_PATH_CANDIDATES if p.exists()), None)
    if existing:
        logger.info("A local census file already exists: %s. Nothing to do.", existing.name)
        return
    if not try_download():
        logger.warning("Could not automatically download the census.")
        print(MANUAL_INSTRUCTIONS)


if __name__ == "__main__":
    main()
