"""ETAPA 3 (opcional) — Adquisicion de manzanas del censo DANE (MGN + CNPV 2018).

El Marco Geoestadistico Nacional (MGN) del DANE se distribuye desde un geoportal con
interfaz JavaScript (no una API REST estable de descarga directa), por lo que la
descarga 100% programatica no es confiable. Este script:

  1. Intenta una descarga best-effort desde URLs candidatas conocidas.
  2. Si falla, imprime instrucciones claras de descarga MANUAL y termina sin error
     (NO bloquea el resto del pipeline: las features demograficas son opcionales).

Una vez tengas el archivo de manzanas localmente, colocalo como uno de:
    data/raw/manzanas_censo.gpkg   (recomendado)
    data/raw/manzanas_censo.geojson
    data/raw/MGN_ANM_MANZANA.shp   (+ .dbf/.shx/.prj)
y vuelve a correr:  uv run python -m src.db   (cargara la tabla manzanas_censo)
                    uv run python -m src.features

Ejecutar:
    uv run python -m src.load_censo
"""

from __future__ import annotations

import requests

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)

# URLs candidatas (pueden cambiar; el geoportal reorganiza rutas periodicamente).
CANDIDATE_URLS = [
    # MGN 2018 a nivel nacional integrado con CNPV (geopackage comprimido).
    "https://geoportal.dane.gov.co/descargas/mgn_2018/MGN2018_INTEGRADO.zip",
]

MANUAL_INSTRUCTIONS = f"""
================ DESCARGA MANUAL DEL CENSO (MGN + CNPV 2018) ================
La descarga programatica no fue posible. Sigue estos pasos (una sola vez):

1. Abre la pagina de descargas geoestadisticas del DANE:
   https://geoportal.dane.gov.co/servicios/descarga-y-metadatos/datos-geoestadisticos/

2. Selecciona:
   - Producto : Marco Geoestadistico Nacional (MGN) integrado con CNPV 2018
   - Nivel    : MANZANA
   - Filtro   : Departamento "11 - Bogota, D.C."
   (Variables del CNPV a nivel manzana: poblacion, viviendas, hogares.
    NOTA: el estrato NO viene en el censo; ver docs/seleccion_area_estudio.md.)

3. Descarga el shapefile o geopackage y descomprime.

4. Coloca la capa de manzanas en data/raw/ con uno de estos nombres:
   {chr(10).join('     - ' + p.name for p in config.CENSO_PATH_CANDIDATES)}

5. Recarga a PostGIS y recalcula features:
     uv run python -m src.db
     uv run python -m src.features
============================================================================
"""


def try_download() -> bool:
    """Intenta descargar el MGN desde las URLs candidatas. True si lo logra."""
    target = config.DATA_RAW / "MGN2018_INTEGRADO.zip"
    headers = {"User-Agent": config.USER_AGENT}
    for url in CANDIDATE_URLS:
        try:
            logger.info("Intentando descarga: %s", url)
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(target, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            logger.info("Descargado -> %s (descomprime y coloca la capa de manzanas)", target.name)
            return True
        except requests.RequestException as exc:
            logger.warning("Fallo la descarga de %s (%s)", url, type(exc).__name__)
    return False


def main() -> None:
    existing = next((p for p in config.CENSO_PATH_CANDIDATES if p.exists()), None)
    if existing:
        logger.info("Ya existe un archivo de censo local: %s. Nada que hacer.", existing.name)
        return
    if not try_download():
        logger.warning("No se pudo descargar automaticamente el censo.")
        print(MANUAL_INSTRUCTIONS)


if __name__ == "__main__":
    main()
