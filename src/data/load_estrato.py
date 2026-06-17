"""ETAPA 3 (opcional) — Adquisicion del estrato socioeconomico por manzana (IDECA Bogota).

El estrato NO viene en el censo DANE (CNPV/MGN). Es una capa aparte que publica la
Secretaria Distrital de Planeacion / IDECA ("Manzana Estratificacion") con el estrato
(1-6) por manzana urbana de Bogota. Es una senal muy relevante para el look-alike:
Tiendas D1 es hard-discount con foco en estratos 1-3.

Como el censo, su distribucion es via geoportal con interfaz JavaScript / servicios
ArcGIS que cambian de ruta, por lo que la descarga 100% programatica no es confiable.
Este script:

  1. Intenta una descarga best-effort desde URLs/servicios candidatos conocidos.
  2. Si falla, imprime instrucciones de descarga MANUAL y termina sin error
     (NO bloquea el pipeline: las features demograficas son opcionales).

Una vez tengas la capa de manzanas con estrato localmente, colocala como uno de:
    data/raw/estrato_bogota.gpkg       (recomendado)
    data/raw/estrato_bogota.geojson
    data/raw/ManzanaEstratificacion.shp   (+ .dbf/.shx/.prj)
y vuelve a correr:  uv run python -m src.data.db       (cargara la tabla manzanas_estrato)
                    uv run python -m src.data.features

Ejecutar:
    uv run python -m src.data.load_estrato
"""

from __future__ import annotations

import requests

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)

# Servicios candidatos (pueden cambiar; los portales reorganizan rutas periodicamente).
# Se intenta una consulta GeoJSON a un FeatureServer de ArcGIS (formato estable cuando
# el servicio existe). Si la capa cambia de id, usar la descarga manual.
CANDIDATE_URLS = [
    # Datos Abiertos Bogota — exportacion GeoJSON del dataset de estratificacion.
    "https://datosabiertos.bogota.gov.co/dataset/manzana-estratificacion",
]

MANUAL_INSTRUCTIONS = f"""
============== DESCARGA MANUAL DEL ESTRATO (IDECA / SDP Bogota) ==============
La descarga programatica no fue posible. Sigue estos pasos (una sola vez):

1. Abre el portal de Datos Abiertos de Bogota o el geoportal de IDECA:
   - https://datosabiertos.bogota.gov.co/dataset/manzana-estratificacion
   - https://www.ideca.gov.co/  (buscar "Manzana Estratificacion")

2. Descarga la capa "Manzana Estratificacion" (Shapefile / GeoPackage / GeoJSON).
   Debe contener un atributo de estrato por manzana (valores 1-6; el 0 suele ser
   "sin estrato"/no residencial — se trata como nulo en features.py).

3. Descomprime si aplica.

4. Coloca la capa en data/raw/ con uno de estos nombres:
   {chr(10).join('     - ' + p.name for p in config.ESTRATO_PATH_CANDIDATES)}

5. Recarga a PostGIS y recalcula features:
     uv run python -m src.data.db
     uv run python -m src.data.features
=============================================================================
"""


def try_download() -> bool:
    """Intenta descargar la capa de estrato desde las URLs candidatas. True si lo logra.

    Nota: solo se acepta si la respuesta parece un GeoJSON (FeatureCollection); las
    paginas HTML de los portales se descartan para no guardar basura.
    """
    target = config.DATA_RAW / "estrato_bogota.geojson"
    headers = {"User-Agent": config.USER_AGENT}
    for url in CANDIDATE_URLS:
        try:
            logger.info("Intentando descarga: %s", url)
            resp = requests.get(url, headers=headers, timeout=120)
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            text_head = resp.text[:200].lstrip()
            looks_geojson = "json" in ctype.lower() or text_head.startswith("{")
            if not looks_geojson or "FeatureCollection" not in resp.text[:2000]:
                logger.warning("La respuesta de %s no parece GeoJSON; se descarta.", url)
                continue
            target.write_bytes(resp.content)
            logger.info("Descargado -> %s", target.name)
            return True
        except requests.RequestException as exc:
            logger.warning("Fallo la descarga de %s (%s)", url, type(exc).__name__)
    return False


def main() -> None:
    existing = next((p for p in config.ESTRATO_PATH_CANDIDATES if p.exists()), None)
    if existing:
        logger.info("Ya existe un archivo de estrato local: %s. Nada que hacer.", existing.name)
        return
    if not try_download():
        logger.warning("No se pudo descargar automaticamente el estrato.")
        print(MANUAL_INSTRUCTIONS)


if __name__ == "__main__":
    main()
