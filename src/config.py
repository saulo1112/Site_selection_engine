"""Configuracion del proyecto Site Selection Engine.

Define las ciudades candidatas para el chequeo de disponibilidad de datos,
los endpoints de Overpass/Nominatim y las constantes de cortesia de red.

Las ciudades se delimitan por su FRONTERA ADMINISTRATIVA en OSM (relacion del
municipio/distrito), no por bounding box, para obtener conteos honestos dentro
de los limites urbanos reales (ver docs/seleccion_area_estudio.md).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Rutas del proyecto ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DOCS = PROJECT_ROOT / "docs"

# --- Endpoints ---
# Overpass: endpoint principal + fallback (rotacion ante 429/timeout).
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# Nominatim: para resolver el relation id de cada ciudad en runtime.
NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"

# Identificacion honesta del cliente (politica de uso de las APIs publicas de OSM).
USER_AGENT = "site-selection-engine/0.1 (portfolio project; contact: saulo.q1112@gmail.com)"

# --- Cortesia de red ---
REQUEST_TIMEOUT = 180          # segundos por consulta Overpass
SLEEP_BETWEEN_QUERIES = 3      # segundos entre consultas (evitar rate-limit)
SLEEP_BETWEEN_CITIES = 5
MAX_RETRIES = 3
BACKOFF_BASE = 5               # backoff exponencial: BACKOFF_BASE * 2**intento

# --- Ciudades candidatas ---
# osm_relation_id: respaldo hardcoded por si Nominatim falla o devuelve otra entidad.
#   Se verifica/loguea el id realmente usado en cada corrida (transparencia).
#   Estos ids corresponden a la relacion del municipio/distrito en OSM y pueden
#   confirmarse en https://www.openstreetmap.org/relation/<id>.
CITIES: dict[str, dict] = {
    "Bogota": {
        "nominatim_query": "Bogota, Colombia",
        "osm_relation_id": 7426387,   # Bogota, Distrito Capital
        "min_admin_level": 6,
    },
    "Cali": {
        "nominatim_query": "Santiago de Cali, Valle del Cauca, Colombia",
        "osm_relation_id": 7240803,   # Santiago de Cali (municipio)
        "min_admin_level": 6,
    },
    "Medellin": {
        "nominatim_query": "Medellin, Antioquia, Colombia",
        "osm_relation_id": 7426591,   # Medellin (municipio)
        "min_admin_level": 6,
    },
    "Barranquilla": {
        "nominatim_query": "Barranquilla, Atlantico, Colombia",
        "osm_relation_id": 1387841,   # Barranquilla (municipio)
        "min_admin_level": 6,
    },
}

# --- Umbral honesto de positivos viables ---
# El clasificador look-alike (v2/v3) usa los hexagonos con D1 como positivos.
# La separacion espacial (spatial CV en v3) descarta positivos dentro de buffers,
# reduciendo la muestra efectiva. Fijamos un minimo orientativo de tiendas D1 para
# que queden positivos suficientes tras esa separacion.
MIN_D1_VIABLE = 40

# --- Plantillas de consultas Overpass (cuerpo, sin cabecera [out:json]) ---
# {area_id} = 3600000000 + relation_id
OVERPASS_QUERIES = {
    # Tiendas D1: supermercados con brand D1 (robusto a variantes de tag).
    "d1": (
        'nwr["shop"="supermarket"]["brand"~"^(D1|Tiendas D1)$",i](area:{area_id});'
    ),
    # Verificacion cruzada por nombre (captura D1 mal etiquetadas sin brand).
    "d1_by_name": (
        'nwr["shop"="supermarket"]["name"~"D1",i](area:{area_id});'
    ),
    # Tiendas Ara (respaldo si D1 es escaso).
    "ara": (
        'nwr["shop"="supermarket"]["brand"~"^Ara$",i](area:{area_id});'
    ),
    # Densidad general de etiquetado OSM: todos los POIs shop=*.
    "shops_total": (
        'nwr["shop"](area:{area_id});'
    ),
}


# =========================================================================== #
#  PIPELINE DE DATOS — Ciudad de estudio: BOGOTA
#  (decision documentada en docs/seleccion_area_estudio.md)
# =========================================================================== #

# --- Area de estudio ---
STUDY_CITY = "Bogota"
STUDY_RELATION_ID = 7426387                       # Bogota, Distrito Capital
STUDY_AREA_ID = 3_600_000_000 + STUDY_RELATION_ID  # id de area para Overpass

# --- Rutas de salida del pipeline ---
BOUNDARY_PATH = DATA_RAW / "bogota_boundary.geojson"
POIS_D1_PATH = DATA_RAW / "pois_d1.geojson"
POIS_COMPETIDORES_PATH = DATA_RAW / "pois_competidores.geojson"
POIS_COMPLEMENTARIOS_PATH = DATA_RAW / "pois_complementarios.geojson"
STREETS_GRAPH_PATH = DATA_RAW / "bogota_streets.graphml"
DOWNLOAD_LOG_PATH = DATA_RAW / "download_log.json"

GRID_PATH = DATA_PROCESSED / "grid_bogota.geojson"
FEATURES_PARQUET_PATH = DATA_PROCESSED / "features.parquet"
FEATURES_CSV_PATH = DATA_PROCESSED / "features.csv"
FEATURES_SUMMARY_PATH = DOCS / "features_summary.md"

# Censo DANE (carga opcional, no bloquea el pipeline). Si existe un archivo local
# (geopackage/shapefile de manzanas del MGN-CNPV 2018 para Bogota), db.py lo carga.
CENSO_PATH_CANDIDATES = [
    DATA_RAW / "manzanas_censo.gpkg",
    DATA_RAW / "manzanas_censo.geojson",
    DATA_RAW / "MGN_ANM_MANZANA.shp",
]

# --- Grid H3 ---
H3_RESOLUTION = 9          # ~0.105 km2 por hexagono (escala de barrio)

# --- Red vial (osmnx) ---
STREET_NETWORK_TYPE = "drive"

# --- Base de datos PostGIS ---
# Default en puerto 5433: el 5432 lo ocupa el contenedor del proyecto EUDR.
# Override con la variable de entorno DATABASE_URL (ver docker-compose.yml).
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/site_selection",
)

# Nombres de tablas en PostGIS.
TABLES = {
    "pois_d1": "pois_d1",
    "pois_competidores": "pois_competidores",
    "pois_complementarios": "pois_complementarios",
    "grid": "grid",
    "streets": "streets",
    "manzanas_censo": "manzanas_censo",
}

# --- Buffers de features (metros) ---
BUFFER_300 = 300
BUFFER_500 = 500
BUFFER_1000 = 1000

# --- Marcas de competidores (supermercados) para pois_competidores ---
# Se incluye D1 para tener el mapa completo de supermercados.
COMPETIDOR_BRANDS = [
    "D1", "Ara", "Justo & Bueno", "Exito", "Carulla",
    "Surtimax", "Olimpica",
]

# --- Consultas Overpass del pipeline (cuerpo, sin cabecera) ---
# Todas restringidas al area administrativa de Bogota (area:STUDY_AREA_ID).

# D1: brand D1 / Tiendas D1, mas verificacion por nombre.
PIPELINE_QUERY_D1 = (
    'nwr["shop"="supermarket"]["brand"~"^(D1|Tiendas D1)$",i](area:{area_id});'
    'nwr["shop"="supermarket"]["name"~"D1",i](area:{area_id});'
)

# Competidores: supermercados de las marcas listadas (incluye D1).
# Regex robusto a acentos (Exito/Éxito, Olimpica/Olímpica).
PIPELINE_QUERY_COMPETIDORES = (
    'nwr["shop"="supermarket"]'
    '["brand"~"D1|Ara|Justo & Bueno|Justo y Bueno|Éxito|Exito|Carulla|Surtimax|Olímpica|Olimpica",i]'
    '(area:{area_id});'
    'nwr["shop"="supermarket"]'
    '["name"~"D1|Ara|Justo & Bueno|Justo y Bueno|Éxito|Exito|Carulla|Surtimax|Olímpica|Olimpica",i]'
    '(area:{area_id});'
)

# Complementarios: cada bloque agrupado por categoria (ver COMPLEMENTARIO_RULES).
PIPELINE_QUERY_COMPLEMENTARIOS = (
    'nwr["amenity"="pharmacy"](area:{area_id});'
    'nwr["shop"="pharmacy"](area:{area_id});'
    'nwr["amenity"="school"](area:{area_id});'
    'nwr["amenity"="college"](area:{area_id});'
    'nwr["amenity"="university"](area:{area_id});'
    'nwr["amenity"="bus_station"](area:{area_id});'
    'nwr["highway"="bus_stop"](area:{area_id});'
    'nwr["amenity"="bank"](area:{area_id});'
    'nwr["amenity"="atm"](area:{area_id});'
)

# Reglas para asignar `categoria` a cada POI complementario (orden importa).
# Cada regla: (categoria, clave_tag, valores_aceptados).
COMPLEMENTARIO_RULES = [
    ("farmacia", "amenity", {"pharmacy"}),
    ("farmacia", "shop", {"pharmacy"}),
    ("colegio", "amenity", {"school", "college", "university"}),
    ("parada_bus", "amenity", {"bus_station"}),
    ("parada_bus", "highway", {"bus_stop"}),
    ("banco_atm", "amenity", {"bank", "atm"}),
]
