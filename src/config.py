"""Configuration for the Site Selection Engine project.

Defines the candidate cities for the data-availability check,
the Overpass/Nominatim endpoints, and the network courtesy constants.

Cities are delimited by their ADMINISTRATIVE BOUNDARY in OSM (relation of the
municipality/district), not by bounding box, to get honest counts within the
real urban limits (see docs/seleccion_area_estudio.md).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DOCS = PROJECT_ROOT / "docs"

# --- Endpoints ---
# Overpass: primary endpoint + fallback (rotation on 429/timeout).
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# Nominatim: to resolve each city's relation id at runtime.
NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"

# Honest client identification (usage policy of OSM's public APIs).
USER_AGENT = "site-selection-engine/0.1 (portfolio project; contact: saulo.q1112@gmail.com)"

# --- Network courtesy ---
REQUEST_TIMEOUT = 180          # seconds per Overpass query
SLEEP_BETWEEN_QUERIES = 3      # seconds between queries (avoid rate-limit)
SLEEP_BETWEEN_CITIES = 5
MAX_RETRIES = 3
BACKOFF_BASE = 5               # exponential backoff: BACKOFF_BASE * 2**attempt

# --- Candidate cities ---
# osm_relation_id: hardcoded fallback in case Nominatim fails or returns a different entity.
#   The id actually used is verified/logged on each run (transparency).
#   These ids correspond to the municipality/district relation in OSM and can
#   be confirmed at https://www.openstreetmap.org/relation/<id>.
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

# --- Honest threshold for viable positives ---
# The look-alike classifier (v2/v3) uses hexagons with D1 as positives.
# Spatial separation (spatial CV in v3) discards positives within buffers,
# reducing the effective sample. We set a rough minimum of D1 stores so
# enough positives remain after that separation.
MIN_D1_VIABLE = 40

# --- Overpass query templates (body only, no [out:json] header) ---
# {area_id} = 3600000000 + relation_id
OVERPASS_QUERIES = {
    # D1 stores: supermarkets with brand D1 (robust to tag variants).
    "d1": (
        'nwr["shop"="supermarket"]["brand"~"^(D1|Tiendas D1)$",i](area:{area_id});'
    ),
    # Cross-check by name (catches mislabeled D1 stores without a brand tag).
    "d1_by_name": (
        'nwr["shop"="supermarket"]["name"~"D1",i](area:{area_id});'
    ),
    # Ara stores (fallback if D1 is scarce).
    "ara": (
        'nwr["shop"="supermarket"]["brand"~"^Ara$",i](area:{area_id});'
    ),
    # General OSM tagging density: all shop=* POIs.
    "shops_total": (
        'nwr["shop"](area:{area_id});'
    ),
}


# =========================================================================== #
#  DATA PIPELINE — Study city: BOGOTA
#  (decision documented in docs/seleccion_area_estudio.md)
# =========================================================================== #

# --- Study area ---
STUDY_CITY = "Bogota"
STUDY_RELATION_ID = 7426387                       # Bogota, Distrito Capital
STUDY_AREA_ID = 3_600_000_000 + STUDY_RELATION_ID  # area id for Overpass

# --- Pipeline output paths ---
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

# DANE census (optional load, does not block the pipeline). If a local file exists
# (geopackage/shapefile of blocks from MGN-CNPV 2018 for Bogota), db.py loads it.
CENSO_PATH_CANDIDATES = [
    DATA_RAW / "manzanas_censo.gpkg",
    DATA_RAW / "manzanas_censo.geojson",
    DATA_RAW / "MGN_ANM_MANZANA.shp",
]

# IDECA socioeconomic stratum (Bogota). The stratum is NOT in the DANE census: it is
# a separate layer ("Manzana Estratificacion" from IDECA / Bogota Open Data) with the
# stratum (1-6) per block. Optional load, does not block the pipeline (see
# src/data/load_estrato.py). Relevant for the look-alike: D1 is a hard-discount chain
# focused on strata 1-3.
ESTRATO_PATH_CANDIDATES = [
    DATA_RAW / "estrato_bogota.gpkg",
    DATA_RAW / "estrato_bogota.geojson",
    DATA_RAW / "ManzanaEstratificacion.shp",
]

# --- H3 grid ---
H3_RESOLUTION = 9          # ~0.105 km2 per hexagon (neighborhood scale)

# --- Street network (osmnx) ---
STREET_NETWORK_TYPE = "drive"

# --- PostGIS database ---
# Defaults to port 5433: 5432 is used by the EUDR project's container.
# Override with the DATABASE_URL environment variable (see docker-compose.yml).
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/site_selection",
)

# Table names in PostGIS.
TABLES = {
    "pois_d1": "pois_d1",
    "pois_competidores": "pois_competidores",
    "pois_complementarios": "pois_complementarios",
    "grid": "grid",
    "streets": "streets",
    "manzanas_censo": "manzanas_censo",
    "manzanas_estrato": "manzanas_estrato",
}

# --- Feature buffers (meters) ---
BUFFER_300 = 300
BUFFER_500 = 500
BUFFER_1000 = 1000

# --- Competitor (supermarket) brands for pois_competidores ---
# D1 is included to have the complete supermarket map.
COMPETIDOR_BRANDS = [
    "D1", "Ara", "Justo & Bueno", "Exito", "Carulla",
    "Surtimax", "Olimpica",
]

# --- Pipeline Overpass queries (body only, no header) ---
# All restricted to Bogota's administrative area (area:STUDY_AREA_ID).

# D1: brand D1 / Tiendas D1, plus a name-based check.
PIPELINE_QUERY_D1 = (
    'nwr["shop"="supermarket"]["brand"~"^(D1|Tiendas D1)$",i](area:{area_id});'
    'nwr["shop"="supermarket"]["name"~"D1",i](area:{area_id});'
)

# Competitors: supermarkets from the listed brands (includes D1).
# Regex robust to accents (Exito/Éxito, Olimpica/Olímpica).
PIPELINE_QUERY_COMPETIDORES = (
    'nwr["shop"="supermarket"]'
    '["brand"~"D1|Ara|Justo & Bueno|Justo y Bueno|Éxito|Exito|Carulla|Surtimax|Olímpica|Olimpica",i]'
    '(area:{area_id});'
    'nwr["shop"="supermarket"]'
    '["name"~"D1|Ara|Justo & Bueno|Justo y Bueno|Éxito|Exito|Carulla|Surtimax|Olímpica|Olimpica",i]'
    '(area:{area_id});'
)

# Complementary POIs: each block grouped by category (see COMPLEMENTARIO_RULES).
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

# Rules to assign `categoria` to each complementary POI (order matters).
# Each rule: (category, tag_key, accepted_values).
COMPLEMENTARIO_RULES = [
    ("farmacia", "amenity", {"pharmacy"}),
    ("farmacia", "shop", {"pharmacy"}),
    ("colegio", "amenity", {"school", "college", "university"}),
    ("parada_bus", "amenity", {"bus_station"}),
    ("parada_bus", "highway", {"bus_stop"}),
    ("banco_atm", "amenity", {"bank", "atm"}),
]


# =========================================================================== #
#  MODEL v1 — MCDA baseline (weighted scoring, no ML)
#  (methodology in docs/metodologia.md §4; results in docs/v1_mcda_resultados.md)
# =========================================================================== #

# --- v1 output paths ---
MCDA_RANKING_PARQUET_PATH = DATA_PROCESSED / "mcda_ranking.parquet"
MCDA_RANKING_CSV_PATH = DATA_PROCESSED / "mcda_ranking.csv"
MCDA_SUMMARY_PATH = DOCS / "v1_mcda_resultados.md"

# --- Honest evaluation (post-hoc) ---
# K for the ranking metrics (NDCG@K, top-K hitting, top-K loss). K=200 ~ 5.5% of the
# grid (3589 hexagons): a realistic shortlist size for "picking K cells to explore".
# Note: the real positive ratio is ~24.5% (879/3589), well above K -> that's why
# top-K hitting has a ceiling < 1 (see "hitting_ceiling" in v1's write_summary).
TOP_K = 200

# --- Features derived from D1 (LEAKAGE) — EXCLUDED from the MCDA score ---
# The label `tiene_d1 = (n_d1_300m >= 1)` is a direct function of these columns.
# Using them as score input would be tautological leakage (same as in v2/v3).
# Kept in sync with D1_DERIVED_COLS in src/features.py.
MCDA_LEAKAGE_COLS = ["n_d1_300m", "n_d1_500m", "dist_d1_km"]

# --- MCDA a priori weights, by variable group ---
# Defined by business reasoning (NOT fit to the label — that would be ML).
#   - competencia (non-D1): active retail zone = good signal of commercial viability.
#   - complementarios: proxy for foot traffic / urban activity.
#   - accesibilidad_vial: ease of access / visibility.
#   - demografia: local market size (null if the census is not loaded;
#     in that case its weight is spread proportionally across the present groups).
# Each feature carries a direction: +1 (more is better) or -1 (inverted after
# normalizing, e.g. distances: closer is better).
MCDA_GROUP_WEIGHTS = {
    "competencia": 0.35,
    "complementarios": 0.35,
    "accesibilidad_vial": 0.15,
    "demografia": 0.15,
}

# Features that make up each group, with their direction (+1 more-is-better,
# -1 less-is-better). Within each group, the group's weight is split evenly
# across its features.
MCDA_GROUP_FEATURES = {
    "competencia": [
        ("n_supermercados_500m", +1),   # more supermarkets (non-D1) = active retail zone
        ("dist_supermercado_km", -1),   # closer to retail = better
    ],
    "complementarios": [
        ("n_farmacias_500m", +1),
        ("n_colegios_500m", +1),
        ("n_paradas_bus_500m", +1),
        ("n_bancos_atm_500m", +1),
    ],
    "accesibilidad_vial": [
        ("densidad_vial", +1),
    ],
    "demografia": [
        ("poblacion_estimada", +1),     # more potential local market
        ("viviendas_estimadas", +1),
        ("estrato_promedio", -1),       # D1 (hard-discount) targets lower strata
    ],
}


# =========================================================================== #
#  MODELS v2 / v3 — look-alike classifier (ML)
#  (methodology in docs/metodologia.md §4; results in docs/v2_lookalike_resultados.md)
# =========================================================================== #

# --- Classification problem definition ---
# Binary label already computed in features.parquet (STAGE 4):
#   tiene_d1 = 1  -> the hexagon ALREADY has >=1 D1 store within <=300m ("D1-type cell").
#   tiene_d1 = 0  -> the hexagon has no nearby D1.
# The classifier estimates P(tiene_d1=1) from non-D1 features; that probability
# is the look-alike score used for ranking. (See docs/metodologia.md §2 and §5.)
LABEL_COL = "tiene_d1"

# --- Predictor features (anti-leakage) ---
# All numeric columns from features.parquet EXCEPT: D1 leakage columns
# (MCDA_LEAKAGE_COLS), identifiers/geo (h3_index, centroids), and the label itself.
# The filter for 100%-null columns (e.g. demographics without census) runs at runtime.
NON_PREDICTOR_COLS = ["h3_index", "lat_centroid", "lon_centroid", LABEL_COL]
MODEL_PREDICTOR_COLS = [
    "n_supermercados_500m",
    "dist_supermercado_km",
    "n_farmacias_500m",
    "n_colegios_500m",
    "n_paradas_bus_500m",
    "n_bancos_atm_500m",
    "densidad_vial",
    "poblacion_estimada",
    "viviendas_estimadas",
    "estrato_promedio",
]  # NOTE: explicitly excludes MCDA_LEAKAGE_COLS (n_d1_300m/500m/dist_d1_km).

# Demographic subset (DANE census + IDECA stratum). v4 uses it to isolate the
# contribution of demographics: compares the model WITHOUT these columns (= v3's
# predictors) against the model WITH them. They stay empty (NaN) if the layers
# are not loaded.
DEMOGRAPHIC_COLS = [
    "poblacion_estimada",
    "viviendas_estimadas",
    "estrato_promedio",
]

# --- v2 train/test split (stratified RANDOM split, naive on purpose) ---
# The random split mixes neighboring hexagons between train and test -> leakage from
# spatial autocorrelation. This is intentional: v3 fixes it with spatial CV and is
# compared against it.
TEST_SIZE = 0.25
RANDOM_STATE = 42

# --- v2 output paths ---
LOOKALIKE_V2_RANKING_PARQUET_PATH = DATA_PROCESSED / "lookalike_v2_ranking.parquet"
LOOKALIKE_V2_RANKING_CSV_PATH = DATA_PROCESSED / "lookalike_v2_ranking.csv"
LOOKALIKE_V2_MODEL_PATH = DATA_PROCESSED / "lookalike_v2.joblib"
LOOKALIKE_V2_SUMMARY_PATH = DOCS / "v2_lookalike_resultados.md"

# --- v3: Spatial Cross-Validation (fixes v2's spatial leakage) ---
# Each res-9 hexagon is grouped by its H3 parent at a coarser resolution -> whole
# geographic blocks go together into train or test (StratifiedGroupKFold). In
# addition, hexagons within <=SPATIAL_CV_BUFFER_RINGS rings of any test cell are
# excluded from train (spatial buffer). This way train and test share no neighborhood.
SPATIAL_CV_BLOCK_RES = 6        # parent res -> ~24 blocks of ~36 km2 over Bogota
SPATIAL_CV_FOLDS = 5
SPATIAL_CV_BUFFER_RINGS = 1     # H3 rings (grid_disk) excluded from train

# --- v3 output paths ---
LOOKALIKE_V3_RANKING_PARQUET_PATH = DATA_PROCESSED / "lookalike_v3_ranking.parquet"
LOOKALIKE_V3_RANKING_CSV_PATH = DATA_PROCESSED / "lookalike_v3_ranking.csv"
LOOKALIKE_V3_MODEL_PATH = DATA_PROCESSED / "lookalike_v3.joblib"
LOOKALIKE_V3_SUMMARY_PATH = DOCS / "v3_spatial_cv_resultados.md"


# =========================================================================== #
#  MODEL v4 — look-alike with DEMOGRAPHICS (DANE census + IDECA stratum)
#  Same honest scheme as v3 (spatial CV). Isolates the demographic contribution:
#  compares predictors WITHOUT demographics (= v3) vs WITH demographics.
#  (results in docs/v4_demografia_resultados.md)
# =========================================================================== #
LOOKALIKE_V4_RANKING_PARQUET_PATH = DATA_PROCESSED / "lookalike_v4_ranking.parquet"
LOOKALIKE_V4_RANKING_CSV_PATH = DATA_PROCESSED / "lookalike_v4_ranking.csv"
LOOKALIKE_V4_MODEL_PATH = DATA_PROCESSED / "lookalike_v4.joblib"
LOOKALIKE_V4_SUMMARY_PATH = DOCS / "v4_demografia_resultados.md"


# =========================================================================== #
#  SERVING — Inference API (FastAPI) + frontend (Streamlit)
#  Serving at RUNTIME is decoupled from PostGIS: it reads versioned artifacts
#  (parquet rankings, .joblib model, POI GeoJSON). PostGIS is only used in the
#  local ETL (see docker-compose.yml). See docs/despliegue.md.
# =========================================================================== #

# Default model served (key into SERVING_RANKINGS / SERVING_MODELS).
# Override with the SERVING_MODEL environment variable.
SERVING_MODEL = os.environ.get("SERVING_MODEL", "v3")

# Rankings available to serve (precomputed for each model).
SERVING_RANKINGS = {
    "mcda": MCDA_RANKING_PARQUET_PATH,
    "v2": LOOKALIKE_V2_RANKING_PARQUET_PATH,
    "v3": LOOKALIKE_V3_RANKING_PARQUET_PATH,
    "v4": LOOKALIKE_V4_RANKING_PARQUET_PATH,
}
# Score column per model (the name varies between rankings).
SERVING_SCORE_COL = {
    "mcda": "score_mcda",
    "v2": "score_lookalike",
    "v3": "score_lookalike_v3",
    "v4": "score_lookalike_v4",
}
# .joblib models for live inference (POST /score).
SERVING_MODELS = {
    "v2": LOOKALIKE_V2_MODEL_PATH,
    "v3": LOOKALIKE_V3_MODEL_PATH,
    "v4": LOOKALIKE_V4_MODEL_PATH,
}
# POI layers (raw GeoJSON) that the frontend can overlay.
SERVING_POI_LAYERS = {
    "d1": POIS_D1_PATH,
    "competidores": POIS_COMPETIDORES_PATH,
    "complementarios": POIS_COMPLEMENTARIOS_PATH,
}

# CORS for the frontend (Streamlit Cloud or another origin). "*" by default for the
# public demo; restrict in production via the env var SERVING_CORS_ORIGINS (comma-separated).
SERVING_CORS_ORIGINS = [
    o.strip() for o in os.environ.get("SERVING_CORS_ORIGINS", "*").split(",") if o.strip()
]

# Base URL of the API consumed by the Streamlit frontend. Empty -> Streamlit falls
# back to reading the local parquet directly (robust fallback for the demo).
API_BASE_URL = os.environ.get("API_BASE_URL", "")
