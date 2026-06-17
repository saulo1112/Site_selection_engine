# Site Selection Engine

Motor de ranking de ubicaciones candidatas para expansión retail/franquicias en
Colombia. Dado un sector de negocio y una ciudad, genera un ranking de hexágonos H3
basado en densidad de competencia/complementarios (OpenStreetMap), variables
demográficas (DANE — Censo 2018) y accesibilidad vial.

## Relación con el portafolio

Segundo proyecto del portafolio de Geospatial AI / Environmental Data Science,
siguiendo el estándar de calidad establecido en **EUDR Forest Risk Assessment Tool**
(pipeline geoespacial con Earth Engine, PostGIS, ML con iteración honesta v1→v2→v3,
FastAPI, dashboard, Docker, README y docs de metodología completos):

- Definición de problema con **valor de negocio real**.
- Justificación de datos con **evidencia**, no a ojo (ver
  [docs/seleccion_area_estudio.md](docs/seleccion_area_estudio.md)).
- Metodología **honesta**, documentando limitaciones explícitamente (ver
  [docs/metodologia.md](docs/metodologia.md)).
- Chequeo **explícito de leakage** (autocorrelación espacial → spatial CV en v3).

## Enfoque

- **Unidad de análisis**: grid hexagonal **H3** sobre la ciudad (a diferencia del
  paper de referencia, que usa centros comerciales discretos).
- **Marca de referencia (look-alike)**: **Tiendas D1** — expansión agresiva y buen
  etiquetado en OSM (`shop=supermarket` + `brand=D1`).
- **Iteración del modelo**:
  - **v1** — MCDA baseline (scoring ponderado, sin ML).
  - **v2** — clasificador look-alike naïve (split aleatorio train/test).
  - **v3** — mismo modelo con **spatial CV** (excluyendo buffers espaciales) para
    corregir el leakage por autocorrelación espacial detectado en v2.
  - **v4** — v3 + **features demográficas** (censo DANE + estrato IDECA); aísla el
    aporte de la demografía comparando, bajo idéntico spatial CV, predictores
    sin/con demografía.
- **Paper de referencia**: Lu, *et al.* (2024). *Retail store location screening: A
  machine learning-based approach.* Journal of Retailing and Consumer Services.
  Tomamos el framework de screening por etapas, las métricas de evaluación (NDCG,
  top-K hitting, top-K loss) y la idea de ensemble secuencial como posible mejora de
  v3. Diferencia clave: su variable objetivo es ingreso real (ASMR); la nuestra es
  una etiqueta look-alike (proxy D1), por lo que nuestros scores son de
  *similitud/prioridad de exploración*, no predicción de desempeño. Ver comparación
  completa en [docs/metodologia.md](docs/metodologia.md).

## Estado actual

- **Ciudad de estudio: Bogotá** (decidida con datos — ver
  [docs/seleccion_area_estudio.md](docs/seleccion_area_estudio.md)). Bogotá es la
  única de las cuatro ciudades candidatas (Bogotá, Cali, Medellín, Barranquilla) con
  señal D1 suficiente (129 tiendas) para garantizar positivos viables en el
  clasificador look-alike tras separación espacial, y cuenta con la mejor
  disponibilidad documentada de datos de estratificación (IDECA).
- **Stack de despliegue: DECIDIDO — híbrido FastAPI + Streamlit/pydeck, demo en vivo.**
  La API (`src/api/`) sirve el ranking e inferencia desde **artefactos versionados**
  (desacoplada de PostGIS); la UI (`app/streamlit_app.py`) renderiza los hexágonos con
  `pydeck H3HexagonLayer` y consume la API (con *fallback* a parquet local). API en
  Render/HF Spaces (Docker), UI en Streamlit Community Cloud. Ver
  [docs/despliegue.md](docs/despliegue.md).
- **Features demográficas (DANE + IDECA): IMPLEMENTADO (pendiente de cargar datos).**
  Población/viviendas (censo DANE CNPV 2018, `src/data/load_censo.py`) y estrato
  socioeconómico (IDECA Bogotá, `src/data/load_estrato.py`), prorrateados por manzana en
  `src/data/features.py`. El estrato es señal clave del look-alike (D1 = hard-discount,
  estratos 1-3). Las capas se adquieren manualmente (geoportales con interfaz JS); el
  pipeline no se bloquea si faltan.
- **Pipeline de datos (ETAPAS 1-4): IMPLEMENTADO.** Descarga OSM → grid H3 → PostGIS
  → tabla de features (`data/processed/features.parquet`). Ver abajo.
- **Modelo v1 (MCDA baseline): IMPLEMENTADO.** Score ponderado interpretable, sin ML
  (`src/models/mcda.py`). NDCG@200=0.80, Precision@200=0.77. Ver
  [docs/v1_mcda_resultados.md](docs/v1_mcda_resultados.md).
- **Modelo v2 (clasificador look-alike, Regresión Logística): IMPLEMENTADO.**
  (`src/models/lookalike.py`). Predice `P(tiene_d1=1)`; ROC-AUC=0.78, NDCG@200=0.83. Ver
  [docs/v2_lookalike_resultados.md](docs/v2_lookalike_resultados.md).
- **Modelo v3 (mismo modelo con Spatial CV): IMPLEMENTADO.**
  (`src/models/lookalike_v3.py` + `src/models/spatial_cv.py`). Validación cruzada espacial
  (bloques H3 res-6, 5 folds, buffer 1 anillo) → métricas honestas out-of-fold:
  ROC-AUC=0.79, NDCG@200=0.84. **Hallazgo:** el desempeño NO cayó vs v2 → el leakage
  espacial era menor de lo esperado (la señal no-D1 generaliza). Ver
  [docs/v3_spatial_cv_resultados.md](docs/v3_spatial_cv_resultados.md).
- **Modelo v4 (v3 + demografía): IMPLEMENTADO (a evaluar al cargar datos).**
  (`src/models/lookalike_v4.py`). Reusa el spatial CV de v3 y compara —honestamente—
  predictores sin/con demografía para aislar su aporte; verifica la hipótesis de negocio
  de que el estrato tiene coeficiente negativo (D1 favorece estratos bajos).
  Genera [docs/v4_demografia_resultados.md](docs/v4_demografia_resultados.md) al correr.
- **Iteración honesta de leakage.** Se detectó y corrigió un leakage de features (D1 como
  su propio "competidor"); y se midió —sin forzar la narrativa— que el leakage espacial era
  bajo. Ver [docs/metodologia.md](docs/metodologia.md) §6.

## Estructura del repo

```
src/
  config.py                    # shared: rutas, params H3, tablas, DATABASE_URL, queries
  logging_config.py            # shared: logging estructurado
  data/                        # PIPELINE DE DATOS (etapas 1-4)
    download.py                #   ETAPA 1 — descarga OSM (boundary, POIs, red vial)
    grid.py                    #   ETAPA 2 — grid hexagonal H3 sobre Bogota
    db.py                      #   ETAPA 3 — carga a PostGIS con indices GIST
    load_censo.py              #   ETAPA 3 (opc.) — adquisicion del censo DANE
    load_estrato.py            #   ETAPA 3 (opc.) — adquisicion del estrato IDECA
    features.py                #   ETAPA 4 — tabla de features (SQL espacial)
  models/                      # MODELOS
    metrics.py                 #   metricas de ranking (NDCG, top-K) reutilizables v1-v4
    mcda.py                    #   v1 — MCDA baseline (scoring ponderado, sin ML)
    lookalike.py               #   v2 — clasificador look-alike (Regresion Logistica)
    spatial_cv.py              #   v3 — folds espaciales H3 + buffer (anti-leakage)
    lookalike_v3.py            #   v3 — mismo modelo con Spatial Cross-Validation
    lookalike_v4.py            #   v4 — v3 + demografia (DANE/IDECA), comparacion honesta
  api/                         # SERVING — FastAPI (desacoplado de PostGIS)
    main.py                    #   endpoints /hexes /hex /score /pois /health
    service.py                 #   carga de artefactos versionados (parquet/joblib/geojson)
    schemas.py                 #   modelos Pydantic
  selection/                   # fase previa
    data_availability_check.py #   chequeo de disponibilidad de datos por ciudad
app/
  streamlit_app.py             # FRONTEND — mapa pydeck H3, consume la API (fallback local)
docs/
  seleccion_area_estudio.md    # chequeo data-driven de ciudad (Overpass + DANE)
  metodologia.md               # objetivo, comparación con Lu et al., plan v1-v4, leakage
  features_summary.md          # balance de etiqueta, estadisticas y correlaciones
  v1_mcda_resultados.md        # pesos, anti-leakage y metricas del MCDA baseline
  v2_lookalike_resultados.md   # clases, coeficientes, diagnostico y metricas de la LR
  v3_spatial_cv_resultados.md  # OOF, comparacion v2 vs v3, veredicto de leakage espacial
  v4_demografia_resultados.md  # aporte de la demografia (generado por lookalike_v4.py)
  despliegue.md                # arquitectura de serving y runbook de la demo en vivo
data/
  raw/                         # OSM crudo (no versionado; salvo pois_*.geojson para overlays)
  processed/                   # features.parquet (no versionado; rankings/joblib versionados)
notebooks/                     # exploración (no versionado)
docker-compose.yml             # PostGIS dedicado (puerto host 5433) — solo ETL local
Dockerfile                     # imagen de la API (serving liviano)
render.yaml                    # blueprint de despliegue de la API en Render
```

## Cómo correr el pipeline de datos

```bash
uv sync

# 1. Levantar PostGIS (puerto 5433; el 5432 lo usa otro proyecto)
docker compose up -d

# 2. Pipeline en orden (cada etapa es ejecutable de forma independiente e idempotente)
uv run python -m src.data.download    # ETAPA 1: OSM -> data/raw/
uv run python -m src.data.grid        # ETAPA 2: grid H3 -> data/processed/grid_bogota.geojson
uv run python -m src.data.db          # ETAPA 3: carga a PostGIS (+ indices GIST)
uv run python -m src.data.features    # ETAPA 4: -> data/processed/features.parquet (+ .csv)

# 3. Modelos (no requieren PostGIS; operan sobre features.parquet)
uv run python -m src.models.mcda         # v1 MCDA -> data/processed/mcda_ranking.parquet
uv run python -m src.models.lookalike    # v2 LR   -> data/processed/lookalike_v2_ranking.parquet
uv run python -m src.models.lookalike_v3 # v3 LR + spatial CV -> lookalike_v3_ranking.parquet
uv run python -m src.models.lookalike_v4 # v4 LR + demografia -> lookalike_v4_ranking.parquet
```

La conexión a PostGIS se lee de `DATABASE_URL` (default
`postgresql://postgres:postgres@localhost:5433/site_selection`). Las features
demográficas son opcionales: si el censo/estrato no están cargados, el pipeline no se
bloquea (ver `src/data/load_censo.py` y `src/data/load_estrato.py`).

### Features demográficas (DANE + IDECA)

```bash
uv run python -m src.data.load_censo     # instrucciones de descarga del censo DANE (manzana)
uv run python -m src.data.load_estrato   # instrucciones de descarga del estrato IDECA
# Coloca las capas en data/raw/ (ver instrucciones impresas), luego recarga y recalcula:
uv run python -m src.data.db
uv run python -m src.data.features
uv run python -m src.models.lookalike_v4 # compara, honestamente, v3 (sin demo) vs v4 (con demo)
```

## Demo en vivo (serving): API + frontend

Stack híbrido **FastAPI + Streamlit/pydeck**, serving **desacoplado de PostGIS** (lee
artefactos versionados). Detalles y despliegue en [docs/despliegue.md](docs/despliegue.md).

```bash
# API de inferencia (docs interactivas en /docs)
uv run uvicorn src.api.main:app --reload

# Frontend (otra terminal). Sin API_BASE_URL cae a leer los parquet locales.
API_BASE_URL=http://127.0.0.1:8000 uv run streamlit run app/streamlit_app.py
```

### Chequeo de disponibilidad de datos (fase previa)

```bash
uv run python -m src.selection.data_availability_check
```

Genera `data/raw/overpass_<ciudad>.json` y actualiza la tabla comparativa en
`docs/seleccion_area_estudio.md`.

## Referencias

- Lu, *et al.* (2024). *Retail store location screening: A machine learning-based
  approach.* Journal of Retailing and Consumer Services.
- DANE (2018). Censo Nacional de Población y Vivienda — Marco Geoestadístico
  Nacional. https://geoportal.dane.gov.co/
