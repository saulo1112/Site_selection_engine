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
- **Stack de despliegue: PENDIENTE.** Opciones bajo evaluación:
  - **A**: FastAPI + frontend custom + Docker (repetir patrón de EUDR).
  - **B**: PostGIS + Streamlit/Dash.
  - **C**: híbrido FastAPI ligero + Kepler.gl/deck.gl.
- **Pipeline de datos (ETAPAS 1-4): IMPLEMENTADO.** Descarga OSM → grid H3 → PostGIS
  → tabla de features (`data/processed/features.parquet`). Ver abajo.
- **Modelo v1 (MCDA baseline): IMPLEMENTADO.** Score ponderado interpretable, sin ML
  (`src/mcda.py`). NDCG@200=0.85, Precision@200=0.82, excluyendo features con leakage de
  D1. Ver [docs/v1_mcda_resultados.md](docs/v1_mcda_resultados.md).
- **Modelos v2/v3 (look-alike + spatial CV): aún no implementados.**

## Estructura del repo

```
src/
  config.py                    # rutas, parametros H3, tablas, DATABASE_URL, queries
  logging_config.py            # logging estructurado compartido
  data_availability_check.py   # chequeo de disponibilidad de datos por ciudad
  download.py                  # ETAPA 1 — descarga OSM (boundary, POIs, red vial)
  grid.py                      # ETAPA 2 — grid hexagonal H3 sobre Bogota
  db.py                        # ETAPA 3 — carga a PostGIS con indices GIST
  load_censo.py                # ETAPA 3 (opc.) — adquisicion del censo DANE
  features.py                  # ETAPA 4 — tabla de features (SQL espacial)
  metrics.py                   # metricas de ranking (NDCG, top-K) reutilizables v1-v3
  mcda.py                      # MODELO v1 — MCDA baseline (scoring ponderado, sin ML)
docs/
  seleccion_area_estudio.md    # chequeo data-driven de ciudad (Overpass + DANE)
  metodologia.md               # objetivo, comparación con Lu et al., plan v1-v3
  features_summary.md          # balance de etiqueta, estadisticas y correlaciones
  v1_mcda_resultados.md        # pesos, anti-leakage y metricas del MCDA baseline
data/
  raw/                         # OSM crudo: boundary, POIs, red vial (no versionado)
  processed/                   # grid_bogota.geojson, features.parquet (no versionado)
notebooks/                     # exploración (no versionado)
docker-compose.yml             # PostGIS dedicado (puerto host 5433)
```

## Cómo correr el pipeline de datos

```bash
uv sync

# 1. Levantar PostGIS (puerto 5433; el 5432 lo usa otro proyecto)
docker compose up -d

# 2. Pipeline en orden (cada etapa es ejecutable de forma independiente e idempotente)
uv run python -m src.download    # ETAPA 1: OSM -> data/raw/
uv run python -m src.grid        # ETAPA 2: grid H3 -> data/processed/grid_bogota.geojson
uv run python -m src.db          # ETAPA 3: carga a PostGIS (+ indices GIST)
uv run python -m src.features    # ETAPA 4: -> data/processed/features.parquet (+ .csv)

# 3. Modelo v1 — MCDA baseline (no requiere PostGIS; opera sobre features.parquet)
uv run python -m src.mcda        # v1: -> data/processed/mcda_ranking.parquet (+ .csv)
```

La conexión a PostGIS se lee de `DATABASE_URL` (default
`postgresql://postgres:postgres@localhost:5433/site_selection`). Las features
demográficas del DANE son opcionales: si el censo no está cargado, el pipeline no se
bloquea (ver `src/load_censo.py`).

### Chequeo de disponibilidad de datos (fase previa)

```bash
uv run python -m src.data_availability_check
```

Genera `data/raw/overpass_<ciudad>.json` y actualiza la tabla comparativa en
`docs/seleccion_area_estudio.md`.

## Referencias

- Lu, *et al.* (2024). *Retail store location screening: A machine learning-based
  approach.* Journal of Retailing and Consumer Services.
- DANE (2018). Censo Nacional de Población y Vivienda — Marco Geoestadístico
  Nacional. https://geoportal.dane.gov.co/
