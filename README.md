# Site Selection Engine

Ranking engine for candidate retail/franchise expansion locations in
Colombia. Given a business sector and a city, it generates a ranking of H3
hexagons based on competitor/complementary density (OpenStreetMap),
demographic variables (DANE — 2018 Census), and road accessibility.

## Relationship to the portfolio

Second project in the Geospatial AI / Environmental Data Science portfolio,
following the quality standard set by **EUDR Forest Risk Assessment Tool**
(geospatial pipeline with Earth Engine, PostGIS, ML with honest v1→v2→v3
iteration, FastAPI, dashboard, Docker, README and full methodology docs):

- Problem definition with **real business value**.
- Data justification backed by **evidence**, not guesswork (see
  [docs/seleccion_area_estudio.md](docs/seleccion_area_estudio.md)).
- **Honest** methodology, explicitly documenting limitations (see
  [docs/metodologia.md](docs/metodologia.md)).
- **Explicit leakage check** (spatial autocorrelation → spatial CV in v3).

## Approach

- **Unit of analysis**: **H3** hexagonal grid over the city (unlike the
  reference paper, which uses discrete shopping malls).
- **Reference brand (look-alike)**: **D1 Stores** — aggressive expansion and
  good OSM tagging (`shop=supermarket` + `brand=D1`).
- **Model iteration**:
  - **v1** — MCDA baseline (weighted scoring, no ML).
  - **v2** — naive look-alike classifier (random train/test split).
  - **v3** — same model with **spatial CV** (excluding spatial buffers) to
    correct the spatial-autocorrelation leakage detected in v2.
  - **v4** — v3 + **demographic features** (DANE census + IDECA stratum);
    isolates the contribution of demographics by comparing, under identical
    spatial CV, predictors without/with demographics.
- **Reference paper**: Lu, *et al.* (2024). *Retail store location screening: A
  machine learning-based approach.* Journal of Retailing and Consumer Services.
  We adopt their staged screening framework, evaluation metrics (NDCG,
  top-K hitting, top-K loss) and the idea of a sequential ensemble as a
  possible improvement over v3. Key difference: their target variable is
  actual revenue (ASMR); ours is a look-alike label (D1 proxy), so our
  scores represent *similarity/exploration priority*, not performance
  prediction. See the full comparison in
  [docs/metodologia.md](docs/metodologia.md).

## Current status

- **Study city: Bogotá** (decided with data — see
  [docs/seleccion_area_estudio.md](docs/seleccion_area_estudio.md)). Bogotá
  is the only one of the four candidate cities (Bogotá, Cali, Medellín,
  Barranquilla) with enough D1 signal (129 stores) to guarantee viable
  positives in the look-alike classifier after spatial separation, and has
  the best documented availability of stratification data (IDECA).
- **Deployment stack: DECIDED — hybrid FastAPI + Streamlit/pydeck, live demo.**
  The API (`src/api/`) serves the ranking and inference from **versioned
  artifacts** (decoupled from PostGIS); the UI (`app/streamlit_app.py`)
  renders the hexagons with `pydeck H3HexagonLayer` and consumes the API
  (with a *fallback* to local parquet). API on Render/HF Spaces (Docker), UI
  on Streamlit Community Cloud. See
  [docs/despliegue.md](docs/despliegue.md).
- **Demographic features (DANE + IDECA): IMPLEMENTED (data loading pending).**
  Population/households (DANE CNPV 2018 census, `src/data/load_censo.py`) and
  socioeconomic stratum (IDECA Bogotá, `src/data/load_estrato.py`), prorated
  by city block in `src/data/features.py`. Stratum is a key look-alike
  signal (D1 = hard-discount, strata 1-3). Layers are acquired manually
  (geoportals with a JS interface); the pipeline does not block if they are
  missing.
- **Data pipeline (STAGES 1-4): IMPLEMENTED.** OSM download → H3 grid →
  PostGIS → features table (`data/processed/features.parquet`). See below.
- **Model v1 (MCDA baseline): IMPLEMENTED.** Interpretable weighted score, no
  ML (`src/models/mcda.py`). NDCG@200=0.80, Precision@200=0.77. See
  [docs/v1_mcda_resultados.md](docs/v1_mcda_resultados.md).
- **Model v2 (look-alike classifier, Logistic Regression): IMPLEMENTED.**
  (`src/models/lookalike.py`). Predicts `P(tiene_d1=1)`; ROC-AUC=0.78,
  NDCG@200=0.83. See
  [docs/v2_lookalike_resultados.md](docs/v2_lookalike_resultados.md).
- **Model v3 (same model with Spatial CV): IMPLEMENTED.**
  (`src/models/lookalike_v3.py` + `src/models/spatial_cv.py`). Spatial cross
  validation (H3 res-6 blocks, 5 folds, 1-ring buffer) → honest out-of-fold
  metrics: ROC-AUC=0.79, NDCG@200=0.84. **Finding:** performance did NOT drop
  vs. v2 → spatial leakage was smaller than expected (the non-D1 signal
  generalizes). See
  [docs/v3_spatial_cv_resultados.md](docs/v3_spatial_cv_resultados.md).
- **Model v4 (v3 + demographics): IMPLEMENTED (to be evaluated once data is
  loaded).** (`src/models/lookalike_v4.py`). Reuses v3's spatial CV and
  honestly compares predictors without/with demographics to isolate their
  contribution; verifies the business hypothesis that stratum has a negative
  coefficient (D1 favors lower strata). Generates
  [docs/v4_demografia_resultados.md](docs/v4_demografia_resultados.md) when
  run.
- **Honest leakage iteration.** A feature leakage was detected and fixed (D1
  counted as its own "competitor"); and it was measured — without forcing
  the narrative — that spatial leakage was low. See
  [docs/metodologia.md](docs/metodologia.md) §6.

## Repo structure

```
src/
  config.py                    # shared: paths, H3 params, tables, DATABASE_URL, queries
  logging_config.py            # shared: structured logging
  data/                        # DATA PIPELINE (stages 1-4)
    download.py                #   STAGE 1 — OSM download (boundary, POIs, road network)
    grid.py                    #   STAGE 2 — H3 hexagonal grid over Bogota
    db.py                      #   STAGE 3 — load into PostGIS with GIST indexes
    load_censo.py              #   STAGE 3 (opt.) — DANE census acquisition
    load_estrato.py            #   STAGE 3 (opt.) — IDECA stratum acquisition
    features.py                #   STAGE 4 — features table (spatial SQL)
  models/                      # MODELS
    metrics.py                 #   ranking metrics (NDCG, top-K) reused across v1-v4
    mcda.py                    #   v1 — MCDA baseline (weighted scoring, no ML)
    lookalike.py                #   v2 — look-alike classifier (Logistic Regression)
    spatial_cv.py               #   v3 — spatial H3 folds + buffer (anti-leakage)
    lookalike_v3.py             #   v3 — same model with Spatial Cross-Validation
    lookalike_v4.py             #   v4 — v3 + demographics (DANE/IDECA), honest comparison
  api/                         # SERVING — FastAPI (decoupled from PostGIS)
    main.py                    #   endpoints /hexes /hex /score /pois /health
    service.py                 #   loads versioned artifacts (parquet/joblib/geojson)
    schemas.py                 #   Pydantic models
  selection/                   # earlier phase
    data_availability_check.py #   per-city data availability check
app/
  streamlit_app.py             # FRONTEND — pydeck H3 map, consumes the API (local fallback)
docs/
  seleccion_area_estudio.md    # data-driven city selection check (Overpass + DANE)
  metodologia.md               # objective, comparison with Lu et al., v1-v4 plan, leakage
  features_summary.md          # label balance, statistics and correlations
  v1_mcda_resultados.md        # weights, anti-leakage and MCDA baseline metrics
  v2_lookalike_resultados.md   # classes, coefficients, diagnostics and LR metrics
  v3_spatial_cv_resultados.md  # OOF, v2 vs v3 comparison, spatial leakage verdict
  v4_demografia_resultados.md  # demographic contribution (generated by lookalike_v4.py)
  despliegue.md                # serving architecture and live demo runbook
data/
  raw/                         # raw OSM data (not versioned; except pois_*.geojson for overlays)
  processed/                   # features.parquet (not versioned; rankings/joblib versioned)
notebooks/                     # exploration (not versioned)
docker-compose.yml             # dedicated PostGIS (host port 5433) — local ETL only
Dockerfile                     # API image (lightweight serving)
render.yaml                    # API deployment blueprint on Render
```

## How to run the data pipeline

```bash
uv sync

# 1. Start PostGIS (port 5433; 5432 is used by another project)
docker compose up -d

# 2. Pipeline in order (each stage is independently runnable and idempotent)
uv run python -m src.data.download    # STAGE 1: OSM -> data/raw/
uv run python -m src.data.grid        # STAGE 2: H3 grid -> data/processed/grid_bogota.geojson
uv run python -m src.data.db          # STAGE 3: load into PostGIS (+ GIST indexes)
uv run python -m src.data.features    # STAGE 4: -> data/processed/features.parquet (+ .csv)

# 3. Models (do not require PostGIS; operate on features.parquet)
uv run python -m src.models.mcda         # v1 MCDA -> data/processed/mcda_ranking.parquet
uv run python -m src.models.lookalike    # v2 LR   -> data/processed/lookalike_v2_ranking.parquet
uv run python -m src.models.lookalike_v3 # v3 LR + spatial CV -> lookalike_v3_ranking.parquet
uv run python -m src.models.lookalike_v4 # v4 LR + demographics -> lookalike_v4_ranking.parquet
```

The PostGIS connection is read from `DATABASE_URL` (default
`postgresql://postgres:postgres@localhost:5433/site_selection`). Demographic
features are optional: if the census/stratum data is not loaded, the
pipeline does not block (see `src/data/load_censo.py` and
`src/data/load_estrato.py`).

### Demographic features (DANE + IDECA)

```bash
uv run python -m src.data.load_censo     # DANE census (block-level) download instructions
uv run python -m src.data.load_estrato   # IDECA stratum download instructions
# Place the layers in data/raw/ (see printed instructions), then reload and recompute:
uv run python -m src.data.db
uv run python -m src.data.features
uv run python -m src.models.lookalike_v4 # honestly compares v3 (no demo) vs v4 (with demo)
```

## Live demo (serving): API + frontend

Hybrid **FastAPI + Streamlit/pydeck** stack, serving **decoupled from
PostGIS** (reads versioned artifacts). Details and deployment in
[docs/despliegue.md](docs/despliegue.md).

```bash
# Inference API (interactive docs at /docs)
uv run uvicorn src.api.main:app --reload

# Frontend (another terminal). Without API_BASE_URL it falls back to local parquet.
API_BASE_URL=http://127.0.0.1:8000 uv run streamlit run app/streamlit_app.py
```

### Data availability check (earlier phase)

```bash
uv run python -m src.selection.data_availability_check
```

Generates `data/raw/overpass_<city>.json` and updates the comparison table in
`docs/seleccion_area_estudio.md`.

## References

- Lu, *et al.* (2024). *Retail store location screening: A machine learning-based
  approach.* Journal of Retailing and Consumer Services.
- DANE (2018). National Population and Housing Census — National
  Geostatistical Framework. https://geoportal.dane.gov.co/
