# Despliegue — arquitectura de serving y demo en vivo

## Decisión de stack

**Híbrido FastAPI (serving) + Streamlit/pydeck (UI), con demo en vivo.** Se descartaron
las alternativas de un solo componente: una UI Streamlit pura no demuestra capa de
servicio, y un frontend JS custom + Kepler.gl agrega mantenimiento sin valor para el
objetivo de portafolio. El híbrido demuestra separación API/UI sin escribir frontend JS.

## Principio rector: serving desacoplado de PostGIS

PostGIS es una dependencia de **ETL local** (cálculo de features en `src/data/`), **no**
de runtime. La API sirve desde **artefactos versionados** generados por el pipeline:

| Artefacto | Generado por | Usado por |
|---|---|---|
| `data/processed/lookalike_v*_ranking.parquet` | `src/models/lookalike_v*.py` | API `/hexes`, `/hex` |
| `data/processed/lookalike_v*.joblib` | idem | API `/score` (inferencia en vivo) |
| `data/processed/features.parquet` | `src/data/features.py` | API `/hex` (detalle), `/score` por h3 |
| `data/raw/pois_*.geojson` | `src/data/download.py` | API `/pois/{name}` (overlays) |

Esto permite hostear ambos servicios barato (sin base de datos administrada). Los
artefactos son pequeños (KB) y se versionan (excepciones en `.gitignore`); ver
"Artefactos de serving versionados" abajo.

## Componentes

```
                 ┌─────────────────────────┐
   navegador ───▶│ Streamlit Cloud (UI)    │   app/streamlit_app.py
                 │  pydeck H3HexagonLayer   │   (mapa, filtros, overlays)
                 └───────────┬─────────────┘
                             │ HTTP (API_BASE_URL, secret)
                             ▼
                 ┌─────────────────────────┐
                 │ FastAPI (Render/HF)      │   src/api/main.py
                 │  /hexes /hex /score /pois│   lee artefactos versionados
                 └─────────────────────────┘
                             ▲
                             │ (build-time, local)
                 ┌─────────────────────────┐
                 │ ETL local + PostGIS      │   src/data/* (docker-compose)
                 │  genera los artefactos   │   src/models/*
                 └─────────────────────────┘
```

**Fallback de robustez:** si `API_BASE_URL` no está configurado o la API no responde,
la app Streamlit lee los parquet locales directamente (`src/api/service.py`), de modo que
la demo corre en un solo proceso sin la API.

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado + modelos disponibles + nº de hexágonos |
| GET | `/models` | Modelos con ranking precomputado |
| GET | `/hexes?model=&top_k=&min_score=&bbox=` | Hexágonos rankeados (mapa) |
| GET | `/hex/{h3_index}?model=` | Detalle: score, rank, features, geometría del anillo |
| POST | `/score` | Inferencia en vivo con el `.joblib` (por `h3_index` o `features`) |
| GET | `/pois/{name}` | GeoJSON de POIs (`d1`/`competidores`/`complementarios`) |

`SERVING_MODEL` (env) fija el modelo por defecto (`v3` hasta cargar demografía; `v4`
después). Docs interactivas en `/docs`.

## Correr local

```bash
uv sync
# API
uv run uvicorn src.api.main:app --reload          # http://127.0.0.1:8000/docs
# UI (en otra terminal). Sin API_BASE_URL usa el fallback a parquet local.
API_BASE_URL=http://127.0.0.1:8000 uv run streamlit run app/streamlit_app.py
```

## Demo en vivo

### API → Render (free tier) o Hugging Face Spaces (Docker)
- `Dockerfile` instala solo dependencias de serving (sin osmnx/geopandas/PostGIS) →
  imagen liviana. Copia `src/`, `data/processed/` y `data/raw/pois_*.geojson`.
- Render: usar `render.yaml` (blueprint). Health check en `/health`. Inyecta `$PORT`.
- HF Spaces: crear Space tipo *Docker*; expone el puerto del `CMD`.

### UI → Streamlit Community Cloud
- App principal: `app/streamlit_app.py`.
- Secret `API_BASE_URL` = URL pública de la API (ver `.streamlit/secrets.toml.example`).
- En la API, fijar `SERVING_CORS_ORIGINS` al dominio de la app Streamlit.

## Artefactos de serving versionados

`data/processed/*` y `data/raw/*` están en `.gitignore`, pero los artefactos de serving
(rankings parquet, `.joblib`, `pois_*.geojson`) se **exceptúan** explícitamente para que
la imagen Docker y Streamlit Cloud corran sin ejecutar el ETL. Son de tamaño KB. Si en el
futuro crecen, migrar a un *release asset* descargado en el `startup` de la API.

## Qué NO se necesita en producción

`docker-compose.yml` (PostGIS) y las dependencias pesadas (`osmnx`, `geopandas`,
`psycopg2`) son solo para regenerar features/artefactos localmente. La API y la UI no las
requieren.
