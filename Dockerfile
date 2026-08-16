# Inference API image (FastAPI). Serving decoupled from PostGIS:
# only needs the versioned artifacts (parquet rankings, .joblib, GeoJSON POIs).
# Target host: Render (free tier) or Hugging Face Spaces (Docker). See docs/despliegue.md.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal serving dependencies (lightweight image; does NOT install osmnx/geopandas/postgis).
RUN pip install \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.29" \
    "pydantic>=2.0" \
    "pandas>=2.0" \
    "pyarrow>=15.0" \
    "scikit-learn>=1.4" \
    "h3>=4.0" \
    "joblib>=1.3"

# Code + artifacts needed to serve.
COPY src/ ./src/
COPY data/processed/ ./data/processed/
# POIs (GeoJSON) for the frontend overlays; optional.
COPY data/raw/pois_*.geojson ./data/raw/

# Default served model (override on the host).
ENV SERVING_MODEL=v3

# Render/HF inject $PORT; defaults to 8000 locally.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
