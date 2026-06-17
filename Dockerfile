# Imagen de la API de inferencia (FastAPI). Serving desacoplado de PostGIS:
# solo necesita los artefactos versionados (rankings parquet, .joblib, GeoJSON POIs).
# Host objetivo: Render (free tier) o Hugging Face Spaces (Docker). Ver docs/despliegue.md.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias minimas de serving (imagen liviana; NO instala osmnx/geopandas/postgis).
RUN pip install \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.29" \
    "pydantic>=2.0" \
    "pandas>=2.0" \
    "pyarrow>=15.0" \
    "scikit-learn>=1.4" \
    "h3>=4.0" \
    "joblib>=1.3"

# Codigo + artefactos necesarios para servir.
COPY src/ ./src/
COPY data/processed/ ./data/processed/
# POIs (GeoJSON) para los overlays del frontend; opcional.
COPY data/raw/pois_*.geojson ./data/raw/

# Modelo servido por defecto (override en el host).
ENV SERVING_MODEL=v3

# Render/HF inyectan $PORT; default 8000 en local.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
