"""Capa de serving (FastAPI) del Site Selection Engine.

Desacoplada de PostGIS: sirve desde artefactos versionados generados por el pipeline
(rankings parquet, modelos .joblib, GeoJSON de POIs). PostGIS solo se usa en el ETL
local. Ver docs/despliegue.md.
"""
