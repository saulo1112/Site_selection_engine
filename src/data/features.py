"""ETAPA 4 — Tabla de features por hexagono (queries SQL espaciales en PostGIS).

Por cada hexagono del grid calcula features de competencia, complementarios, red vial
y (si el censo esta cargado) demograficas, mas la etiqueta look-alike `tiene_d1`.
Guarda features.parquet (+ .csv) y un resumen en docs/features_summary.md.

Ejecutar de forma independiente (requiere ETAPA 3 cargada):
    uv run python -m src.data.features
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src import config
from src.data.db import get_engine
from src.logging_config import get_logger

logger = get_logger(__name__)

# Columnas candidatas de poblacion/vivienda en el MGN-CNPV 2018 (varian por version).
CENSO_POP_CANDIDATES = ["stp27_pers", "tp27_perso", "personas", "poblacion", "tp34_1_se"]
CENSO_VIV_CANDIDATES = ["tp19_ee_e1", "viviendas", "tp16_hog", "tp9_1_uso", "stp19_vivi"]
# Columna candidata de estrato (1-6) en la capa de IDECA/SDP (varia por version).
ESTRATO_COL_CANDIDATES = [
    "estrato", "estrato_ur", "cod_estrat", "codigo_estrato", "estrato_no",
]

# Lista canonica de features numericas para el resumen.
FEATURE_COLS = [
    "n_d1_300m", "n_d1_500m", "dist_d1_km", "n_supermercados_500m",
    "dist_supermercado_km", "n_farmacias_500m", "n_colegios_500m",
    "n_paradas_bus_500m", "n_bancos_atm_500m", "densidad_vial",
    "poblacion_estimada", "viviendas_estimadas", "estrato_promedio",
]

# Features derivadas directamente de la ubicacion de D1: la etiqueta
# `tiene_d1 = (n_d1_300m >= 1)` es funcion de estas, por lo que NO deben usarse
# como predictores en v2/v3 (target leakage). Se documenta en el resumen.
D1_DERIVED_COLS = ["n_d1_300m", "n_d1_500m", "dist_d1_km"]

# Columnas demograficas (nulas si el censo / estrato no estan cargados).
CENSO_FEATURE_COLS = ["poblacion_estimada", "viviendas_estimadas", "estrato_promedio"]


# --------------------------------------------------------------------------- #
# Deteccion de capas opcionales
# --------------------------------------------------------------------------- #
def _table_exists(engine: Engine, table: str) -> bool:
    with engine.connect() as conn:
        return bool(conn.execute(text("SELECT to_regclass(:t)"), {"t": table}).scalar())


def _detect_column(engine: Engine, table: str, candidates: list[str]) -> str | None:
    with engine.connect() as conn:
        cols = {r[0].lower() for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = :t"
        ), {"t": table})}
    return next((c for c in candidates if c.lower() in cols), None)


# --------------------------------------------------------------------------- #
# Construccion del SQL
# --------------------------------------------------------------------------- #
def build_features_sql(engine: Engine) -> str:
    has_streets = _table_exists(engine, config.TABLES["streets"])
    has_censo = _table_exists(engine, config.TABLES["manzanas_censo"])
    has_estrato = _table_exists(engine, config.TABLES["manzanas_estrato"])

    # --- Red vial ---
    if has_streets:
        densidad_vial = """
        COALESCE((
            SELECT SUM(ST_Length(ST_Intersection(s.geom, g.geom)::geography))
            FROM streets s
            WHERE ST_Intersects(s.geom, g.geom)
        ), 0) / NULLIF(ST_Area(g.geom::geography), 0) AS densidad_vial"""
    else:
        logger.warning("Tabla 'streets' ausente -> densidad_vial = NULL")
        densidad_vial = "NULL::double precision AS densidad_vial"

    # --- Demografia: poblacion / viviendas (prorrateo por area, magnitud EXTENSIVA) ---
    censo_table = config.TABLES["manzanas_censo"]
    if has_censo:
        pop_col = _detect_column(engine, censo_table, CENSO_POP_CANDIDATES)
        viv_col = _detect_column(engine, censo_table, CENSO_VIV_CANDIDATES)
        logger.info("Censo disponible. Columnas detectadas: poblacion=%s, viviendas=%s",
                    pop_col, viv_col)
        poblacion = _prorate_sum_expr(censo_table, pop_col, "poblacion_estimada")
        viviendas = _prorate_sum_expr(censo_table, viv_col, "viviendas_estimadas")
    else:
        logger.warning("Tabla 'manzanas_censo' ausente -> poblacion/viviendas = NULL "
                       "(ver src/data/load_censo.py)")
        poblacion = "NULL::double precision AS poblacion_estimada"
        viviendas = "NULL::double precision AS viviendas_estimadas"

    # --- Demografia: estrato (promedio ponderado por area, magnitud INTENSIVA) ---
    # El estrato es ordinal (1-6): no se suma sino que se promedia ponderando por el
    # area de interseccion. Los valores <=0 (no residencial / sin estrato) se ignoran.
    estrato_table = config.TABLES["manzanas_estrato"]
    if has_estrato:
        estrato_col = _detect_column(engine, estrato_table, ESTRATO_COL_CANDIDATES)
        logger.info("Estrato disponible. Columna detectada: estrato=%s", estrato_col)
        estrato = _prorate_avg_expr(estrato_table, estrato_col, "estrato_promedio")
    else:
        logger.warning("Tabla 'manzanas_estrato' ausente -> estrato_promedio = NULL "
                       "(ver src/data/load_estrato.py)")
        estrato = "NULL::double precision AS estrato_promedio"

    return f"""
    SELECT
        g.h3_index,
        g.lat_centroid,
        g.lon_centroid,

        -- Competencia
        (SELECT count(*) FROM pois_d1 d
            WHERE ST_DWithin(g.geom::geography, d.geom::geography, {config.BUFFER_300}))
            AS n_d1_300m,
        (SELECT count(*) FROM pois_d1 d
            WHERE ST_DWithin(g.geom::geography, d.geom::geography, {config.BUFFER_500}))
            AS n_d1_500m,
        (SELECT ST_Distance(g.geom::geography, d.geom::geography) / 1000.0
            FROM pois_d1 d ORDER BY g.geom <-> d.geom LIMIT 1) AS dist_d1_km,
        -- Competencia NO-D1: se excluye D1 (COALESCE(es_d1,0)=0) porque D1 es el
        -- objetivo look-alike, no un competidor a medir. Incluirlo filtraria la
        -- etiqueta (todo positivo tendria un "supermercado" = el propio D1 a <=300m).
        (SELECT count(*) FROM pois_competidores c
            WHERE COALESCE(c.es_d1, 0) = 0
              AND ST_DWithin(g.geom::geography, c.geom::geography, {config.BUFFER_500}))
            AS n_supermercados_500m,
        (SELECT ST_Distance(g.geom::geography, c.geom::geography) / 1000.0
            FROM pois_competidores c
            WHERE COALESCE(c.es_d1, 0) = 0
            ORDER BY g.geom <-> c.geom LIMIT 1)
            AS dist_supermercado_km,

        -- Complementarios (buffer 500m)
        (SELECT count(*) FROM pois_complementarios p
            WHERE p.categoria = 'farmacia'
              AND ST_DWithin(g.geom::geography, p.geom::geography, {config.BUFFER_500}))
            AS n_farmacias_500m,
        (SELECT count(*) FROM pois_complementarios p
            WHERE p.categoria = 'colegio'
              AND ST_DWithin(g.geom::geography, p.geom::geography, {config.BUFFER_500}))
            AS n_colegios_500m,
        (SELECT count(*) FROM pois_complementarios p
            WHERE p.categoria = 'parada_bus'
              AND ST_DWithin(g.geom::geography, p.geom::geography, {config.BUFFER_500}))
            AS n_paradas_bus_500m,
        (SELECT count(*) FROM pois_complementarios p
            WHERE p.categoria = 'banco_atm'
              AND ST_DWithin(g.geom::geography, p.geom::geography, {config.BUFFER_500}))
            AS n_bancos_atm_500m,

        -- Red vial
        {densidad_vial},

        -- Demografia
        {poblacion},
        {viviendas},
        {estrato}

    FROM grid g
    """


def _prorate_sum_expr(table: str, col: str | None, alias: str) -> str:
    """Prorrateo de una magnitud EXTENSIVA (poblacion, viviendas): suma ponderada por
    la fraccion de cada manzana que cae dentro del hexagono. Sin doble conteo."""
    if col is None:
        return f"NULL::double precision AS {alias}"
    return f"""
    (SELECT COALESCE(SUM(
        m.{col}::double precision
        * ST_Area(ST_Intersection(m.geom, g.geom)::geography)
        / NULLIF(ST_Area(m.geom::geography), 0)
     ), 0)
     FROM {table} m
     WHERE ST_Intersects(m.geom, g.geom)) AS {alias}"""


def _prorate_avg_expr(table: str, col: str | None, alias: str) -> str:
    """Promedio ponderado de una magnitud INTENSIVA (estrato 1-6): media de los valores
    de manzana ponderada por el area de interseccion con el hexagono. Se descartan
    valores <=0 (no residencial / sin estrato). Devuelve NULL si no hay manzana valida."""
    if col is None:
        return f"NULL::double precision AS {alias}"
    return f"""
    (SELECT SUM(m.{col}::double precision * a) / NULLIF(SUM(a), 0)
     FROM (
        SELECT mm.{col},
               ST_Area(ST_Intersection(mm.geom, g.geom)::geography) AS a
        FROM {table} mm
        WHERE ST_Intersects(mm.geom, g.geom)
          AND mm.{col} IS NOT NULL
          AND mm.{col}::double precision > 0
     ) m) AS {alias}"""


# --------------------------------------------------------------------------- #
# Resumen
# --------------------------------------------------------------------------- #
def _demografia_section(df: pd.DataFrame, n_total: int) -> list[str]:
    """Seccion de cobertura demografica (censo DANE + estrato IDECA), honesta.

    Las manzanas no cubren todo el grid (zonas no residenciales/rurales o sin estrato),
    asi que reporta explicitamente cuantos hexagonos quedan sin dato y por que.
    """
    present = [c for c in CENSO_FEATURE_COLS if c in df.columns and df[c].notna().any()]
    if not present:
        return [
            "\n## Features demograficas\n",
            "_No disponibles en esta corrida: no estaban cargadas las tablas "
            "`manzanas_censo` (poblacion/viviendas) ni `manzanas_estrato` (estrato). "
            "Ver `src/data/load_censo.py` y `src/data/load_estrato.py` para habilitarlas. "
            "El modelo v4 las incluye; con NULL parcial se imputa la mediana "
            "(ver `src/models/lookalike.py::build_model`)._\n",
        ]

    cov_rows = [
        f"| `{c}` | {df[c].notna().sum()} | "
        f"{df[c].notna().mean() * 100:.1f}% | {df[c].isna().mean() * 100:.1f}% |"
        for c in present
    ]
    out = [
        "\n## Features demograficas — cobertura\n",
        "Censo DANE (CNPV/MGN 2018, poblacion/viviendas) + estrato IDECA, prorrateados "
        "por area de interseccion manzana<->hexagono. Las manzanas no cubren todo el "
        "grid (zonas no residenciales/rurales; estrato 0 = sin estrato, tratado como "
        "nulo), por lo que parte de los hexagonos queda **sin dato** (NULL). El modelo "
        "v4 imputa la **mediana** para esos casos en vez de descartarlos.\n",
        f"_Total de hexagonos: **{n_total}**._\n",
        "| Feature | Hex con dato | % con dato | % NULL |",
        "|---|---|---|---|",
        *cov_rows,
    ]
    if "estrato_promedio" in present:
        out.append(
            "\n> **Hipotesis look-alike (a verificar en v4):** D1 es hard-discount con "
            "foco en estratos bajos -> se espera que `estrato_promedio` tenga relacion "
            "**negativa** con `tiene_d1` (a menor estrato, mas probable presencia de D1). "
            "El coeficiente de la LR en v4 lo confirmara o no, honestamente.\n"
        )
    return out


def write_summary(df: pd.DataFrame) -> None:
    n_total = len(df)
    n_pos = int((df["tiene_d1"] == 1).sum())
    n_neg = n_total - n_pos
    ratio = (n_pos / n_neg) if n_neg else float("inf")

    present_feats = [c for c in FEATURE_COLS if c in df.columns and df[c].notna().any()]
    desc = df[present_feats].describe().T
    desc["pct_nulos"] = df[present_feats].isna().mean().mul(100).round(2)
    corr = df[present_feats + ["tiene_d1"]].corr(numeric_only=True)["tiene_d1"].drop("tiene_d1")

    # Correlacion residual entre distancia a competidor (no-D1) y distancia a D1: mide
    # si la senal de "cerca de un supermercado" es solo co-localizacion con D1 (legitima,
    # no leakage) o si quedo alguna fuga residual tras el fix de es_d1=0 (ver metodologia §6.1).
    resid_corr = float("nan")
    if "dist_supermercado_km" in df.columns and "dist_d1_km" in df.columns:
        resid_corr = float(df["dist_supermercado_km"].corr(df["dist_d1_km"]))

    lines = [
        "# Resumen de la tabla de features\n",
        f"_Generado por `src/data/features.py`. Total de hexagonos: **{n_total}**._\n",
        "## Balance de la etiqueta `tiene_d1`\n",
        f"- Positivos (tiene_d1=1): **{n_pos}**",
        f"- Negativos (tiene_d1=0): **{n_neg}**",
        f"- Ratio positivos/negativos: **{ratio:.4f}** "
        f"({100 * n_pos / n_total:.2f}% positivos)\n",
        "> **Nota de modelado:** dataset desbalanceado. Estrategias a "
        "considerar en v2/v3: `class_weight='balanced'`, metricas de ranking "
        "(NDCG, top-K) en vez de accuracy, y umbral calibrado. La separacion espacial "
        "(spatial CV, v3) reducira aun mas los positivos efectivos.\n",
        "> **Nota de leakage (critica):** la etiqueta `tiene_d1` se define como "
        "`n_d1_300m >= 1`. Por lo tanto las features derivadas de la ubicacion de D1 "
        f"(`{'`, `'.join(D1_DERIVED_COLS)}`) son funciones directas de la etiqueta y "
        "**NO deben usarse como predictores** en el modelo look-alike (target leakage): "
        "su alta correlacion con `tiene_d1` es tautologica, no informativa. El modelo "
        "debe aprender de las features de competidores, complementarios, red vial y "
        "demografia. Esto es independiente del leakage espacial por autocorrelacion, "
        "que se aborda con spatial CV en v3.\n",
        "> **Nota de competencia (no-D1):** `n_supermercados_500m` y "
        "`dist_supermercado_km` miden solo competidores **distintos de D1** "
        "(`es_d1 = 0`). Incluir a D1 introduciria leakage: todo positivo tendria un "
        "'supermercado' (el propio D1) a <=300m. Ver docs/metodologia.md §6.\n",
        f"> **Correlacion residual `dist_supermercado_km` <-> `dist_d1_km`**: "
        f"**{resid_corr:.4f}**. Se interpreta como co-localizacion real (zonas con "
        "comercio denso tienden a tener tanto D1 como otros supermercados cerca), no "
        "como leakage: ya se excluyo a D1 de `dist_supermercado_km` (nota anterior). "
        "Ver docs/metodologia.md §6.1.\n",
        "## Estadisticas descriptivas por feature\n",
        desc.round(4).to_markdown(),
        "\n## Correlacion de cada feature con `tiene_d1`\n",
        corr.round(4).sort_values(ascending=False).to_frame("corr_con_tiene_d1").to_markdown(),
    ]
    lines += _demografia_section(df, n_total)
    config.FEATURES_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Resumen escrito -> %s", config.FEATURES_SUMMARY_PATH.name)


# --------------------------------------------------------------------------- #
# Orquestacion
# --------------------------------------------------------------------------- #
def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    try:
        if not _table_exists(engine, config.TABLES["grid"]):
            raise RuntimeError("Tabla 'grid' ausente. Corre 'uv run python -m src.data.db' primero.")

        sql = build_features_sql(engine)
        logger.info("Ejecutando query espacial de features...")
        df = pd.read_sql(text(sql), engine)
        logger.info("Features calculadas para %d hexagonos", len(df))

        # Etiqueta look-alike
        df["tiene_d1"] = (df["n_d1_300m"] >= 1).astype(int)

        # Asegurar dtype numerico en columnas demograficas (object->float si todo NULL)
        for col in CENSO_FEATURE_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.to_parquet(config.FEATURES_PARQUET_PATH, index=False)
        df.to_csv(config.FEATURES_CSV_PATH, index=False)
        logger.info("Guardado -> %s (+ .csv)", config.FEATURES_PARQUET_PATH.name)

        write_summary(df)
        logger.info("ETAPA 4 completa.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
