"""STAGE 4 — Feature table per hexagon (spatial SQL queries in PostGIS).

For each hexagon in the grid, computes competition, complementary, road network
and (if the census is loaded) demographic features, plus the look-alike label
`tiene_d1`. Saves features.parquet (+ .csv) and a summary in docs/features_summary.md.

Run standalone (requires STAGE 3 loaded):
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

# Candidate population/housing columns in the MGN-CNPV 2018 (vary by version).
CENSO_POP_CANDIDATES = ["stp27_pers", "tp27_perso", "personas", "poblacion", "tp34_1_se"]
# manz_viv = dwellings per block in the MGN-CNPV 2018 (DANE block SHP/GPKG layer).
# The base MGN does not include a person count at the block level; poblacion_estimada will be NULL.
CENSO_VIV_CANDIDATES = ["manz_viv", "tp19_ee_e1", "viviendas", "tp16_hog", "tp9_1_uso", "stp19_vivi"]
# ESTRATO (uppercase): actual name in PostgreSQL when the IDECA layer is loaded with
# geopandas/to_postgis preserving uppercase — column created as "ESTRATO" (quoted).
ESTRATO_COL_CANDIDATES = [
    "ESTRATO", "estrato", "estrato_ur", "cod_estrat", "codigo_estrato", "estrato_no",
]

# Canonical list of numeric features for the summary.
FEATURE_COLS = [
    "n_d1_300m", "n_d1_500m", "dist_d1_km", "n_supermercados_500m",
    "dist_supermercado_km", "n_farmacias_500m", "n_colegios_500m",
    "n_paradas_bus_500m", "n_bancos_atm_500m", "densidad_vial",
    "poblacion_estimada", "viviendas_estimadas", "estrato_promedio",
]

# Features derived directly from D1's location: the label
# `tiene_d1 = (n_d1_300m >= 1)` is a function of these, so they must NOT be used
# as predictors in v2/v3 (target leakage). Documented in the summary.
D1_DERIVED_COLS = ["n_d1_300m", "n_d1_500m", "dist_d1_km"]

# Demographic columns (null if the census / estrato layers are not loaded).
CENSO_FEATURE_COLS = ["poblacion_estimada", "viviendas_estimadas", "estrato_promedio"]


# --------------------------------------------------------------------------- #
# Detection of optional layers
# --------------------------------------------------------------------------- #
def _table_exists(engine: Engine, table: str) -> bool:
    with engine.connect() as conn:
        return bool(conn.execute(text("SELECT to_regclass(:t)"), {"t": table}).scalar())


def _detect_column(engine: Engine, table: str, candidates: list[str]) -> str | None:
    """Returns the ACTUAL column name in the DB (with its exact casing).

    Compares against the candidates in lowercase, but returns the original
    DB name so the SQL can quote it correctly with double quotes.
    """
    with engine.connect() as conn:
        actual = {r[0].lower(): r[0] for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = :t"
        ), {"t": table})}
    return next((actual[c.lower()] for c in candidates if c.lower() in actual), None)


# --------------------------------------------------------------------------- #
# SQL construction
# --------------------------------------------------------------------------- #
def build_features_sql(engine: Engine) -> str:
    has_streets = _table_exists(engine, config.TABLES["streets"])
    has_censo = _table_exists(engine, config.TABLES["manzanas_censo"])
    has_estrato = _table_exists(engine, config.TABLES["manzanas_estrato"])

    # --- Road network ---
    if has_streets:
        densidad_vial = """
        COALESCE((
            SELECT SUM(ST_Length(ST_Intersection(s.geom, g.geom)::geography))
            FROM streets s
            WHERE ST_Intersects(s.geom, g.geom)
        ), 0) / NULLIF(ST_Area(g.geom::geography), 0) AS densidad_vial"""
    else:
        logger.warning("Table 'streets' missing -> densidad_vial = NULL")
        densidad_vial = "NULL::double precision AS densidad_vial"

    # --- Demographics: population / dwellings (area-prorated, EXTENSIVE magnitude) ---
    censo_table = config.TABLES["manzanas_censo"]
    if has_censo:
        pop_col = _detect_column(engine, censo_table, CENSO_POP_CANDIDATES)
        viv_col = _detect_column(engine, censo_table, CENSO_VIV_CANDIDATES)
        logger.info("Census available. Detected columns: poblacion=%s, viviendas=%s",
                    pop_col, viv_col)
        poblacion = _prorate_sum_expr(censo_table, pop_col, "poblacion_estimada")
        viviendas = _prorate_sum_expr(censo_table, viv_col, "viviendas_estimadas")
    else:
        logger.warning("Table 'manzanas_censo' missing -> poblacion/viviendas = NULL "
                       "(see src/data/load_censo.py)")
        poblacion = "NULL::double precision AS poblacion_estimada"
        viviendas = "NULL::double precision AS viviendas_estimadas"

    # --- Demographics: estrato (area-weighted average, INTENSIVE magnitude) ---
    # Estrato is ordinal (1-6): it is not summed but averaged, weighted by the
    # intersection area. Values <=0 (non-residential / no estrato) are ignored.
    estrato_table = config.TABLES["manzanas_estrato"]
    if has_estrato:
        estrato_col = _detect_column(engine, estrato_table, ESTRATO_COL_CANDIDATES)
        logger.info("Estrato available. Detected column: estrato=%s", estrato_col)
        estrato = _prorate_avg_expr(estrato_table, estrato_col, "estrato_promedio")
    else:
        logger.warning("Table 'manzanas_estrato' missing -> estrato_promedio = NULL "
                       "(see src/data/load_estrato.py)")
        estrato = "NULL::double precision AS estrato_promedio"

    return f"""
    SELECT
        g.h3_index,
        g.lat_centroid,
        g.lon_centroid,

        -- Competition
        (SELECT count(*) FROM pois_d1 d
            WHERE ST_DWithin(g.geom::geography, d.geom::geography, {config.BUFFER_300}))
            AS n_d1_300m,
        (SELECT count(*) FROM pois_d1 d
            WHERE ST_DWithin(g.geom::geography, d.geom::geography, {config.BUFFER_500}))
            AS n_d1_500m,
        (SELECT ST_Distance(g.geom::geography, d.geom::geography) / 1000.0
            FROM pois_d1 d ORDER BY g.geom <-> d.geom LIMIT 1) AS dist_d1_km,
        -- NON-D1 competition: D1 is excluded (COALESCE(es_d1,0)=0) because D1 is the
        -- look-alike target, not a competitor to measure. Including it would leak the
        -- label (every positive would have a "supermarket" = D1 itself at <=300m).
        (SELECT count(*) FROM pois_competidores c
            WHERE COALESCE(c.es_d1, 0) = 0
              AND ST_DWithin(g.geom::geography, c.geom::geography, {config.BUFFER_500}))
            AS n_supermercados_500m,
        (SELECT ST_Distance(g.geom::geography, c.geom::geography) / 1000.0
            FROM pois_competidores c
            WHERE COALESCE(c.es_d1, 0) = 0
            ORDER BY g.geom <-> c.geom LIMIT 1)
            AS dist_supermercado_km,

        -- Complementary POIs (500m buffer)
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

        -- Road network
        {densidad_vial},

        -- Demographics
        {poblacion},
        {viviendas},
        {estrato}

    FROM grid g
    """


def _prorate_sum_expr(table: str, col: str | None, alias: str) -> str:
    """Proration of an EXTENSIVE magnitude (population, dwellings): sum weighted by
    the fraction of each block that falls inside the hexagon. No double counting.
    The column name is quoted with double quotes to support both lowercase names
    (manz_viv) and uppercase preserved by geopandas/to_postgis."""
    if col is None:
        return f"NULL::double precision AS {alias}"
    return f"""
    (SELECT COALESCE(SUM(
        m."{col}"::double precision
        * ST_Area(ST_Intersection(m.geom, g.geom)::geography)
        / NULLIF(ST_Area(m.geom::geography), 0)
     ), 0)
     FROM {table} m
     WHERE ST_Intersects(m.geom, g.geom)) AS {alias}"""


def _prorate_avg_expr(table: str, col: str | None, alias: str) -> str:
    """Weighted average of an INTENSIVE magnitude (estrato 1-6): mean of the block
    values weighted by the intersection area with the hexagon. Values <=0
    (non-residential / no estrato) are discarded. Returns NULL if no valid block.
    The column name is quoted with double quotes (supports uppercase from IDECA)."""
    if col is None:
        return f"NULL::double precision AS {alias}"
    return f"""
    (SELECT SUM(m."{col}"::double precision * a) / NULLIF(SUM(a), 0)
     FROM (
        SELECT mm."{col}",
               ST_Area(ST_Intersection(mm.geom, g.geom)::geography) AS a
        FROM {table} mm
        WHERE ST_Intersects(mm.geom, g.geom)
          AND mm."{col}" IS NOT NULL
          AND mm."{col}"::double precision > 0
     ) m) AS {alias}"""


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _demografia_section(df: pd.DataFrame, n_total: int) -> list[str]:
    """Demographic coverage section (DANE census + IDECA estrato), honestly reported.

    Blocks do not cover the whole grid (non-residential/rural zones or no estrato),
    so this explicitly reports how many hexagons are left without data and why.
    """
    present = [c for c in CENSO_FEATURE_COLS if c in df.columns and df[c].notna().any()]
    if not present:
        return [
            "\n## Demographic features\n",
            "_Not available in this run: the `manzanas_censo` (population/dwellings) "
            "and `manzanas_estrato` (estrato) tables were not loaded. "
            "See `src/data/load_censo.py` and `src/data/load_estrato.py` to enable them. "
            "The v4 model includes them; partial NULLs are imputed with the median "
            "(see `src/models/lookalike.py::build_model`)._\n",
        ]

    cov_rows = [
        f"| `{c}` | {df[c].notna().sum()} | "
        f"{df[c].notna().mean() * 100:.1f}% | {df[c].isna().mean() * 100:.1f}% |"
        for c in present
    ]
    out = [
        "\n## Demographic features — coverage\n",
        "DANE census (CNPV/MGN 2018, population/dwellings) + IDECA estrato, prorated "
        "by block<->hexagon intersection area. Blocks do not cover the whole "
        "grid (non-residential/rural zones; estrato 0 = no estrato, treated as "
        "null), so part of the hexagons are left **without data** (NULL). The "
        "v4 model imputes the **median** for those cases instead of discarding them.\n",
        f"_Total hexagons: **{n_total}**._\n",
        "| Feature | Hex with data | % with data | % NULL |",
        "|---|---|---|---|",
        *cov_rows,
    ]
    if "estrato_promedio" in present:
        out.append(
            "\n> **Look-alike hypothesis (to verify in v4):** D1 is a hard-discount chain "
            "focused on low estratos -> `estrato_promedio` is expected to have a "
            "**negative** relationship with `tiene_d1` (the lower the estrato, the more "
            "likely D1 is present). The LR coefficient in v4 will confirm this or not, honestly.\n"
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

    # Residual correlation between non-D1 competitor distance and D1 distance: measures
    # whether the "close to a supermarket" signal is just co-location with D1 (legitimate,
    # not leakage) or whether some residual leak remains after the es_d1=0 fix (see methodology §6.1).
    resid_corr = float("nan")
    if "dist_supermercado_km" in df.columns and "dist_d1_km" in df.columns:
        resid_corr = float(df["dist_supermercado_km"].corr(df["dist_d1_km"]))

    lines = [
        "# Feature table summary\n",
        f"_Generated by `src/data/features.py`. Total hexagons: **{n_total}**._\n",
        "## Balance of the `tiene_d1` label\n",
        f"- Positives (tiene_d1=1): **{n_pos}**",
        f"- Negatives (tiene_d1=0): **{n_neg}**",
        f"- Positive/negative ratio: **{ratio:.4f}** "
        f"({100 * n_pos / n_total:.2f}% positives)\n",
        "> **Modeling note:** imbalanced dataset. Strategies to "
        "consider in v2/v3: `class_weight='balanced'`, ranking metrics "
        "(NDCG, top-K) instead of accuracy, and a calibrated threshold. Spatial separation "
        "(spatial CV, v3) will further reduce the effective positives.\n",
        "> **Leakage note (critical):** the `tiene_d1` label is defined as "
        "`n_d1_300m >= 1`. Therefore the features derived from D1's location "
        f"(`{'`, `'.join(D1_DERIVED_COLS)}`) are direct functions of the label and "
        "**must NOT be used as predictors** in the look-alike model (target leakage): "
        "their high correlation with `tiene_d1` is tautological, not informative. The model "
        "should learn from the competitor, complementary, road network and "
        "demographic features. This is independent of the spatial autocorrelation leakage, "
        "which is addressed with spatial CV in v3.\n",
        "> **Competition note (non-D1):** `n_supermercados_500m` and "
        "`dist_supermercado_km` measure only competitors **other than D1** "
        "(`es_d1 = 0`). Including D1 would introduce leakage: every positive would have a "
        "'supermarket' (D1 itself) at <=300m. See docs/metodologia.md §6.\n",
        f"> **Residual correlation `dist_supermercado_km` <-> `dist_d1_km`**: "
        f"**{resid_corr:.4f}**. Interpreted as genuine co-location (areas with "
        "dense commerce tend to have both D1 and other supermarkets nearby), not "
        "as leakage: D1 was already excluded from `dist_supermercado_km` (previous note). "
        "See docs/metodologia.md §6.1.\n",
        "## Descriptive statistics per feature\n",
        desc.round(4).to_markdown(),
        "\n## Correlation of each feature with `tiene_d1`\n",
        corr.round(4).sort_values(ascending=False).to_frame("corr_con_tiene_d1").to_markdown(),
    ]
    lines += _demografia_section(df, n_total)
    config.FEATURES_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary written -> %s", config.FEATURES_SUMMARY_PATH.name)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    try:
        if not _table_exists(engine, config.TABLES["grid"]):
            raise RuntimeError("Table 'grid' missing. Run 'uv run python -m src.data.db' first.")

        sql = build_features_sql(engine)
        logger.info("Running spatial features query...")
        df = pd.read_sql(text(sql), engine)
        logger.info("Features computed for %d hexagons", len(df))

        # Look-alike label
        df["tiene_d1"] = (df["n_d1_300m"] >= 1).astype(int)

        # Ensure numeric dtype on demographic columns (object->float if all NULL)
        for col in CENSO_FEATURE_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.to_parquet(config.FEATURES_PARQUET_PATH, index=False)
        df.to_csv(config.FEATURES_CSV_PATH, index=False)
        logger.info("Saved -> %s (+ .csv)", config.FEATURES_PARQUET_PATH.name)

        write_summary(df)
        logger.info("STAGE 4 complete.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
