"""MODELO v4 — Clasificador look-alike con DEMOGRAFIA (censo DANE + estrato IDECA).

Mismo modelo (Regresion Logistica) y misma validacion HONESTA de v3 (spatial CV con
bloques H3 res-6 + buffer). Lo unico que cambia es el conjunto de predictores: v4 anade
las features demograficas (`poblacion_estimada`, `viviendas_estimadas`, `estrato_promedio`)
que en v3 no existian (censo no cargado).

Pregunta que responde v4, honestamente: **¿la demografia mejora el ranking?**
Para aislar su aporte se evaluan DOS modelos con identico esquema de spatial CV:
  - BASE  : predictores sin demografia  (== predictores de v3)
  - FULL  : BASE + demografia            (== v4)
y se comparan sus metricas OOF. Si la demografia no mueve la aguja, se reporta tal cual
(igual que el hallazgo honesto de v3 sobre el leakage espacial menor de lo esperado).

Manejo de NULL: las manzanas no cubren todo el grid -> demografia con NULL parcial.
`build_model()` (en lookalike.py) incluye un `SimpleImputer(median)` ajustado dentro de
cada fold; no se descartan hexagonos.

Ejecutar (requiere data/processed/features.parquet con demografia cargada):
    uv run python -m src.models.lookalike_v4
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src import config
from src.logging_config import get_logger
from src.models.lookalike import (
    build_model,
    coefficients_table,
    load_features,
    select_predictors,
)
from src.models.lookalike_v3 import evaluate_scores, spatial_cv_oof
from src.models.metrics import ranking_report

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Particion de predictores: base (sin demografia) vs full (con demografia)
# --------------------------------------------------------------------------- #
def split_predictors(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Devuelve (predictores_full, predictores_base, demograficas_presentes).

    `predictores_full` = todos los predictores no-leakage utilizables (incluye las
    demograficas que tengan algun dato). `predictores_base` excluye las demograficas
    (replica el conjunto de v3). `demograficas_presentes` son las que de hecho entraron.
    """
    full = select_predictors(df)  # ya excluye leakage y columnas 100% nulas
    demo_present = [c for c in config.DEMOGRAPHIC_COLS if c in full]
    base = [c for c in full if c not in config.DEMOGRAPHIC_COLS]
    return full, base, demo_present


def _precision_at_k_oof(df: pd.DataFrame, proba_oof: np.ndarray, k: int) -> float:
    """Precision@K honesta sobre las predicciones OOF."""
    return float(
        df.assign(_p=proba_oof).sort_values("_p", ascending=False)
        .head(k)[config.LABEL_COL].mean()
    )


# --------------------------------------------------------------------------- #
# Resumen
# --------------------------------------------------------------------------- #
def write_summary(
    df: pd.DataFrame,
    base_pred: list[str],
    full_pred: list[str],
    demo_present: list[str],
    base_oof: dict,
    full_oof: dict,
    base_prec: float,
    full_prec: float,
    coefs: pd.DataFrame,
) -> None:
    k = config.TOP_K
    cm = full_oof["confusion_matrix"]

    def _delta(a: float, b: float) -> str:
        d = b - a
        return f"{'+' if d >= 0 else ''}{d:.4f}"

    comp_rows = [
        ("ROC-AUC (OOF)", base_oof["roc_auc"], full_oof["roc_auc"]),
        ("PR-AUC (OOF)", base_oof["pr_auc"], full_oof["pr_auc"]),
        (f"NDCG@{k}", base_oof["ranking"]["ndcg_at_k"], full_oof["ranking"]["ndcg_at_k"]),
        (f"Precision@{k}", base_prec, full_prec),
        (f"top-{k} hitting", base_oof["ranking"]["topk_hitting"], full_oof["ranking"]["topk_hitting"]),
    ]

    # Cobertura demografica (las manzanas no cubren todo el grid).
    cov_rows = [
        f"| `{c}` | {df[c].notna().mean() * 100:.1f}% | {df[c].isna().mean() * 100:.1f}% |"
        for c in demo_present
    ]

    lines = [
        "# Resultados v4 — Look-alike con demografia (censo DANE + estrato IDECA)\n",
        f"_Generado por `src/models/lookalike_v4.py`. Total de hexagonos: **{len(df)}**._\n",
        "## Que cambia respecto a v3\n",
        "**Mismo modelo** (Regresion Logistica) y **misma validacion** (spatial CV con "
        "bloques H3 res-6 + buffer 1 anillo). Lo unico nuevo: se anaden features "
        "**demograficas** prorrateadas por manzana — poblacion y viviendas (censo DANE "
        "CNPV 2018) y estrato socioeconomico (IDECA Bogota).\n",
    ]
    if not demo_present:
        lines += [
            "> **AVISO:** no se detectaron features demograficas con datos en "
            "`features.parquet` (censo/estrato no cargados). v4 degenera a v3. Carga las "
            "capas (`src/data/load_censo.py`, `src/data/load_estrato.py`), recalcula "
            "features y vuelve a correr v4 para una comparacion real.\n",
        ]
    else:
        lines += [
            f"Demograficas que entraron al modelo: `{'`, `'.join(demo_present)}`.\n",
            "**Cobertura** (los hexagonos sin manzana residencial quedan NULL y se imputan "
            "con la mediana dentro de cada fold):\n",
            "| Feature | % con dato | % NULL (imputado) |",
            "|---|---|---|",
            *cov_rows,
            "",
        ]

    lines += [
        "## Aporte de la demografia — BASE (sin demo, = v3) vs FULL (con demo, = v4)\n",
        "Ambos evaluados con **identico** esquema de spatial CV (predicciones OOF). "
        "Asi el delta aisla el aporte de la demografia, no del metodo de validacion.\n",
        f"- BASE  ({len(base_pred)} preds): `{'`, `'.join(base_pred)}`",
        f"- FULL  ({len(full_pred)} preds): `{'`, `'.join(full_pred)}`\n",
        "| Metrica | BASE (v3) | FULL (v4) | Δ (v4 - v3) |",
        "|---|---|---|---|",
        *[f"| {nm} | {a:.4f} | {b:.4f} | {_delta(a, b)} |" for nm, a, b in comp_rows],
        "",
        _verdict_text(base_oof, full_oof, demo_present),
        "\n## Diagnostico honesto v4 (predicciones OOF)\n",
        f"- **ROC-AUC**: {full_oof['roc_auc']:.4f} | **PR-AUC**: {full_oof['pr_auc']:.4f}",
        f"- **NDCG@{k}**: {full_oof['ranking']['ndcg_at_k']:.4f} | "
        f"**Precision@{k}**: {full_prec:.4f} | "
        f"**top-{k} hitting**: {full_oof['ranking']['topk_hitting']:.4f}\n",
        "**Matriz de confusion OOF** (umbral 0.5; filas = real, columnas = predicho):\n",
        "| | pred 0 | pred 1 |",
        "|---|---|---|",
        f"| **real 0** | {cm[0, 0]} | {cm[0, 1]} |",
        f"| **real 1** | {cm[1, 0]} | {cm[1, 1]} |",
        "\n**Reporte por clase (OOF):**\n",
        "```",
        full_oof["classification_report"].rstrip(),
        "```\n",
        "## Interpretabilidad — coeficientes del modelo final (full-data)\n",
        "Sobre features estandarizadas (comparables). Signo = direccion del efecto sobre "
        "`P(tipo-D1)`; magnitud = importancia relativa. El ranking de produccion usa el "
        "modelo full-data; las metricas de arriba vienen de las OOF.\n",
        coefs.round(4).to_markdown(index=False),
        _estrato_reading(coefs, demo_present),
        "\n## Limitacion\n",
        "Sigue siendo un score de **similitud / prioridad de exploracion** (no de "
        "desempeno), con el supuesto look-alike de que la localizacion de D1 es buena "
        "(docs/metodologia.md §5). La demografia anade contexto de mercado, no convierte "
        "el score en una prediccion de ventas.\n",
    ]
    config.LOOKALIKE_V4_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Resumen escrito -> %s", config.LOOKALIKE_V4_SUMMARY_PATH.name)


def _verdict_text(base_oof: dict, full_oof: dict, demo_present: list[str]) -> str:
    if not demo_present:
        return (
            "> **Veredicto:** sin demografia cargada, v4 == v3 (no hay nada que comparar)."
        )
    d_ndcg = full_oof["ranking"]["ndcg_at_k"] - base_oof["ranking"]["ndcg_at_k"]
    d_auc = full_oof["roc_auc"] - base_oof["roc_auc"]
    if d_ndcg > 0.01 or d_auc > 0.01:
        return (
            "> **Veredicto:** la demografia **mejora** el desempeno "
            f"(Δ NDCG@K {d_ndcg:+.4f}, Δ ROC-AUC {d_auc:+.4f}). Aporta senal de mercado "
            "(tamano poblacional y estrato) que las features de OSM no capturaban. v4 "
            "pasa a ser el modelo de produccion."
        )
    if d_ndcg < -0.01 or d_auc < -0.01:
        return (
            "> **Veredicto:** la demografia **empeora** el desempeno OOF "
            f"(Δ NDCG@K {d_ndcg:+.4f}, Δ ROC-AUC {d_auc:+.4f}). Posible ruido/redundancia "
            "o cobertura demografica insuficiente. Se mantiene v3 como produccion y se "
            "documenta el resultado (honesto, no se fuerza la mejora)."
        )
    return (
        "> **Veredicto:** la demografia **no mueve materialmente** las metricas "
        f"(Δ NDCG@K {d_ndcg:+.4f}, Δ ROC-AUC {d_auc:+.4f}). Hallazgo honesto: la senal de "
        "competencia/complementarios/vial ya capturaba casi toda la informacion lineal; "
        "la demografia aporta interpretabilidad (p.ej. el coeficiente de estrato) mas que "
        "poder predictivo. Decision de produccion por parsimonia/interpretabilidad."
    )


def _estrato_reading(coefs: pd.DataFrame, demo_present: list[str]) -> str:
    if "estrato_promedio" not in demo_present:
        return ""
    row = coefs.loc[coefs["feature"] == "estrato_promedio"]
    if row.empty:
        return ""
    coef = float(row["coef"].iloc[0])
    signo = "negativa" if coef < 0 else "positiva"
    coherente = "**coherente**" if coef < 0 else "**contraria a la esperada**"
    return (
        f"\n> **Lectura del estrato:** coeficiente de `estrato_promedio` = {coef:.4f} "
        f"(relacion {signo} con P(tipo-D1)). Es {coherente} con la hipotesis de negocio "
        "(D1 es hard-discount con foco en estratos bajos: a menor estrato, mas probable "
        "la presencia de D1)."
    )


# --------------------------------------------------------------------------- #
# Orquestacion
# --------------------------------------------------------------------------- #
def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_features()
    full_pred, base_pred, demo_present = split_predictors(df)
    y = df[config.LABEL_COL].astype(int)

    if not demo_present:
        logger.warning("No hay features demograficas con datos: v4 degenera a v3. "
                       "Carga censo/estrato y recalcula features para una comparacion real.")

    logger.info("Spatial CV: res padre=%d, folds=%d, buffer=%d anillo(s)",
                config.SPATIAL_CV_BLOCK_RES, config.SPATIAL_CV_FOLDS,
                config.SPATIAL_CV_BUFFER_RINGS)

    # OOF con demografia (FULL = v4).
    proba_full, _ = spatial_cv_oof(df, full_pred)
    full_oof = evaluate_scores(y.to_numpy(), proba_full, k=config.TOP_K)
    full_prec = _precision_at_k_oof(df, proba_full, config.TOP_K)
    logger.info("FULL (v4) OOF -> ROC-AUC=%.4f | NDCG@%d=%.4f",
                full_oof["roc_auc"], config.TOP_K, full_oof["ranking"]["ndcg_at_k"])

    # OOF sin demografia (BASE = v3) para aislar el aporte.
    if base_pred and base_pred != full_pred:
        proba_base, _ = spatial_cv_oof(df, base_pred)
    else:
        proba_base = proba_full  # no hay demografia: base == full
    base_oof = evaluate_scores(y.to_numpy(), proba_base, k=config.TOP_K)
    base_prec = _precision_at_k_oof(df, proba_base, config.TOP_K)
    logger.info("BASE (v3) OOF -> ROC-AUC=%.4f | NDCG@%d=%.4f",
                base_oof["roc_auc"], config.TOP_K, base_oof["ranking"]["ndcg_at_k"])

    # Modelo final full-data para el ranking de produccion (con demografia).
    final_model = build_model()
    final_model.fit(df[full_pred], y)
    score_prod = final_model.predict_proba(df[full_pred])[:, 1]
    coefs = coefficients_table(final_model, full_pred)

    ranking = df[["h3_index", "lat_centroid", "lon_centroid", config.LABEL_COL]].copy()
    ranking["score_lookalike_v4"] = score_prod   # produccion (modelo full-data)
    ranking["score_oof"] = proba_full            # honesto (transparencia)
    ranking["rank_lookalike_v4"] = (
        ranking["score_lookalike_v4"].rank(ascending=False, method="first").astype(int)
    )
    ranking = ranking.sort_values("rank_lookalike_v4").reset_index(drop=True)

    ranking.to_parquet(config.LOOKALIKE_V4_RANKING_PARQUET_PATH, index=False)
    ranking.to_csv(config.LOOKALIKE_V4_RANKING_CSV_PATH, index=False)
    joblib.dump(final_model, config.LOOKALIKE_V4_MODEL_PATH)
    logger.info("Ranking -> %s (+ .csv) | modelo -> %s",
                config.LOOKALIKE_V4_RANKING_PARQUET_PATH.name,
                config.LOOKALIKE_V4_MODEL_PATH.name)

    write_summary(df, base_pred, full_pred, demo_present,
                  base_oof, full_oof, base_prec, full_prec, coefs)
    logger.info("v4 (demografia) completo.")


if __name__ == "__main__":
    main()
