"""MODELO v3 — Clasificador look-alike con SPATIAL CROSS-VALIDATION.

Mismo modelo que v2 (Regresion Logistica), pero evaluado con validacion cruzada
ESPACIAL en vez de un split aleatorio. Objetivo: una estimacion HONESTA de
generalizacion y cuantificar cuanto del desempeno de v2 era leakage por autocorrelacion
espacial (celdas vecinas repartidas entre train y test).

Como funciona (ver src/models/spatial_cv.py):
  - Bloques espaciales = padre H3 a resolucion gruesa; bloques enteros van a train o test.
  - StratifiedGroupKFold reparte bloques en folds (balanceando positivos).
  - Buffer: se excluyen del train las celdas a <=k anillos de cualquier celda de test.
  - Se ensamblan predicciones OUT-OF-FOLD (OOF): cada celda la predice un modelo que NO
    vio su vecindario -> las metricas OOF son la evaluacion honesta de v3.

Anti-leakage (tres capas): (1) predictores sin columnas D1 y competencia solo no-D1;
(2) StandardScaler ajustado dentro de cada fold (Pipeline) -> sin fuga de escalado;
(3) bloques espaciales + buffer -> train y test no comparten vecindario.

Ejecutar (requiere data/processed/features.parquet de la ETAPA 4):
    uv run python -m src.models.lookalike_v3
"""

from __future__ import annotations

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src import config
from src.logging_config import get_logger
from src.models.lookalike import (
    build_model,
    coefficients_table,
    load_features,
    select_predictors,
)
from src.models.metrics import ranking_report
from src.models.spatial_cv import assign_spatial_blocks, buffered_spatial_folds

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Evaluacion generica de un vector de scores contra la etiqueta
# --------------------------------------------------------------------------- #
def evaluate_scores(y_true: npt.ArrayLike, proba: npt.ArrayLike, k: int) -> dict:
    """Diagnostico de clasificacion (umbral 0.5) + metricas de ranking sobre un score."""
    y = np.asarray(y_true, dtype=np.int_)
    p = np.asarray(proba, dtype=np.float64)
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "confusion_matrix": confusion_matrix(y, pred),
        "classification_report": classification_report(
            y, pred, digits=4,
            target_names=["clase_0 (sin D1)", "clase_1 (tipo-D1)"], zero_division=0,
        ),
        "ranking": ranking_report(p, y, k=k),
    }


# --------------------------------------------------------------------------- #
# Predicciones out-of-fold con spatial CV
# --------------------------------------------------------------------------- #
def spatial_cv_oof(
    df: pd.DataFrame, predictors: list[str],
) -> tuple[np.ndarray, list[dict]]:
    """Ensambla predicciones OOF P(clase=1) con folds espaciales + buffer.

    Devuelve (proba_oof alineado con df, info por fold). Verifica que cada celda se
    predice exactamente una vez y que ninguna celda de test esta en su propio train.
    """
    h3_indices = df["h3_index"].tolist()
    X = df[predictors].to_numpy()
    y = df[config.LABEL_COL].astype(int).to_numpy()

    proba_oof = np.full(len(df), np.nan, dtype=np.float64)
    covered = np.zeros(len(df), dtype=bool)
    fold_info: list[dict] = []

    folds = buffered_spatial_folds(
        h3_indices, y,
        n_folds=config.SPATIAL_CV_FOLDS,
        coarse_res=config.SPATIAL_CV_BLOCK_RES,
        buffer_rings=config.SPATIAL_CV_BUFFER_RINGS,
        random_state=config.RANDOM_STATE,
    )
    for i, (train_idx, test_idx, n_removed) in enumerate(folds, start=1):
        # Anti-leakage: ninguna celda de test puede estar en su train.
        assert not set(test_idx) & set(train_idx), "solapamiento train/test en el fold"

        model = build_model()
        model.fit(X[train_idx], y[train_idx])
        proba_oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        covered[test_idx] = True

        info = {
            "fold": i,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_buffer_removidas": n_removed,
            "test_positivos": int(y[test_idx].sum()),
        }
        fold_info.append(info)
        logger.info("Fold %d -> train=%d, test=%d (pos=%d), buffer removio %d celdas",
                    i, info["n_train"], info["n_test"], info["test_positivos"], n_removed)

    if not covered.all():
        raise RuntimeError(f"{(~covered).sum()} celdas sin prediccion OOF (cobertura incompleta).")
    return proba_oof, fold_info


# --------------------------------------------------------------------------- #
# Metricas de v2 (split aleatorio) para la comparacion honesta
# --------------------------------------------------------------------------- #
def v2_random_split_metrics(df: pd.DataFrame, predictors: list[str]) -> dict:
    """Reproduce la evaluacion de v2 (split aleatorio) para comparar contra v3."""
    X = df[predictors]
    y = df[config.LABEL_COL].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y,
    )
    model = build_model()
    model.fit(X_tr, y_tr)
    proba_te = model.predict_proba(X_te)[:, 1]      # diagnostico en test aleatorio
    proba_all = model.predict_proba(X)[:, 1]        # ranking sobre el grid (como v2)
    res = evaluate_scores(y_te, proba_te, k=config.TOP_K)
    res["ranking"] = ranking_report(proba_all, y.to_numpy(), k=config.TOP_K)
    return res


# --------------------------------------------------------------------------- #
# Resumen
# --------------------------------------------------------------------------- #
def write_summary(
    df: pd.DataFrame,
    predictors: list[str],
    oof: dict,
    v2: dict,
    fold_info: list[dict],
    coefs: pd.DataFrame,
    precision_v3: float,
) -> None:
    k = config.TOP_K
    cm = oof["confusion_matrix"]
    n_blocks = len(set(assign_spatial_blocks(df["h3_index"].tolist(),
                                             config.SPATIAL_CV_BLOCK_RES)))

    def _delta(a: float, b: float) -> str:
        d = b - a
        signo = "+" if d >= 0 else ""
        return f"{signo}{d:.4f}"

    fold_rows = [
        f"| {fi['fold']} | {fi['n_train']} | {fi['n_test']} | {fi['test_positivos']} | "
        f"{fi['n_buffer_removidas']} |"
        for fi in fold_info
    ]
    comp_rows = [
        ("ROC-AUC (test/OOF)", v2["roc_auc"], oof["roc_auc"]),
        ("PR-AUC (test/OOF)", v2["pr_auc"], oof["pr_auc"]),
        ("NDCG@%d" % k, v2["ranking"]["ndcg_at_k"], oof["ranking"]["ndcg_at_k"]),
        ("top-%d hitting" % k, v2["ranking"]["topk_hitting"], oof["ranking"]["topk_hitting"]),
    ]

    lines = [
        "# Resultados v3 — Clasificador look-alike con Spatial CV\n",
        f"_Generado por `src/models/lookalike_v3.py`. Total de hexagonos: **{len(df)}**._\n",
        "## Que cambia respecto a v2\n",
        "**Mismo modelo** (Regresion Logistica), **misma definicion de clases** "
        "(clase 1 = celda con D1 a <=300m). Lo unico que cambia es la **validacion**: en "
        "vez de un split aleatorio, validacion cruzada **espacial**. Cada hexagono se "
        "predice *out-of-fold* (OOF) por un modelo que no vio su vecindario -> estimacion "
        "honesta de como generaliza el modelo a zonas nuevas de la ciudad.\n",
        "## Esquema de spatial CV (anti-leakage espacial)\n",
        f"- **Bloques espaciales**: padre H3 a resolucion **{config.SPATIAL_CV_BLOCK_RES}** "
        f"-> {n_blocks} bloques (~36 km2 c/u). Bloques enteros van a train o test.",
        f"- **Folds**: StratifiedGroupKFold, **{config.SPATIAL_CV_FOLDS}** folds "
        "(respeta bloques, balancea positivos).",
        f"- **Buffer**: se excluyen del train las celdas a <=**{config.SPATIAL_CV_BUFFER_RINGS}** "
        "anillo(s) H3 de cualquier celda de test.",
        f"- **Predictores** (sin leakage): `{'`, `'.join(predictors)}`.\n",
        "**Tamanos por fold:**\n",
        "| Fold | Train | Test | Test pos. | Buffer removidas |",
        "|---|---|---|---|---|",
        *fold_rows,
        "\n## Diagnostico honesto (predicciones OOF)\n",
        f"- **ROC-AUC**: {oof['roc_auc']:.4f}",
        f"- **PR-AUC**: {oof['pr_auc']:.4f}",
        f"- **NDCG@{k}**: {oof['ranking']['ndcg_at_k']:.4f} | "
        f"**Precision@{k}**: {precision_v3:.4f} | "
        f"**top-{k} hitting**: {oof['ranking']['topk_hitting']:.4f}\n",
        "**Matriz de confusion OOF** (umbral 0.5; filas = real, columnas = predicho):\n",
        "| | pred 0 | pred 1 |",
        "|---|---|---|",
        f"| **real 0** | {cm[0, 0]} | {cm[0, 1]} |",
        f"| **real 1** | {cm[1, 0]} | {cm[1, 1]} |",
        "\n**Reporte por clase (OOF):**\n",
        "```",
        oof["classification_report"].rstrip(),
        "```\n",
        "## Veredicto de leakage — v2 (split aleatorio) vs v3 (spatial CV)\n",
        "| Metrica | v2 aleatorio | v3 spatial CV | Δ (v3 - v2) |",
        "|---|---|---|---|",
        *[f"| {nm} | {a:.4f} | {b:.4f} | {_delta(a, b)} |" for nm, a, b in comp_rows],
        "",
        _verdict_text(v2, oof),
        "\n## Interpretabilidad — coeficientes del modelo final (full-data)\n",
        "El ranking de produccion usa un modelo reentrenado con **todos** los datos; el "
        "desempeno reportado arriba viene de las OOF (no de este ajuste full-data).\n",
        coefs.round(4).to_markdown(index=False),
        "\n## Limitacion\n",
        "Sigue siendo un score de **similitud / prioridad de exploracion** (no de "
        "desempeno), con el supuesto look-alike de que la localizacion de D1 es buena "
        "(docs/metodologia.md §5). La spatial CV corrige el leakage espacial, no los "
        "sesgos del proxy ni del etiquetado OSM.\n",
    ]
    config.LOOKALIKE_V3_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Resumen escrito -> %s", config.LOOKALIKE_V3_SUMMARY_PATH.name)


def _verdict_text(v2: dict, oof: dict) -> str:
    # Convencion consistente con la tabla: Delta = v3(OOF) - v2(aleatorio).
    d_ndcg = oof["ranking"]["ndcg_at_k"] - v2["ranking"]["ndcg_at_k"]
    d_auc = oof["roc_auc"] - v2["roc_auc"]
    if d_ndcg < -0.02 or d_auc < -0.02:
        return (
            "> **Veredicto:** el desempeno **cae** al pasar a spatial CV "
            f"(Δ NDCG@K {d_ndcg:+.4f}, Δ ROC-AUC {d_auc:+.4f}). Esto **confirma** que "
            "parte de las metricas de v2 estaban infladas por leakage de autocorrelacion "
            "espacial; v3 es la estimacion honesta de generalizacion."
        )
    return (
        "> **Veredicto:** el desempeno **se mantiene** bajo spatial CV "
        f"(Δ NDCG@K {d_ndcg:+.4f}, Δ ROC-AUC {d_auc:+.4f}; v3 incluso iguala o supera "
        "levemente a v2). El leakage por autocorrelacion espacial resulto **menor de lo "
        "esperado**: con un modelo lineal sobre features de buffer (campos espaciales "
        "suaves), un split aleatorio y uno espacial generalizan parecido. Es un hallazgo "
        "valido y honesto — la senal no-D1 (competencia/complementarios/vial) se sostiene "
        "en zonas no vistas, no era un espejismo del split."
    )


# --------------------------------------------------------------------------- #
# Orquestacion
# --------------------------------------------------------------------------- #
def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_features()
    predictors = select_predictors(df)
    y = df[config.LABEL_COL].astype(int)

    logger.info("Spatial CV: res padre=%d, folds=%d, buffer=%d anillo(s)",
                config.SPATIAL_CV_BLOCK_RES, config.SPATIAL_CV_FOLDS,
                config.SPATIAL_CV_BUFFER_RINGS)
    proba_oof, fold_info = spatial_cv_oof(df, predictors)
    oof = evaluate_scores(y.to_numpy(), proba_oof, k=config.TOP_K)
    logger.info("OOF -> ROC-AUC=%.4f | PR-AUC=%.4f | NDCG@%d=%.4f",
                oof["roc_auc"], oof["pr_auc"], config.TOP_K, oof["ranking"]["ndcg_at_k"])

    v2 = v2_random_split_metrics(df, predictors)
    logger.info("v2 (aleatorio) -> ROC-AUC=%.4f | NDCG@%d=%.4f (para comparar)",
                v2["roc_auc"], config.TOP_K, v2["ranking"]["ndcg_at_k"])

    # Modelo final full-data para el ranking de produccion.
    final_model = build_model()
    final_model.fit(df[predictors], y)
    score_prod = final_model.predict_proba(df[predictors])[:, 1]
    coefs = coefficients_table(final_model, predictors)

    ranking = df[["h3_index", "lat_centroid", "lon_centroid", config.LABEL_COL]].copy()
    ranking["score_lookalike_v3"] = score_prod   # produccion (modelo full-data)
    ranking["score_oof"] = proba_oof             # honesto (transparencia)
    ranking["rank_lookalike_v3"] = (
        ranking["score_lookalike_v3"].rank(ascending=False, method="first").astype(int)
    )
    ranking = ranking.sort_values("rank_lookalike_v3").reset_index(drop=True)
    precision_v3 = float(
        df.assign(p=proba_oof).sort_values("p", ascending=False)
        .head(config.TOP_K)[config.LABEL_COL].mean()
    )  # Precision@K honesta (sobre OOF)

    ranking.to_parquet(config.LOOKALIKE_V3_RANKING_PARQUET_PATH, index=False)
    ranking.to_csv(config.LOOKALIKE_V3_RANKING_CSV_PATH, index=False)
    joblib.dump(final_model, config.LOOKALIKE_V3_MODEL_PATH)
    logger.info("Ranking -> %s (+ .csv) | modelo -> %s",
                config.LOOKALIKE_V3_RANKING_PARQUET_PATH.name,
                config.LOOKALIKE_V3_MODEL_PATH.name)

    write_summary(df, predictors, oof, v2, fold_info, coefs, precision_v3)
    logger.info("v3 (spatial CV) completo.")


if __name__ == "__main__":
    main()
