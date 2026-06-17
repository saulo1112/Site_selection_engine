"""MODELO v2 — Clasificador look-alike (Regresion Logistica, con ML).

PROBLEMA DE CLASIFICACION (clases a predecir)
---------------------------------------------
La etiqueta `tiene_d1` (calculada en la ETAPA 4) es binaria:
  - Clase 1 (positiva): el hexagono YA tiene >=1 tienda D1 a <=300m -> "celda tipo-D1",
    un sitio que D1 ya eligio.
  - Clase 0 (negativa): el hexagono no tiene D1 cercano.
El clasificador aprende, a partir de las features NO-D1 (competidores, complementarios,
red vial y demografia si esta disponible), a estimar `P(clase=1)`. Esa probabilidad es el
*score look-alike*: "que tan parecida es esta celda a las que D1 escogio". Con ese score
se rankean todos los hexagonos.

Limitacion honesta (positive-unlabeled): las negativas mezclan sitios genuinamente malos
con sitios buenos donde D1 aun no llega. Por eso el score es de *similitud / prioridad de
exploracion*, NO una prediccion de desempeno (ver docs/metodologia.md §5).

Anti-leakage: las features derivadas de la ubicacion de D1 (`config.MCDA_LEAKAGE_COLS`)
se EXCLUYEN de los predictores; usarlas seria leakage tautologico (la etiqueta es funcion
de `n_d1_300m`).

ADVERTENCIA METODOLOGICA (v2 es naive): el split train/test es ALEATORIO, por lo que
hexagonos vecinos (espacialmente autocorrelacionados) caen a ambos lados -> las metricas
estaran probablemente infladas por leakage espacial. v3 lo corrige con spatial CV y
compara. Esto es intencional y se documenta.

Ejecutar de forma independiente (requiere data/processed/features.parquet de la ETAPA 4):
    uv run python -m src.models.lookalike
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.logging_config import get_logger
from src.models.metrics import ranking_report

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Carga y seleccion de predictores
# --------------------------------------------------------------------------- #
def load_features() -> pd.DataFrame:
    """Lee la tabla de features de la ETAPA 4."""
    path = config.FEATURES_PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre el pipeline de datos primero "
            "(uv run python -m src.data.features)."
        )
    df = pd.read_parquet(path)
    logger.info("Features cargadas: %d hexagonos, %d columnas", len(df), df.shape[1])
    return df


def select_predictors(df: pd.DataFrame) -> list[str]:
    """Predictores no-leakage, presentes y con datos (descarta columnas 100% nulas)."""
    predictors: list[str] = []
    for col in config.MODEL_PREDICTOR_COLS:
        if col in config.MCDA_LEAKAGE_COLS:  # defensa explicita anti-leakage
            logger.warning("Excluida por LEAKAGE de los predictores: %s", col)
            continue
        if col not in df.columns:
            logger.warning("Predictor ausente, se omite: %s", col)
            continue
        if not df[col].notna().any():
            logger.warning("Predictor 100%% nulo, se omite: %s", col)
            continue
        predictors.append(col)
    if not predictors:
        raise RuntimeError("No quedaron predictores utilizables para el clasificador.")
    excluded = sorted(set(config.MCDA_LEAKAGE_COLS))
    logger.info("Predictores usados (%d): %s", len(predictors), ", ".join(predictors))
    logger.info("Excluidos por leakage de D1: %s", ", ".join(excluded))
    return predictors


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def build_model() -> Pipeline:
    """Regresion Logistica con imputacion + estandarizacion (LR es sensible a la escala).

    - `SimpleImputer(median)`: las features demograficas (censo/estrato, v4) tienen NULL
      PARCIAL (manzanas no cubren todo el grid). La mediana se ajusta DENTRO de cada fold
      (parte del Pipeline) -> sin fuga de informacion entre train y test. Para v2/v3,
      cuyas features no tienen NaN, el imputer es un no-op inocuo.
    - `class_weight='balanced'`: compensa el desbalance (~24.5% positivos) penalizando
      mas los errores en la clase minoritaria.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        )),
    ])


# --------------------------------------------------------------------------- #
# Evaluacion
# --------------------------------------------------------------------------- #
def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    proba_all: np.ndarray,
    labels_all: np.ndarray,
) -> dict:
    """Diagnostico de clasificacion (en test) + metricas de ranking (sobre todo el grid)."""
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    report = classification_report(y_test, pred_test, digits=4,
                                   target_names=["clase_0 (sin D1)", "clase_1 (tipo-D1)"],
                                   zero_division=0)
    cm = confusion_matrix(y_test, pred_test)            # filas=real, cols=pred
    roc_auc = roc_auc_score(y_test, proba_test)
    pr_auc = average_precision_score(y_test, proba_test)  # honesto con desbalance

    ranking = ranking_report(proba_all, labels_all, k=config.TOP_K)

    logger.info("Clasificacion (test) -> ROC-AUC=%.4f | PR-AUC=%.4f", roc_auc, pr_auc)
    logger.info("Ranking (grid completo) @K=%d -> NDCG=%.4f | hitting=%.4f | loss=%.4f",
                int(ranking["k"]), ranking["ndcg_at_k"], ranking["topk_hitting"],
                ranking["topk_loss"])
    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "ranking": ranking,
        "n_test": int(len(y_test)),
        "pos_rate_test": float(y_test.mean()),
    }


def coefficients_table(model: Pipeline, predictors: list[str]) -> pd.DataFrame:
    """Coeficientes de la LR por feature (sobre features estandarizadas -> comparables).

    Signo = direccion del efecto sobre P(tipo-D1); magnitud = importancia relativa.
    """
    coefs = model.named_steps["clf"].coef_[0]
    tbl = pd.DataFrame({"feature": predictors, "coef": coefs})
    tbl["abs_coef"] = tbl["coef"].abs()
    return tbl.sort_values("abs_coef", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Resumen
# --------------------------------------------------------------------------- #
def _read_v1_metrics() -> dict[str, float]:
    """Lee NDCG/Precision@K de v1 (MCDA) desde su ranking, para comparar v1 vs v2."""
    path = config.MCDA_RANKING_PARQUET_PATH
    if not path.exists():
        return {}
    r = pd.read_parquet(path).sort_values("rank_mcda")
    rep = ranking_report(r["score_mcda"].to_numpy(), r["tiene_d1"].to_numpy(),
                         k=config.TOP_K)
    topk = r.head(config.TOP_K)
    rep["precision_at_k"] = float(topk["tiene_d1"].mean())
    return rep


def write_summary(
    df: pd.DataFrame,
    predictors: list[str],
    results: dict,
    coefs: pd.DataFrame,
    precision_v2: float,
) -> None:
    k = config.TOP_K
    cm = results["confusion_matrix"]
    rk = results["ranking"]
    v1 = _read_v1_metrics()

    excluded_demo = [c for c in ("poblacion_estimada", "viviendas_estimadas")
                     if c not in predictors]

    comparison_rows = [
        f"| NDCG@{k} | {v1.get('ndcg_at_k', float('nan')):.4f} | {rk['ndcg_at_k']:.4f} |",
        f"| Precision@{k} | {v1.get('precision_at_k', float('nan')):.4f} | {precision_v2:.4f} |",
        f"| top-{k} hitting | {v1.get('topk_hitting', float('nan')):.4f} | {rk['topk_hitting']:.4f} |",
    ]

    lines = [
        "# Resultados v2 — Clasificador look-alike (Regresion Logistica)\n",
        f"_Generado por `src/models/lookalike.py`. Total de hexagonos: **{len(df)}**._\n",
        "## Clases a predecir\n",
        "Clasificacion **binaria** sobre la etiqueta `tiene_d1` (calculada en la ETAPA 4):\n",
        "- **Clase 1 (positiva, ~24.5%):** el hexagono YA tiene >=1 tienda D1 a <=300m "
        "(\"celda tipo-D1\", un sitio que D1 ya eligio).",
        "- **Clase 0 (negativa):** el hexagono no tiene D1 cercano.\n",
        "El modelo estima `P(clase=1)` a partir de las features **no-D1** y ese "
        "probabilistico es el **score look-alike** con el que se rankean los hexagonos.\n",
        "## Predictores (anti-leakage)\n",
        f"Se usan **{len(predictors)}** features: `{'`, `'.join(predictors)}`.\n",
        "Se **excluyen** las derivadas de D1 "
        f"(`{'`, `'.join(config.MCDA_LEAKAGE_COLS)}`): la etiqueta es funcion directa de "
        "ellas, usarlas seria leakage tautologico.",
    ]
    if excluded_demo:
        lines.append(
            f"Las demograficas (`{'`, `'.join(excluded_demo)}`) no estan disponibles "
            "(censo DANE no cargado, ver `src/data/load_censo.py`) y quedan fuera.\n"
        )
    else:
        lines.append("")

    lines += [
        "## Particion (v2 = split aleatorio, naive)\n",
        f"Split **aleatorio estratificado** {int((1 - config.TEST_SIZE) * 100)}/"
        f"{int(config.TEST_SIZE * 100)} (`random_state={config.RANDOM_STATE}`). "
        "**Advertencia:** el split aleatorio reparte hexagonos vecinos "
        "(espacialmente autocorrelacionados) entre train y test, por lo que las metricas "
        "probablemente esten **infladas por leakage espacial**. v3 lo corrige con "
        "spatial CV y compara (ver docs/metodologia.md §6).\n",
        "## Diagnostico de clasificacion (conjunto de test)\n",
        f"- **ROC-AUC**: {results['roc_auc']:.4f}",
        f"- **PR-AUC** (average precision, mas honesta con clases desbalanceadas): "
        f"{results['pr_auc']:.4f}",
        f"- Test: {results['n_test']} hexagonos, {results['pos_rate_test'] * 100:.1f}% positivos.\n",
        "**Matriz de confusion** (umbral 0.5; filas = real, columnas = predicho):\n",
        "| | pred 0 | pred 1 |",
        "|---|---|---|",
        f"| **real 0** | {cm[0, 0]} | {cm[0, 1]} |",
        f"| **real 1** | {cm[1, 0]} | {cm[1, 1]} |",
        "\n**Reporte por clase** (precision / recall / F1):\n",
        "```",
        results["classification_report"].rstrip(),
        "```\n",
        "## Interpretabilidad — coeficientes de la LR\n",
        "Sobre features estandarizadas (comparables entre si). Signo = direccion del "
        "efecto sobre `P(tipo-D1)`; magnitud = importancia relativa.\n",
        coefs.round(4).to_markdown(index=False),
        "\n## Metricas de ranking (sobre todo el grid)\n",
        f"- **NDCG@{k}**: {rk['ndcg_at_k']:.4f}",
        f"- **Precision@{k}**: {precision_v2:.4f}",
        f"- **top-{k} hitting**: {rk['topk_hitting']:.4f} / **loss**: {rk['topk_loss']:.4f}\n",
        "## Comparacion v1 (MCDA) vs v2 (LR)\n",
        "| Metrica | v1 MCDA | v2 LR |",
        "|---|---|---|",
        *comparison_rows,
        "\n> **Lectura honesta:** si v2 no supera materialmente a v1, es un resultado "
        "valido: el MCDA ya captura casi toda la senal lineal disponible. Y recordar que "
        "cualquier ventaja de v2 aqui puede ser, en parte, leakage espacial -> v3 dira "
        "cuanto se sostiene.\n",
        "## Limitacion\n",
        "Problema **positive-unlabeled**: las negativas incluyen buenos sitios donde D1 "
        "aun no llega. El score es de **similitud / prioridad de exploracion**, no de "
        "desempeno; hereda el supuesto de que la estrategia de localizacion de D1 es buena "
        "(docs/metodologia.md §5).\n",
    ]
    config.LOOKALIKE_V2_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Resumen escrito -> %s", config.LOOKALIKE_V2_SUMMARY_PATH.name)


# --------------------------------------------------------------------------- #
# Orquestacion
# --------------------------------------------------------------------------- #
def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_features()
    predictors = select_predictors(df)

    X = df[predictors]
    y = df[config.LABEL_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y,
    )
    logger.info("Split aleatorio estratificado: train=%d, test=%d", len(X_train), len(X_test))

    model = build_model()
    model.fit(X_train, y_train)
    logger.info("Modelo entrenado: %s", model.named_steps["clf"].__class__.__name__)

    # Score look-alike para TODO el grid (P(clase=1)).
    proba_all = model.predict_proba(X)[:, 1]
    labels_all = y.to_numpy()

    results = evaluate(model, X_test, y_test, proba_all, labels_all)
    coefs = coefficients_table(model, predictors)

    # Ranking de salida.
    ranking = df[["h3_index", "lat_centroid", "lon_centroid", config.LABEL_COL]].copy()
    ranking["score_lookalike"] = proba_all
    ranking["rank_lookalike"] = (
        ranking["score_lookalike"].rank(ascending=False, method="first").astype(int)
    )
    ranking = ranking.sort_values("rank_lookalike").reset_index(drop=True)
    precision_v2 = float(ranking.head(config.TOP_K)[config.LABEL_COL].mean())

    ranking.to_parquet(config.LOOKALIKE_V2_RANKING_PARQUET_PATH, index=False)
    ranking.to_csv(config.LOOKALIKE_V2_RANKING_CSV_PATH, index=False)
    joblib.dump(model, config.LOOKALIKE_V2_MODEL_PATH)
    logger.info("Ranking -> %s (+ .csv) | modelo -> %s",
                config.LOOKALIKE_V2_RANKING_PARQUET_PATH.name,
                config.LOOKALIKE_V2_MODEL_PATH.name)

    write_summary(df, predictors, results, coefs, precision_v2)
    logger.info("v2 (look-alike LR) completo.")


if __name__ == "__main__":
    main()
