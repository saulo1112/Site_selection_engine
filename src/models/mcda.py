"""MODELO v1 — MCDA baseline (Multi-Criteria Decision Analysis, sin ML).

Calcula un score ponderado e interpretable por hexagono a partir de la tabla de features,
normalizando cada variable (min-max) y combinandola con pesos a priori definidos por
razonamiento de negocio (NO ajustados a la etiqueta). El resultado es un ranking de
prioridad de exploracion, NO una prediccion de desempeno (ver docs/metodologia.md §5).

Anti-leakage (critico): las features derivadas de la ubicacion de D1
(`config.MCDA_LEAKAGE_COLS`) se EXCLUYEN del score, porque la etiqueta
`tiene_d1 = (n_d1_300m >= 1)` es funcion directa de ellas. La etiqueta solo se usa
DESPUES, como validacion honesta post-hoc (NDCG@K, top-K hitting/loss), nunca como insumo.

Ejecutar de forma independiente (requiere data/processed/features.parquet de la ETAPA 4):
    uv run python -m src.models.mcda
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.logging_config import get_logger
from src.models.metrics import ranking_report

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Carga
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


# --------------------------------------------------------------------------- #
# Normalizacion y score
# --------------------------------------------------------------------------- #
def _minmax(series: pd.Series, direction: int) -> pd.Series:
    """Normaliza a [0, 1]. direction=-1 invierte (mejor-menos -> alto tras normalizar)."""
    lo, hi = series.min(), series.max()
    if hi == lo:  # columna constante: aporta 0 de discriminacion
        return pd.Series(0.0, index=series.index)
    norm = (series - lo) / (hi - lo)
    return (1.0 - norm) if direction < 0 else norm


def _usable_features(df: pd.DataFrame) -> dict[str, list[tuple[str, int]]]:
    """Por grupo, filtra las features presentes, no-leakage y con datos (no todo-NaN).

    Las features de `config.MCDA_LEAKAGE_COLS` se excluyen explicitamente.
    Las columnas 100% nulas (p.ej. demografia sin censo) se descartan y se loguea.
    """
    usable: dict[str, list[tuple[str, int]]] = {}
    for group, feats in config.MCDA_GROUP_FEATURES.items():
        kept: list[tuple[str, int]] = []
        for col, direction in feats:
            if col in config.MCDA_LEAKAGE_COLS:
                logger.warning("Excluida por LEAKAGE del score MCDA: %s", col)
                continue
            if col not in df.columns:
                logger.warning("Feature ausente, se omite: %s", col)
                continue
            if not df[col].notna().any():
                logger.warning("Feature 100%% nula, se omite del grupo '%s': %s", group, col)
                continue
            kept.append((col, direction))
        if kept:
            usable[group] = kept
        else:
            logger.warning("Grupo '%s' sin features utilizables -> se omite.", group)
    return usable


def _effective_group_weights(usable: dict[str, list[tuple[str, int]]]) -> dict[str, float]:
    """Renormaliza los pesos de grupo a 1.0 sobre los grupos efectivamente presentes."""
    present = {g: config.MCDA_GROUP_WEIGHTS[g] for g in usable}
    total = sum(present.values())
    if total == 0:
        raise RuntimeError("Ningun grupo de features quedo disponible para el MCDA.")
    eff = {g: w / total for g, w in present.items()}
    for g, w in eff.items():
        if abs(w - config.MCDA_GROUP_WEIGHTS[g]) > 1e-9:
            logger.info("Peso del grupo '%s' renormalizado: %.3f -> %.3f",
                        g, config.MCDA_GROUP_WEIGHTS[g], w)
    return eff


def compute_mcda_score(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Anade `score_mcda` y `rank_mcda`. Devuelve (df, metadata de pesos efectivos)."""
    usable = _usable_features(df)
    group_w = _effective_group_weights(usable)

    out = df.copy()
    score = pd.Series(0.0, index=out.index)
    feature_weights: dict[str, float] = {}

    for group, feats in usable.items():
        per_feature_w = group_w[group] / len(feats)  # peso uniforme dentro del grupo
        for col, direction in feats:
            contribution = _minmax(out[col].astype(float), direction) * per_feature_w
            score = score + contribution
            feature_weights[col] = per_feature_w

    out["score_mcda"] = score
    # rank 1 = mejor (score mas alto). 'first' para desempate determinista.
    out["rank_mcda"] = out["score_mcda"].rank(ascending=False, method="first").astype(int)
    out = out.sort_values("rank_mcda").reset_index(drop=True)

    meta = {
        "usable": usable,
        "group_weights": group_w,
        "feature_weights": feature_weights,
        "excluded_leakage": list(config.MCDA_LEAKAGE_COLS),
    }
    logger.info("Score MCDA calculado con %d features en %d grupos.",
                len(feature_weights), len(group_w))
    return out, meta


# --------------------------------------------------------------------------- #
# Evaluacion honesta (post-hoc, NO usada para ajustar pesos)
# --------------------------------------------------------------------------- #
def evaluate_against_label(df: pd.DataFrame) -> dict[str, float]:
    """Valida el ranking MCDA contra `tiene_d1` con metricas de ranking (top-K)."""
    if "tiene_d1" not in df.columns:
        logger.warning("Sin columna 'tiene_d1' -> se omite la evaluacion post-hoc.")
        return {}
    report = ranking_report(df["score_mcda"].to_numpy(), df["tiene_d1"].to_numpy(),
                            k=config.TOP_K)
    logger.info("Eval post-hoc @K=%d -> NDCG=%.4f | hitting=%.4f | loss=%.4f",
                int(report["k"]), report["ndcg_at_k"], report["topk_hitting"],
                report["topk_loss"])
    return report


# --------------------------------------------------------------------------- #
# Resumen
# --------------------------------------------------------------------------- #
def write_summary(df: pd.DataFrame, meta: dict, metrics: dict[str, float]) -> None:
    n_total = len(df)
    k = config.TOP_K
    topk = df.head(k)
    n_pos_total = int(df["tiene_d1"].sum()) if "tiene_d1" in df.columns else 0
    n_pos_topk = int(topk["tiene_d1"].sum()) if "tiene_d1" in df.columns else 0
    precision_topk = (n_pos_topk / k) if k else float("nan")
    # Techo del hitting: con mas positivos que K, ni un ranking perfecto puede
    # superar K/positivos. Reportarlo evita malinterpretar un hitting "bajo".
    hitting_ceiling = (min(k, n_pos_total) / n_pos_total) if n_pos_total else float("nan")

    # Tabla de pesos por feature.
    weight_rows = [
        f"| `{col}` | {grp} | {meta['feature_weights'][col]:.4f} |"
        for grp, feats in meta["usable"].items()
        for col, _ in feats
    ]

    excluded_demo = [
        col for grp in ("demografia",)
        for col, _ in config.MCDA_GROUP_FEATURES.get(grp, [])
        if grp not in meta["usable"]
    ]

    lines = [
        "# Resultados v1 — MCDA baseline\n",
        f"_Generado por `src/models/mcda.py`. Total de hexagonos: **{n_total}**._\n",
        "## Que es esta version\n",
        "Score ponderado e **interpretable**, sin ML: normalizacion min-max por feature "
        "y combinacion lineal con pesos a priori definidos por razonamiento de negocio "
        "(no ajustados a la etiqueta). Es la **linea base** contra la cual se mediran "
        "v2 (clasificador look-alike) y v3 (spatial CV).\n",
        "## Anti-leakage\n",
        "Las features derivadas de la ubicacion de D1 "
        f"(`{'`, `'.join(meta['excluded_leakage'])}`) se **excluyen del score**: la "
        "etiqueta `tiene_d1` se define como `n_d1_300m >= 1`, por lo que usarlas seria "
        "leakage tautologico. La etiqueta se usa **solo despues**, como validacion "
        "honesta (metricas de ranking abajo), nunca como insumo del score.\n",
    ]
    if excluded_demo:
        lines.append(
            "> **Nota:** las features demograficas "
            f"(`{'`, `'.join(excluded_demo)}`) no estan disponibles en esta corrida "
            "(censo DANE no cargado, ver `src/data/load_censo.py`). Su peso de grupo se "
            "redistribuyo proporcionalmente entre los grupos presentes.\n"
        )

    lines += [
        "## Pesos efectivos por grupo\n",
        "| Grupo | Peso efectivo |",
        "|---|---|",
        *[f"| {g} | {w:.4f} |" for g, w in meta["group_weights"].items()],
        "\n## Pesos por feature\n",
        "| Feature | Grupo | Peso |",
        "|---|---|---|",
        *weight_rows,
        "\n## Evaluacion honesta post-hoc (ranking vs. `tiene_d1`)\n",
        "Estas metricas **no** se usaron para elegir pesos; solo miden, a posteriori, "
        "que tan bien el ranking MCDA recupera las celdas donde D1 ya esta presente.\n",
        f"- **NDCG@{k}**: {metrics.get('ndcg_at_k', float('nan')):.4f} "
        "(calidad del orden en el top-K; 1.0 = perfecto).",
        f"- **Precision@{k}**: {precision_topk:.4f} "
        f"({n_pos_topk} de las {k} celdas mejor rankeadas ya tienen D1).",
        f"- **top-{k} hitting** (recall de positivos en el top-K): "
        f"{metrics.get('topk_hitting', float('nan')):.4f} "
        f"(techo = {hitting_ceiling:.4f}, porque hay {n_pos_total} positivos > K={k}: "
        "ni un ranking perfecto puede capturarlos todos en solo K celdas).",
        f"- **top-{k} loss** (positivos fuera del top-K): "
        f"{metrics.get('topk_loss', float('nan')):.4f}",
        f"- Positivos en el top-{k}: **{n_pos_topk}** de {n_pos_total} positivos totales "
        f"({n_total} hexagonos).\n",
        "> **Lectura honesta:** el `hitting` parece bajo solo porque hay muchos mas "
        f"positivos ({n_pos_total}) que celdas seleccionadas (K={k}). La metrica mas "
        f"informativa aqui es **Precision@{k} = {precision_topk:.3f}** y el "
        f"**NDCG@{k} = {metrics.get('ndcg_at_k', float('nan')):.3f}**: un baseline sin ML "
        "que ordena bien las celdas mas parecidas a las de D1. v2/v3 deberan superarlo "
        "(o, en v3, revelar cuanto de esto era leakage espacial).\n",
        "## Limitacion\n",
        "El score es de **similitud / prioridad de exploracion**, no de desempeno: mide "
        "parecido a las celdas donde D1 ya opera, no ventas esperadas. Hereda el supuesto "
        "look-alike de que la estrategia de localizacion de D1 es buena "
        "(ver docs/metodologia.md §5).\n",
    ]
    config.MCDA_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Resumen escrito -> %s", config.MCDA_SUMMARY_PATH.name)


# --------------------------------------------------------------------------- #
# Orquestacion
# --------------------------------------------------------------------------- #
def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_features()
    df, meta = compute_mcda_score(df)
    metrics = evaluate_against_label(df)

    cols = ["h3_index", "lat_centroid", "lon_centroid", "score_mcda", "rank_mcda"]
    if "tiene_d1" in df.columns:
        cols.append("tiene_d1")
    ranking = df[cols]
    ranking.to_parquet(config.MCDA_RANKING_PARQUET_PATH, index=False)
    ranking.to_csv(config.MCDA_RANKING_CSV_PATH, index=False)
    logger.info("Ranking guardado -> %s (+ .csv)", config.MCDA_RANKING_PARQUET_PATH.name)

    write_summary(df, meta, metrics)
    logger.info("v1 (MCDA) completo.")


if __name__ == "__main__":
    main()
