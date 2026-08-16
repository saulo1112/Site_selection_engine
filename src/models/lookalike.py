"""MODEL v2 — Look-alike classifier (Logistic Regression, with ML).

CLASSIFICATION PROBLEM (classes to predict)
--------------------------------------------
The `tiene_d1` label (computed in STAGE 4) is binary:
  - Class 1 (positive): the hexagon ALREADY has >=1 D1 store within <=300m -> "D1-type
    cell", a site D1 already chose.
  - Class 0 (negative): the hexagon has no nearby D1.
The classifier learns, from the non-D1 features (competitors, complementary businesses,
road network and demographics if available), to estimate `P(class=1)`. That probability is
the *look-alike score*: "how similar is this cell to the ones D1 chose". That score is used
to rank all hexagons.

Honest limitation (positive-unlabeled): the negatives mix genuinely bad sites with good
sites D1 simply hasn't reached yet. That's why the score reflects *similarity /
exploration priority*, NOT a performance prediction (see docs/metodologia.md §5).

Anti-leakage: features derived from the D1 location (`config.MCDA_LEAKAGE_COLS`) are
EXCLUDED from the predictors; using them would be tautological leakage (the label is a
function of `n_d1_300m`).

METHODOLOGICAL WARNING (v2 is naive): the train/test split is RANDOM, so neighboring
hexagons (spatially autocorrelated) end up on both sides -> the metrics are likely
inflated by spatial leakage. v3 fixes this with spatial CV and compares. This is
intentional and documented.

Run standalone (requires data/processed/features.parquet from STAGE 4):
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
# Loading and predictor selection
# --------------------------------------------------------------------------- #
def load_features() -> pd.DataFrame:
    """Reads the features table from STAGE 4."""
    path = config.FEATURES_PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run the data pipeline first "
            "(uv run python -m src.data.features)."
        )
    df = pd.read_parquet(path)
    logger.info("Features loaded: %d hexagons, %d columns", len(df), df.shape[1])
    return df


def select_predictors(df: pd.DataFrame) -> list[str]:
    """Non-leakage predictors, present and with data (drops 100% null columns)."""
    predictors: list[str] = []
    for col in config.MODEL_PREDICTOR_COLS:
        if col in config.MCDA_LEAKAGE_COLS:  # explicit anti-leakage guard
            logger.warning("Excluded from predictors due to LEAKAGE: %s", col)
            continue
        if col not in df.columns:
            logger.warning("Predictor missing, skipped: %s", col)
            continue
        if not df[col].notna().any():
            logger.warning("Predictor 100%% null, skipped: %s", col)
            continue
        predictors.append(col)
    if not predictors:
        raise RuntimeError("No usable predictors remained for the classifier.")
    excluded = sorted(set(config.MCDA_LEAKAGE_COLS))
    logger.info("Predictors used (%d): %s", len(predictors), ", ".join(predictors))
    logger.info("Excluded due to D1 leakage: %s", ", ".join(excluded))
    return predictors


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model() -> Pipeline:
    """Logistic Regression with imputation + standardization (LR is scale-sensitive).

    - `SimpleImputer(median)`: the demographic features (census/stratum, v4) have
      PARTIAL NULLs (city blocks don't cover the whole grid). The median is fit INSIDE
      each fold (part of the Pipeline) -> no information leakage between train and test.
      For v2/v3, whose features have no NaN, the imputer is a harmless no-op.
    - `class_weight='balanced'`: compensates for the class imbalance (~24.5% positives)
      by penalizing errors on the minority class more heavily.
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
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    proba_all: np.ndarray,
    labels_all: np.ndarray,
) -> dict:
    """Classification diagnostics (on test) + ranking metrics (over the whole grid)."""
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    report = classification_report(y_test, pred_test, digits=4,
                                   target_names=["class_0 (no D1)", "class_1 (D1-type)"],
                                   zero_division=0)
    cm = confusion_matrix(y_test, pred_test)            # rows=actual, cols=predicted
    roc_auc = roc_auc_score(y_test, proba_test)
    pr_auc = average_precision_score(y_test, proba_test)  # honest under imbalance

    ranking = ranking_report(proba_all, labels_all, k=config.TOP_K)

    logger.info("Classification (test) -> ROC-AUC=%.4f | PR-AUC=%.4f", roc_auc, pr_auc)
    logger.info("Ranking (full grid) @K=%d -> NDCG=%.4f | hitting=%.4f | loss=%.4f",
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
    """LR coefficients per feature (over standardized features -> comparable).

    Sign = direction of the effect on P(D1-type); magnitude = relative importance.
    """
    coefs = model.named_steps["clf"].coef_[0]
    tbl = pd.DataFrame({"feature": predictors, "coef": coefs})
    tbl["abs_coef"] = tbl["coef"].abs()
    return tbl.sort_values("abs_coef", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _read_v1_metrics() -> dict[str, float]:
    """Reads v1 (MCDA) NDCG/Precision@K from its ranking, to compare v1 vs v2."""
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
        "# Results v2 — Look-alike classifier (Logistic Regression)\n",
        f"_Generated by `src/models/lookalike.py`. Total hexagons: **{len(df)}**._\n",
        "## Classes to predict\n",
        "**Binary** classification over the `tiene_d1` label (computed in STAGE 4):\n",
        "- **Class 1 (positive, ~24.5%):** the hexagon ALREADY has >=1 D1 store within "
        "<=300m (\"D1-type cell\", a site D1 already chose).",
        "- **Class 0 (negative):** the hexagon has no nearby D1.\n",
        "The model estimates `P(class=1)` from the **non-D1** features, and that "
        "probability is the **look-alike score** used to rank the hexagons.\n",
        "## Predictors (anti-leakage)\n",
        f"**{len(predictors)}** features are used: `{'`, `'.join(predictors)}`.\n",
        "The features derived from D1 are **excluded** "
        f"(`{'`, `'.join(config.MCDA_LEAKAGE_COLS)}`): the label is a direct function of "
        "them, using them would be tautological leakage.",
    ]
    if excluded_demo:
        lines.append(
            f"The demographic features (`{'`, `'.join(excluded_demo)}`) are not available "
            "(DANE census not loaded, see `src/data/load_censo.py`) and are left out.\n"
        )
    else:
        lines.append("")

    lines += [
        "## Split (v2 = random split, naive)\n",
        f"**Stratified random** split {int((1 - config.TEST_SIZE) * 100)}/"
        f"{int(config.TEST_SIZE * 100)} (`random_state={config.RANDOM_STATE}`). "
        "**Warning:** the random split scatters neighboring hexagons "
        "(spatially autocorrelated) between train and test, so the metrics are "
        "likely **inflated by spatial leakage**. v3 fixes this with "
        "spatial CV and compares (see docs/metodologia.md §6).\n",
        "## Classification diagnostics (test set)\n",
        f"- **ROC-AUC**: {results['roc_auc']:.4f}",
        f"- **PR-AUC** (average precision, more honest under class imbalance): "
        f"{results['pr_auc']:.4f}",
        f"- Test: {results['n_test']} hexagons, {results['pos_rate_test'] * 100:.1f}% positives.\n",
        "**Confusion matrix** (threshold 0.5; rows = actual, columns = predicted):\n",
        "| | pred 0 | pred 1 |",
        "|---|---|---|",
        f"| **actual 0** | {cm[0, 0]} | {cm[0, 1]} |",
        f"| **actual 1** | {cm[1, 0]} | {cm[1, 1]} |",
        "\n**Per-class report** (precision / recall / F1):\n",
        "```",
        results["classification_report"].rstrip(),
        "```\n",
        "## Interpretability — LR coefficients\n",
        "Over standardized features (comparable to each other). Sign = direction of the "
        "effect on `P(D1-type)`; magnitude = relative importance.\n",
        coefs.round(4).to_markdown(index=False),
        "\n## Ranking metrics (over the whole grid)\n",
        f"- **NDCG@{k}**: {rk['ndcg_at_k']:.4f}",
        f"- **Precision@{k}**: {precision_v2:.4f}",
        f"- **top-{k} hitting**: {rk['topk_hitting']:.4f} / **loss**: {rk['topk_loss']:.4f}\n",
        "## Comparison v1 (MCDA) vs v2 (LR)\n",
        "| Metric | v1 MCDA | v2 LR |",
        "|---|---|---|",
        *comparison_rows,
        "\n> **Honest reading:** if v2 doesn't materially beat v1, that's a valid "
        "result: the MCDA already captures almost all of the available linear signal. "
        "And keep in mind that any advantage v2 shows here may partly be spatial "
        "leakage -> v3 will tell how much of it holds up.\n",
        "## Limitation\n",
        "**Positive-unlabeled** problem: the negatives include good sites D1 simply "
        "hasn't reached yet. The score reflects **similarity / exploration priority**, not "
        "performance; it inherits the assumption that D1's location strategy is good "
        "(docs/metodologia.md §5).\n",
    ]
    config.LOOKALIKE_V2_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary written -> %s", config.LOOKALIKE_V2_SUMMARY_PATH.name)


# --------------------------------------------------------------------------- #
# Orchestration
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
    logger.info("Stratified random split: train=%d, test=%d", len(X_train), len(X_test))

    model = build_model()
    model.fit(X_train, y_train)
    logger.info("Model trained: %s", model.named_steps["clf"].__class__.__name__)

    # Look-alike score for the WHOLE grid (P(class=1)).
    proba_all = model.predict_proba(X)[:, 1]
    labels_all = y.to_numpy()

    results = evaluate(model, X_test, y_test, proba_all, labels_all)
    coefs = coefficients_table(model, predictors)

    # Output ranking.
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
    logger.info("Ranking -> %s (+ .csv) | model -> %s",
                config.LOOKALIKE_V2_RANKING_PARQUET_PATH.name,
                config.LOOKALIKE_V2_MODEL_PATH.name)

    write_summary(df, predictors, results, coefs, precision_v2)
    logger.info("v2 (look-alike LR) complete.")


if __name__ == "__main__":
    main()
