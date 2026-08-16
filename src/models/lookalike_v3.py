"""MODEL v3 — Look-alike classifier with SPATIAL CROSS-VALIDATION.

Same model as v2 (Logistic Regression), but evaluated with SPATIAL cross-validation
instead of a random split. Goal: an HONEST estimate of generalization, and to quantify
how much of v2's performance was leakage from spatial autocorrelation (neighboring
cells scattered between train and test).

How it works (see src/models/spatial_cv.py):
  - Spatial blocks = H3 parent at a coarse resolution; whole blocks go to train or test.
  - StratifiedGroupKFold distributes blocks into folds (balancing positives).
  - Buffer: cells within <=k rings of any test cell are excluded from train.
  - OUT-OF-FOLD (OOF) predictions are assembled: each cell is predicted by a model that
    did NOT see its neighborhood -> the OOF metrics are v3's honest evaluation.

Anti-leakage (three layers): (1) predictors with no D1 columns and competition
features are non-D1 only; (2) StandardScaler fit inside each fold (Pipeline) -> no
scaling leakage; (3) spatial blocks + buffer -> train and test don't share a
neighborhood.

Run (requires data/processed/features.parquet from STAGE 4):
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
# Generic evaluation of a score vector against the label
# --------------------------------------------------------------------------- #
def evaluate_scores(y_true: npt.ArrayLike, proba: npt.ArrayLike, k: int) -> dict:
    """Classification diagnostics (threshold 0.5) + ranking metrics over a score."""
    y = np.asarray(y_true, dtype=np.int_)
    p = np.asarray(proba, dtype=np.float64)
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "confusion_matrix": confusion_matrix(y, pred),
        "classification_report": classification_report(
            y, pred, digits=4,
            target_names=["class_0 (no D1)", "class_1 (D1-type)"], zero_division=0,
        ),
        "ranking": ranking_report(p, y, k=k),
    }


# --------------------------------------------------------------------------- #
# Out-of-fold predictions with spatial CV
# --------------------------------------------------------------------------- #
def spatial_cv_oof(
    df: pd.DataFrame, predictors: list[str],
) -> tuple[np.ndarray, list[dict]]:
    """Assembles OOF P(class=1) predictions with spatial folds + buffer.

    Returns (proba_oof aligned with df, per-fold info). Verifies that each cell is
    predicted exactly once and that no test cell is in its own train set.
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
        # Anti-leakage: no test cell can be in its own train set.
        assert not set(test_idx) & set(train_idx), "train/test overlap in the fold"

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
        logger.info("Fold %d -> train=%d, test=%d (pos=%d), buffer removed %d cells",
                    i, info["n_train"], info["n_test"], info["test_positivos"], n_removed)

    if not covered.all():
        raise RuntimeError(f"{(~covered).sum()} cells have no OOF prediction (incomplete coverage).")
    return proba_oof, fold_info


# --------------------------------------------------------------------------- #
# v2 metrics (random split) for the honest comparison
# --------------------------------------------------------------------------- #
def v2_random_split_metrics(df: pd.DataFrame, predictors: list[str]) -> dict:
    """Reproduces v2's evaluation (random split) to compare against v3."""
    X = df[predictors]
    y = df[config.LABEL_COL].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y,
    )
    model = build_model()
    model.fit(X_tr, y_tr)
    proba_te = model.predict_proba(X_te)[:, 1]      # diagnostics on the random test set
    proba_all = model.predict_proba(X)[:, 1]        # ranking over the grid (like v2)
    res = evaluate_scores(y_te, proba_te, k=config.TOP_K)
    res["ranking"] = ranking_report(proba_all, y.to_numpy(), k=config.TOP_K)
    return res


# --------------------------------------------------------------------------- #
# Summary
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
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

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
        "# Results v3 — Look-alike classifier with Spatial CV\n",
        f"_Generated by `src/models/lookalike_v3.py`. Total hexagons: **{len(df)}**._\n",
        "## What changes from v2\n",
        "**Same model** (Logistic Regression), **same class definition** "
        "(class 1 = cell with D1 within <=300m). The only thing that changes is the "
        "**validation**: instead of a random split, **spatial** cross-validation. Each "
        "hexagon is predicted *out-of-fold* (OOF) by a model that never saw its "
        "neighborhood -> an honest estimate of how the model generalizes to new areas of "
        "the city.\n",
        "## Spatial CV scheme (anti spatial-leakage)\n",
        f"- **Spatial blocks**: H3 parent at resolution **{config.SPATIAL_CV_BLOCK_RES}** "
        f"-> {n_blocks} blocks (~36 km2 each). Whole blocks go to train or test.",
        f"- **Folds**: StratifiedGroupKFold, **{config.SPATIAL_CV_FOLDS}** folds "
        "(respects blocks, balances positives).",
        f"- **Buffer**: cells within <=**{config.SPATIAL_CV_BUFFER_RINGS}** H3 "
        "ring(s) of any test cell are excluded from train.",
        f"- **Predictors** (no leakage): `{'`, `'.join(predictors)}`.\n",
        "**Sizes per fold:**\n",
        "| Fold | Train | Test | Test pos. | Buffer removed |",
        "|---|---|---|---|---|",
        *fold_rows,
        "\n## Honest diagnostics (OOF predictions)\n",
        f"- **ROC-AUC**: {oof['roc_auc']:.4f}",
        f"- **PR-AUC**: {oof['pr_auc']:.4f}",
        f"- **NDCG@{k}**: {oof['ranking']['ndcg_at_k']:.4f} | "
        f"**Precision@{k}**: {precision_v3:.4f} | "
        f"**top-{k} hitting**: {oof['ranking']['topk_hitting']:.4f}\n",
        "**OOF confusion matrix** (threshold 0.5; rows = actual, columns = predicted):\n",
        "| | pred 0 | pred 1 |",
        "|---|---|---|",
        f"| **actual 0** | {cm[0, 0]} | {cm[0, 1]} |",
        f"| **actual 1** | {cm[1, 0]} | {cm[1, 1]} |",
        "\n**Per-class report (OOF):**\n",
        "```",
        oof["classification_report"].rstrip(),
        "```\n",
        "## Leakage verdict — v2 (random split) vs v3 (spatial CV)\n",
        "| Metric | v2 random | v3 spatial CV | Δ (v3 - v2) |",
        "|---|---|---|---|",
        *[f"| {nm} | {a:.4f} | {b:.4f} | {_delta(a, b)} |" for nm, a, b in comp_rows],
        "",
        _verdict_text(v2, oof),
        "\n## Interpretability — final model coefficients (full-data)\n",
        "The production ranking uses a model retrained on **all** the data; the "
        "performance reported above comes from the OOF predictions (not this full-data fit).\n",
        coefs.round(4).to_markdown(index=False),
        "\n## Limitation\n",
        "It's still a **similarity / exploration priority** score (not "
        "performance), with the look-alike assumption that D1's siting is good "
        "(docs/metodologia.md §5). Spatial CV fixes the spatial leakage, not the "
        "biases from the proxy or the OSM labeling.\n",
    ]
    config.LOOKALIKE_V3_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary written -> %s", config.LOOKALIKE_V3_SUMMARY_PATH.name)


def _verdict_text(v2: dict, oof: dict) -> str:
    # Convention consistent with the table: Delta = v3(OOF) - v2(random).
    d_ndcg = oof["ranking"]["ndcg_at_k"] - v2["ranking"]["ndcg_at_k"]
    d_auc = oof["roc_auc"] - v2["roc_auc"]
    if d_ndcg < -0.02 or d_auc < -0.02:
        return (
            "> **Verdict:** performance **drops** when moving to spatial CV "
            f"(Δ NDCG@K {d_ndcg:+.4f}, Δ ROC-AUC {d_auc:+.4f}). This **confirms** that "
            "part of v2's metrics were inflated by spatial-autocorrelation leakage; v3 is "
            "the honest generalization estimate."
        )
    return (
        "> **Verdict:** performance **holds up** under spatial CV "
        f"(Δ NDCG@K {d_ndcg:+.4f}, Δ ROC-AUC {d_auc:+.4f}; v3 even matches or slightly "
        "beats v2). Leakage from spatial autocorrelation turned out **smaller than "
        "expected**: with a linear model over buffer-based features (smooth spatial "
        "fields), a random split and a spatial split generalize similarly. This is a "
        "valid and honest finding — the non-D1 signal (competition/complementary/road "
        "network) holds up in unseen areas, it wasn't a mirage of the split."
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_features()
    predictors = select_predictors(df)
    y = df[config.LABEL_COL].astype(int)

    logger.info("Spatial CV: parent res=%d, folds=%d, buffer=%d ring(s)",
                config.SPATIAL_CV_BLOCK_RES, config.SPATIAL_CV_FOLDS,
                config.SPATIAL_CV_BUFFER_RINGS)
    proba_oof, fold_info = spatial_cv_oof(df, predictors)
    oof = evaluate_scores(y.to_numpy(), proba_oof, k=config.TOP_K)
    logger.info("OOF -> ROC-AUC=%.4f | PR-AUC=%.4f | NDCG@%d=%.4f",
                oof["roc_auc"], oof["pr_auc"], config.TOP_K, oof["ranking"]["ndcg_at_k"])

    v2 = v2_random_split_metrics(df, predictors)
    logger.info("v2 (random) -> ROC-AUC=%.4f | NDCG@%d=%.4f (for comparison)",
                v2["roc_auc"], config.TOP_K, v2["ranking"]["ndcg_at_k"])

    # Final full-data model for the production ranking.
    final_model = build_model()
    final_model.fit(df[predictors], y)
    score_prod = final_model.predict_proba(df[predictors])[:, 1]
    coefs = coefficients_table(final_model, predictors)

    ranking = df[["h3_index", "lat_centroid", "lon_centroid", config.LABEL_COL]].copy()
    ranking["score_lookalike_v3"] = score_prod   # production (full-data model)
    ranking["score_oof"] = proba_oof             # honest (transparency)
    ranking["rank_lookalike_v3"] = (
        ranking["score_lookalike_v3"].rank(ascending=False, method="first").astype(int)
    )
    ranking = ranking.sort_values("rank_lookalike_v3").reset_index(drop=True)
    precision_v3 = float(
        df.assign(p=proba_oof).sort_values("p", ascending=False)
        .head(config.TOP_K)[config.LABEL_COL].mean()
    )  # Honest Precision@K (over OOF)

    ranking.to_parquet(config.LOOKALIKE_V3_RANKING_PARQUET_PATH, index=False)
    ranking.to_csv(config.LOOKALIKE_V3_RANKING_CSV_PATH, index=False)
    joblib.dump(final_model, config.LOOKALIKE_V3_MODEL_PATH)
    logger.info("Ranking -> %s (+ .csv) | model -> %s",
                config.LOOKALIKE_V3_RANKING_PARQUET_PATH.name,
                config.LOOKALIKE_V3_MODEL_PATH.name)

    write_summary(df, predictors, oof, v2, fold_info, coefs, precision_v3)
    logger.info("v3 (spatial CV) complete.")


if __name__ == "__main__":
    main()
