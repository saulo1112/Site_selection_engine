"""MODEL v1 — MCDA baseline (Multi-Criteria Decision Analysis, no ML).

Computes a weighted, interpretable score per hexagon from the features table,
normalizing each variable (min-max) and combining it with a-priori weights defined by
business reasoning (NOT tuned against the label). The result is an exploration-priority
ranking, NOT a performance prediction (see docs/metodologia.md §5).

Anti-leakage (critical): features derived from the D1 location
(`config.MCDA_LEAKAGE_COLS`) are EXCLUDED from the score, because the label
`tiene_d1 = (n_d1_300m >= 1)` is a direct function of them. The label is only used
AFTERWARDS, as an honest post-hoc validation (NDCG@K, top-K hitting/loss), never as an
input.

Run standalone (requires data/processed/features.parquet from STAGE 4):
    uv run python -m src.models.mcda
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.logging_config import get_logger
from src.models.metrics import ranking_report

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Loading
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


# --------------------------------------------------------------------------- #
# Normalization and score
# --------------------------------------------------------------------------- #
def _minmax(series: pd.Series, direction: int) -> pd.Series:
    """Normalizes to [0, 1]. direction=-1 inverts (lower-is-better -> high after normalizing)."""
    lo, hi = series.min(), series.max()
    if hi == lo:  # constant column: contributes 0 discrimination
        return pd.Series(0.0, index=series.index)
    norm = (series - lo) / (hi - lo)
    return (1.0 - norm) if direction < 0 else norm


def _usable_features(df: pd.DataFrame) -> dict[str, list[tuple[str, int]]]:
    """Per group, filters features that are present, non-leakage and have data (not all-NaN).

    Features in `config.MCDA_LEAKAGE_COLS` are explicitly excluded.
    Columns that are 100% null (e.g. demographics without census) are dropped and logged.
    """
    usable: dict[str, list[tuple[str, int]]] = {}
    for group, feats in config.MCDA_GROUP_FEATURES.items():
        kept: list[tuple[str, int]] = []
        for col, direction in feats:
            if col in config.MCDA_LEAKAGE_COLS:
                logger.warning("Excluded from MCDA score due to LEAKAGE: %s", col)
                continue
            if col not in df.columns:
                logger.warning("Feature missing, skipped: %s", col)
                continue
            if not df[col].notna().any():
                logger.warning("Feature 100%% null, skipped from group '%s': %s", group, col)
                continue
            kept.append((col, direction))
        if kept:
            usable[group] = kept
        else:
            logger.warning("Group '%s' has no usable features -> skipped.", group)
    return usable


def _effective_group_weights(usable: dict[str, list[tuple[str, int]]]) -> dict[str, float]:
    """Renormalizes group weights to 1.0 over the groups actually present."""
    present = {g: config.MCDA_GROUP_WEIGHTS[g] for g in usable}
    total = sum(present.values())
    if total == 0:
        raise RuntimeError("No feature group remained available for the MCDA.")
    eff = {g: w / total for g, w in present.items()}
    for g, w in eff.items():
        if abs(w - config.MCDA_GROUP_WEIGHTS[g]) > 1e-9:
            logger.info("Weight of group '%s' renormalized: %.3f -> %.3f",
                        g, config.MCDA_GROUP_WEIGHTS[g], w)
    return eff


def compute_mcda_score(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Adds `score_mcda` and `rank_mcda`. Returns (df, effective-weights metadata)."""
    usable = _usable_features(df)
    group_w = _effective_group_weights(usable)

    out = df.copy()
    score = pd.Series(0.0, index=out.index)
    feature_weights: dict[str, float] = {}

    for group, feats in usable.items():
        per_feature_w = group_w[group] / len(feats)  # uniform weight within the group
        for col, direction in feats:
            contribution = _minmax(out[col].astype(float), direction) * per_feature_w
            score = score + contribution
            feature_weights[col] = per_feature_w

    out["score_mcda"] = score
    # rank 1 = best (highest score). 'first' for deterministic tie-breaking.
    out["rank_mcda"] = out["score_mcda"].rank(ascending=False, method="first").astype(int)
    out = out.sort_values("rank_mcda").reset_index(drop=True)

    meta = {
        "usable": usable,
        "group_weights": group_w,
        "feature_weights": feature_weights,
        "excluded_leakage": list(config.MCDA_LEAKAGE_COLS),
    }
    logger.info("MCDA score computed with %d features across %d groups.",
                len(feature_weights), len(group_w))
    return out, meta


# --------------------------------------------------------------------------- #
# Honest evaluation (post-hoc, NOT used to tune weights)
# --------------------------------------------------------------------------- #
def evaluate_against_label(df: pd.DataFrame) -> dict[str, float]:
    """Validates the MCDA ranking against `tiene_d1` with ranking metrics (top-K)."""
    if "tiene_d1" not in df.columns:
        logger.warning("No 'tiene_d1' column -> skipping post-hoc evaluation.")
        return {}
    report = ranking_report(df["score_mcda"].to_numpy(), df["tiene_d1"].to_numpy(),
                            k=config.TOP_K)
    logger.info("Post-hoc eval @K=%d -> NDCG=%.4f | hitting=%.4f | loss=%.4f",
                int(report["k"]), report["ndcg_at_k"], report["topk_hitting"],
                report["topk_loss"])
    return report


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def write_summary(df: pd.DataFrame, meta: dict, metrics: dict[str, float]) -> None:
    n_total = len(df)
    k = config.TOP_K
    topk = df.head(k)
    n_pos_total = int(df["tiene_d1"].sum()) if "tiene_d1" in df.columns else 0
    n_pos_topk = int(topk["tiene_d1"].sum()) if "tiene_d1" in df.columns else 0
    precision_topk = (n_pos_topk / k) if k else float("nan")
    # Hitting ceiling: with more positives than K, not even a perfect ranking can
    # exceed K/positives. Reporting it avoids misreading a "low" hitting value.
    hitting_ceiling = (min(k, n_pos_total) / n_pos_total) if n_pos_total else float("nan")

    # Per-feature weight table.
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
        "# Results v1 — MCDA baseline\n",
        f"_Generated by `src/models/mcda.py`. Total hexagons: **{n_total}**._\n",
        "## What this version is\n",
        "Weighted, **interpretable** score, no ML: min-max normalization per feature "
        "and linear combination with a-priori weights defined by business reasoning "
        "(not tuned against the label). It is the **baseline** against which "
        "v2 (look-alike classifier) and v3 (spatial CV) will be measured.\n",
        "## Anti-leakage\n",
        "Features derived from the D1 location "
        f"(`{'`, `'.join(meta['excluded_leakage'])}`) are **excluded from the score**: the "
        "`tiene_d1` label is defined as `n_d1_300m >= 1`, so using them would be "
        "tautological leakage. The label is used **only afterwards**, as an "
        "honest validation (ranking metrics below), never as an input to the score.\n",
    ]
    if excluded_demo:
        lines.append(
            "> **Note:** the demographic features "
            f"(`{'`, `'.join(excluded_demo)}`) are not available in this run "
            "(DANE census not loaded, see `src/data/load_censo.py`). Its group weight was "
            "redistributed proportionally among the present groups.\n"
        )

    lines += [
        "## Effective weights per group\n",
        "| Group | Effective weight |",
        "|---|---|",
        *[f"| {g} | {w:.4f} |" for g, w in meta["group_weights"].items()],
        "\n## Weights per feature\n",
        "| Feature | Group | Weight |",
        "|---|---|---|",
        *weight_rows,
        "\n## Honest post-hoc evaluation (ranking vs. `tiene_d1`)\n",
        "These metrics were **not** used to choose weights; they only measure, after "
        "the fact, how well the MCDA ranking recovers the cells where D1 is already present.\n",
        f"- **NDCG@{k}**: {metrics.get('ndcg_at_k', float('nan')):.4f} "
        "(quality of the ordering in the top-K; 1.0 = perfect).",
        f"- **Precision@{k}**: {precision_topk:.4f} "
        f"({n_pos_topk} of the {k} top-ranked cells already have D1).",
        f"- **top-{k} hitting** (recall of positives in the top-K): "
        f"{metrics.get('topk_hitting', float('nan')):.4f} "
        f"(ceiling = {hitting_ceiling:.4f}, because there are {n_pos_total} positives > K={k}: "
        "not even a perfect ranking can capture them all within just K cells).",
        f"- **top-{k} loss** (positives left out of the top-K): "
        f"{metrics.get('topk_loss', float('nan')):.4f}",
        f"- Positives in the top-{k}: **{n_pos_topk}** out of {n_pos_total} total positives "
        f"({n_total} hexagons).\n",
        "> **Honest reading:** `hitting` looks low only because there are far more "
        f"positives ({n_pos_total}) than selected cells (K={k}). The more informative "
        f"metric here is **Precision@{k} = {precision_topk:.3f}** and the "
        f"**NDCG@{k} = {metrics.get('ndcg_at_k', float('nan')):.3f}**: a no-ML baseline "
        "that ranks cells similar to D1's fairly well. v2/v3 should improve on it "
        "(or, in v3's case, reveal how much of this was spatial leakage).\n",
        "## Limitation\n",
        "The score reflects **similarity / exploration priority**, not performance: it "
        "measures resemblance to cells where D1 already operates, not expected sales. It "
        "inherits the look-alike assumption that D1's location strategy is good "
        "(see docs/metodologia.md §5).\n",
    ]
    config.MCDA_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary written -> %s", config.MCDA_SUMMARY_PATH.name)


# --------------------------------------------------------------------------- #
# Orchestration
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
    logger.info("Ranking saved -> %s (+ .csv)", config.MCDA_RANKING_PARQUET_PATH.name)

    write_summary(df, meta, metrics)
    logger.info("v1 (MCDA) complete.")


if __name__ == "__main__":
    main()
