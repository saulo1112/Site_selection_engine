# Methodology — Site Selection Engine

> Living document. Sections marked _(pending)_ are completed as the v1 → v2 → v3
> iteration progresses.

---

## 1. Objective and business value

**Problem.** A retail chain / franchise planning to expand in a city needs to decide
**where** to open new locations. Doing so "by eye" or based only on real-estate intuition
is costly and biased.

**System objective.** Given a **business sector** and a **city**, produce a
**ranking of candidate locations** over an H3 hexagonal grid covering the city,
prioritizing cells based on:
- density of **competitors / complementary businesses** (OSM/Overpass POIs),
- **demographic** variables (DANE — CNPV 2018),
- **road accessibility** (OSM street network).

**Look-alike approach.** **D1 Stores** are used as the reference brand (aggressive
and recent expansion in Colombia, good OSM tagging as `shop=supermarket` +
`brand=D1`). The hypothesis: cells that "resemble" those D1 has already chosen are
good exploration candidates.

**Value delivered.** A **screening** tool that narrows the universe of
locations down to a prioritized, auditable list, not a replacement for the final decision.
The score is one of **exploration priority / similarity**, not a sales prediction
(see §2 and §5).

---

## 2. Comparison with the reference paper (Lu et al., 2024)

> Lu, *et al.* (2024). *Retail store location screening: A machine learning-based
> approach.* **Journal of Retailing and Consumer Services (JRCS).**

| Dimension | Lu et al. (2024) | This project |
|---|---|---|
| Unit of analysis | Discrete shopping malls | **H3 hexagonal grid** over the city |
| Target variable | Actual revenue (ASMR — *Average Store Monthly Revenue*) | **Look-alike label** (proxy: is there a D1 in the cell?) |
| Problem type | Performance regression | **Similarity classification + ranking** |
| Evaluation metrics | NDCG, top-K hitting, top-K loss (+ RMSE) | NDCG, top-K hitting, top-K loss (**shared**) |
| Spatial leakage risk | Not central to their design (discrete, dispersed units) | **High**: neighboring cells are correlated → spatial autocorrelation; corrected with **spatial CV** in v3 |
| Score interpretation | Prediction of economic performance | **Similarity / exploration priority** (not performance) |

**What we take from the paper:**
1. **Staged screening framework** (progressive filtering of candidates).
2. **Ranking metrics** beyond plain accuracy/RMSE: **NDCG**, **top-K hitting**,
   **top-K loss** — appropriate for "did we get the best K locations right?".
3. **Sequential ensemble** (Lasso-first for variable selection + a second model
   over the residuals) as a possible improvement for v3.

**Key difference to document honestly.** Their target is **actual revenue**; ours
is a **look-alike (proxy) label**. That is why our scores represent
**similarity / exploration priority**, not performance prediction. We inherit the
strong assumption that **D1's location strategy is good** (see §5).

---

## 3. Data and justification

| Source | Use | Granularity | Note |
|---|---|---|---|
| OSM / Overpass | POIs (competitors, complementary businesses), D1 label | Point | Signal measured in the area check |
| OSM (road network) | Accessibility (via `osmnx`) | Arc/node | Centrality, distance to roads |
| DANE — CNPV 2018 / MGN | Demographics (population, housing units, households) | City block / sector | Uniform national coverage |
| Municipal stratification | Socioeconomic stratum (income proxy) | City block / block side | **NOT** a census variable; varies by city |

The choice of **study city** and the *data-driven* justification of the
availability of these sources are in **[seleccion_area_estudio.md](seleccion_area_estudio.md)**.

> **Note on the D1 count.** The city check used a strict Overpass query
> (`brand` only, 129 for Bogotá) to compare the 4 cities under a uniform criterion.
> The production pipeline uses a more permissive query (`brand` or `name~"D1"`) and loads
> **166** D1 POIs (`data/raw/pois_d1.geojson`) — the actual number that enters the model.
> See the full reconciliation in `seleccion_area_estudio.md` §5.

---

## 4. Iteration plan v1 → v2 → v3 → v4

The honest iteration (replicating the EUDR project standard) documents not only the
final model but **why** each version was insufficient.

### v1 — MCDA baseline (no ML)
- **What.** *Multi-Criteria Decision Analysis*: weighted score of normalized variables
  (min-max), a priori weights per group (competitors 41%, complementary 41%,
  road accessibility 18% after renormalizing — demographics were left out because the census
  is not loaded). Implemented in `src/models/mcda.py`; reusable metrics in
  `src/models/metrics.py`.
- **Why first.** Interpretable, cheap baseline; a reference against which to measure
  whether ML adds anything.
- **Anti-leakage.** Features derived from D1 (`n_d1_300m`, `n_d1_500m`, `dist_d1_km`)
  are **excluded from the score** (the `tiene_d1` label is a direct function of them). Also,
  the competitor features measure only **non-D1** competitors (see §6, feature leakage
  discovered and corrected). The label is used only as post-hoc validation.
- **Results** (see [v1_mcda_resultados.md](v1_mcda_resultados.md), post feature-leakage
  correction). Honest evaluation over the 3589 hexagons, K=200:
  - **NDCG@200 = 0.8033**, **Precision@200 = 0.765**, **top-200 hitting = 0.1741**
    (ceiling 0.2275, since there are 879 positives ≫ K=200).
  - Reading: an ML-free baseline ranks the cells most similar to D1's reasonably well. v2/v3
    should beat it — or, in v3, reveal how much of this holds up without spatial leakage.

### v2 — Naive look-alike classifier
- **What.** Logistic Regression (`class_weight='balanced'`, standardized features) that
  estimates `P(tiene_d1=1)` from the non-D1 features; the probability is the
  look-alike score. **Random stratified** train/test split 75/25. Implemented in
  `src/models/lookalike.py`.
- **Classes.** Binary: class 1 = hexagon with ≥1 D1 within ≤300 m ("D1-type cell"); class 0 = no nearby D1.
- **Known risk from the outset.** The random split mixes neighboring cells between train and
  test → **leakage from spatial autocorrelation** → overly optimistic metrics (corrected by v3).
- **Results** (see [v2_lookalike_resultados.md](v2_lookalike_resultados.md), post
  feature-leakage correction). Test: **ROC-AUC = 0.7801**, **PR-AUC = 0.5970**;
  class 1 with **recall = 0.686** (predicts both classes, does not collapse). Ranking over the grid:
  **NDCG@200 = 0.8349**, **Precision@200 = 0.805** — slightly beats v1. The real advantage
  over v1 will be confirmed (or not) in v3 once spatial leakage is removed.

### v3 — Same model with Spatial CV
- **What.** Identical Logistic Regression, but validated with **spatial cross-validation**:
  H3 blocks at parent resolution 6 (24 blocks of ~36 km²), `StratifiedGroupKFold` with 5
  folds, and a **1-ring buffer** (cells within ≤1 ring of any test cell are excluded from
  train). Each hexagon is predicted **out-of-fold** by a model that did not see its
  neighborhood. Implemented in `src/models/lookalike_v3.py` and `src/models/spatial_cv.py`.
- **Why.** Provides an **honest** estimate of generalization and measures how much of v2's
  performance was spatial leakage.
- **Results** (see [v3_spatial_cv_resultados.md](v3_spatial_cv_resultados.md)).
  OOF: **ROC-AUC = 0.7934**, **PR-AUC = 0.5899**, **NDCG@200 = 0.8400**,
  **Precision@200 = 0.815**; class 1 with recall = 0.718 (does not collapse).
- **Honest finding (against the initial hypothesis).** We expected a **drop** in metrics
  as evidence of spatial leakage. **It did not happen**: v3 matches (even slightly exceeds)
  v2 (Δ NDCG@200 = +0.0051, Δ ROC-AUC = +0.0132). Interpretation: with a linear model
  over buffer features (smooth spatial fields), a random split and a spatial one
  generalize similarly; the non-D1 signal **holds up in unseen areas**, it was not
  a mirage of the split. Documenting this — and not forcing the expected narrative — is exactly
  the honest iteration of the EUDR standard.
- **Possible improvement (from the paper, future work).** Sequential ensemble Lasso-first + 2nd model
  over residuals; especially useful if demographics/stratum are added in the future and
  non-linearities appear.

### v4 — v3 + demographics (DANE + IDECA)
- **What.** Same model (Logistic Regression) and **same** spatial CV scheme as v3, but
  adding demographic features prorated by city block: **population** and **housing units**
  (DANE CNPV 2018 census) and socioeconomic **stratum** (IDECA Bogotá). Implemented in
  `src/models/lookalike_v4.py`.
- **Proration (two natures).** Population/housing units are **extensive** quantities → sum
  weighted by the fraction of each block's area within the hexagon (no double counting),
  via `_prorate_sum_expr`. Stratum is **intensive/ordinal** (1-6) → **weighted average**
  by intersection area (not summed), discarding stratum 0/non-residential, via
  `_prorate_avg_expr` (both in `src/data/features.py`).
- **Partial NULL + imputation.** City blocks do not cover the entire grid (non-residential
  / no-stratum areas) → demographics have partial NULLs. Instead of discarding hexagons, the `Pipeline`
  incorporates a `SimpleImputer(median)` fit **within each fold** (`build_model` in
  `src/models/lookalike.py`), with no leakage between train and test. **Coverage** (% with data)
  is reported in [features_summary.md](features_summary.md) and in the v4 results.
- **Isolating the contribution (honest design).** v4 evaluates, under identical spatial CV, two sets
  of predictors: **BASE** (without demographics, = v3) and **FULL** (+ demographics, = v4). The Δ of
  the OOF metrics isolates the contribution of demographics, not of the validation method. If
  demographics **do not** move the metrics, this is reported as-is (same criterion as v3's honest
  finding); v4 is adopted for production only if it improves on or ties v3 with better
  interpretability.
- **Business hypothesis to verify.** D1 is *hard-discount* focused on lower strata →
  `estrato_promedio` is expected to have a **negative** coefficient (the lower the stratum, the higher
  P(D1-type)). The LR coefficient in v4 confirms this, or not.
- **Results.** Generated when the layers are loaded and the module is run, in
  [v4_demografia_resultados.md](v4_demografia_resultados.md).

---

## 5. Honest limitations

1. **The score reflects similarity, not performance.** It measures resemblance to cells with D1, not
   expected sales. A "high score" means "worth exploring", not "will be profitable".
2. **Look-alike assumption.** Assumes D1's location strategy is good. If D1
   is systematically wrong, the model replicates its bias.
3. **OSM tagging bias.** POI counts depend on how well-mapped the city is;
   under-mapped areas look "empty" without actually being so.
4. **Stratum as an income proxy.** Socioeconomic stratum approximates income but
   is not income; its availability and currency vary by city.
5. **Temporal staticness.** The census is from 2018; OSM is dynamic but unevenly maintained.

---

## 6. Explicit leakage check

Two types of leakage are distinguished, with different treatments.

### 6.1 Feature/target leakage (discovered and corrected)

- **Type (a) — tautological, always excluded.** The `tiene_d1` label is defined as
  `n_d1_300m >= 1`. By construction, features derived from D1 (`n_d1_300m`,
  `n_d1_500m`, `dist_d1_km`) are a direct function of the label. They are **never**
  used as predictors (neither in MCDA nor in the LR); see `config.MCDA_LEAKAGE_COLS`.
- **Type (b) — D1 inside "competitors" (discovered during v2, corrected).** The
  competitor features (`n_supermercados_500m`, `dist_supermercado_km`) were computed
  over `pois_competidores`, which **included D1**. Since every positive has a D1 within ≤300 m,
  that same D1 counted as a "supermarket": **100%** of the positives ended up with
  `dist_supermercado_km ≤ 0.30` (mechanical cap) and `n_supermercados_500m ≥ 1`. In the LR the
  coefficient of `dist_supermercado_km` spiked to **-6.33**, dominating the model.
  - **Fix.** In `src/data/features.py` the competitor subqueries now filter
    `COALESCE(es_d1, 0) = 0` (they measure only **non-D1** competitors; D1 is the
    look-alike target, not a competitor to measure).
  - **Evidence of the fix.** Positives with `dist_supermercado_km ≤ 0.30`: **100% → 65.1%**;
    with `n_supermercados_500m ≥ 1`: **100% → 81.7%**; LR coefficient: **-6.33 → -1.09**.
    Honest drop in metrics: v1 NDCG@200 0.8495→0.8033; v2 ROC-AUC 0.9177→0.7801,
    PR-AUC 0.7724→0.5970. The residual correlation `dist_supermercado_km`↔`dist_d1_km` =
    **0.7653** (computed in `src/data/features.py::write_summary()`, see
    [features_summary.md](features_summary.md)) is genuine co-location (legitimate
    look-alike signal), not leakage.

### 6.2 Spatial-autocorrelation leakage (v2 → v3)

- **Mechanism (v2).** Neighboring H3 cells have correlated features and labels; a
  random split distributes them between train and test → potentially optimistic metrics.
- **Fix (v3).** Spatial CV: H3 blocks at parent resolution 6, `StratifiedGroupKFold`
  with 5 folds and a **1-ring** H3 buffer (`grid_disk`) excluded from train around each
  test cell. Out-of-fold predictions = honest estimate.
- **Result (measured, not expected).** The drop **did not** materialize: v3 matches/slightly
  exceeds v2. Spatial-autocorrelation leakage was **smaller than anticipated**
  for this linear model over buffer features. This is an honest finding: the non-D1 signal
  generalizes to unseen areas. (The buffer radius was not tuned to "manufacture" a drop;
  1 ring is consistent with cells of ~174 m edge and blocks of ~6 km.)

| Metric (K=200) | v2 (random split) | v3 (spatial CV, OOF) | Δ (v3 − v2) |
|---|---|---|---|
| ROC-AUC | 0.7801 | 0.7934 | +0.0132 |
| PR-AUC | 0.5970 | 0.5899 | −0.0071 |
| NDCG@200 | 0.8349 | 0.8400 | +0.0051 |
| top-200 hitting | 0.1832 | 0.1854 | +0.0023 |

---

## References

- Lu, *et al.* (2024). *Retail store location screening: A machine learning-based
  approach.* Journal of Retailing and Consumer Services.
- DANE (2018). National Population and Housing Census (CNPV) — National
  Geostatistical Framework (MGN). https://geoportal.dane.gov.co/
