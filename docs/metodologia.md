# Metodología — Site Selection Engine

> Documento vivo. Las secciones marcadas con _(pendiente)_ se completan a medida que
> avanza la iteración v1 → v2 → v3.

---

## 1. Objetivo y valor de negocio

**Problema.** Una cadena de retail / franquicia que planea expandirse en una ciudad
necesita decidir **dónde** abrir nuevos puntos. Hacerlo "a ojo" o solo por intuición
inmobiliaria es costoso y sesgado.

**Objetivo del sistema.** Dado un **sector de negocio** y una **ciudad**, producir un
**ranking de ubicaciones candidatas** sobre un grid hexagonal H3 que cubre la ciudad,
priorizando celdas según:
- densidad de **competencia / complementarios** (POIs de OSM/Overpass),
- variables **demográficas** (DANE — CNPV 2018),
- **accesibilidad vial** (red de calles de OSM).

**Enfoque look-alike.** Se usa **Tiendas D1** como marca de referencia (expansión
agresiva y reciente en Colombia, buen etiquetado en OSM como `shop=supermarket` +
`brand=D1`). La hipótesis: las celdas que "se parecen" a las que D1 ya eligió son
buenas candidatas de exploración.

**Valor entregado.** Una herramienta de **screening** que reduce el universo de
ubicaciones a una lista priorizada y auditable, no un reemplazo de la decisión final.
El score es de **prioridad de exploración / similitud**, no una predicción de ventas
(ver §2 y §5).

---

## 2. Comparación con el paper de referencia (Lu et al., 2024)

> Lu, *et al.* (2024). *Retail store location screening: A machine learning-based
> approach.* **Journal of Retailing and Consumer Services (JRCS).**

| Dimensión | Lu et al. (2024) | Este proyecto |
|---|---|---|
| Unidad de análisis | Centros comerciales discretos | **Grid hexagonal H3** sobre la ciudad |
| Variable objetivo | Ingreso real (ASMR — *Average Store Monthly Revenue*) | **Etiqueta look-alike** (proxy: ¿hay D1 en la celda?) |
| Tipo de problema | Regresión de desempeño | **Clasificación de similitud + ranking** |
| Métricas de evaluación | NDCG, top-K hitting, top-K loss (+ RMSE) | NDCG, top-K hitting, top-K loss (**compartidas**) |
| Riesgo de leakage espacial | No central en su diseño (unidades discretas, dispersas) | **Alto**: celdas vecinas correlacionadas → autocorrelación espacial; corregido con **spatial CV** en v3 |
| Interpretación del score | Predicción de desempeño económico | **Similitud / prioridad de exploración** (no desempeño) |

**Qué tomamos del paper:**
1. **Framework de screening por etapas** (filtrado progresivo de candidatos).
2. **Métricas de ranking** más allá de accuracy/RMSE plano: **NDCG**, **top-K hitting**,
   **top-K loss** — apropiadas para "¿acerté las mejores K ubicaciones?".
3. **Ensemble secuencial** (Lasso-first para selección de variables + un segundo modelo
   sobre los residuales) como posible mejora para v3.

**Diferencia clave a documentar honestamente.** Su objetivo es **ingreso real**; el
nuestro es una **etiqueta look-alike (proxy)**. Por eso nuestros scores son de
**similitud / prioridad de exploración**, no de predicción de desempeño. Heredamos el
supuesto fuerte de que **la estrategia de localización de D1 es buena** (ver §5).

---

## 3. Datos y justificación

| Fuente | Uso | Granularidad | Nota |
|---|---|---|---|
| OSM / Overpass | POIs (competencia, complementarios), etiqueta D1 | Punto | Señal medida en el chequeo de área |
| OSM (red vial) | Accesibilidad (vía `osmnx`) | Arco/nodo | Centralidad, distancia a vías |
| DANE — CNPV 2018 / MGN | Demografía (población, viviendas, hogares) | Manzana / sector | Cobertura nacional uniforme |
| Estratificación municipal | Estrato socioeconómico (proxy de ingreso) | Manzana / lado de manzana | **NO** es variable del censo; varía por ciudad |

La elección de **ciudad de estudio** y la justificación *data-driven* de la
disponibilidad de estas fuentes están en **[seleccion_area_estudio.md](seleccion_area_estudio.md)**.

---

## 4. Plan de iteración v1 → v2 → v3

La iteración honesta (replicando el estándar del proyecto EUDR) documenta no solo el
modelo final sino **por qué** cada versión fue insuficiente.

### v1 — MCDA baseline (sin ML)
- **Qué.** *Multi-Criteria Decision Analysis*: score ponderado de variables normalizadas
  (min-max), pesos a priori por grupo (competencia 41%, complementarios 41%,
  accesibilidad vial 18% tras renormalizar — la demografía quedó fuera porque el censo
  no está cargado). Implementado en `src/models/mcda.py`; métricas reutilizables en
  `src/models/metrics.py`.
- **Por qué primero.** Línea base interpretable y barata; referencia contra la cual medir
  si el ML aporta algo.
- **Anti-leakage.** Las features derivadas de D1 (`n_d1_300m`, `n_d1_500m`, `dist_d1_km`)
  se **excluyen del score** (la etiqueta `tiene_d1` es función directa de ellas). Además,
  las features de competencia miden solo competidores **no-D1** (ver §6, leakage de
  features descubierto y corregido). La etiqueta solo se usa como validación post-hoc.
- **Resultados** (ver [v1_mcda_resultados.md](v1_mcda_resultados.md), post-corrección de
  leakage de features). Evaluación honesta sobre los 3589 hexágonos, K=200:
  - **NDCG@200 = 0.8033**, **Precision@200 = 0.765**, **top-200 hitting = 0.1741**
    (techo 0.2275, pues hay 879 positivos ≫ K=200).
  - Lectura: un baseline sin ML ordena bien las celdas más parecidas a las de D1. v2/v3
    deberán superarlo —o, en v3, revelar cuánto de esto se sostiene sin leakage espacial.

### v2 — Clasificador look-alike naïve
- **Qué.** Regresión Logística (`class_weight='balanced'`, features estandarizadas) que
  estima `P(tiene_d1=1)` desde las features no-D1; el probabilístico es el score
  look-alike. Split train/test **aleatorio estratificado** 75/25. Implementado en
  `src/models/lookalike.py`.
- **Clases.** Binaria: clase 1 = hexágono con ≥1 D1 a ≤300 m ("tipo-D1"); clase 0 = sin D1.
- **Riesgo conocido de antemano.** El split aleatorio mezcla celdas vecinas entre train y
  test → **leakage por autocorrelación espacial** → métricas optimistas (lo corrige v3).
- **Resultados** (ver [v2_lookalike_resultados.md](v2_lookalike_resultados.md), post-
  corrección de leakage de features). Test: **ROC-AUC = 0.7801**, **PR-AUC = 0.5970**;
  clase 1 con **recall = 0.686** (predice ambas clases, no colapsa). Ranking sobre el grid:
  **NDCG@200 = 0.8349**, **Precision@200 = 0.805** — supera levemente a v1. La ventaja real
  sobre v1 se confirmará (o no) en v3 al quitar el leakage espacial.

### v3 — Mismo modelo con Spatial CV
- **Qué.** Idéntica Regresión Logística, pero validación con **spatial cross-validation**:
  bloques H3 a resolución padre 6 (24 bloques de ~36 km²), `StratifiedGroupKFold` de 5
  folds, y **buffer de 1 anillo** (se excluyen del train las celdas a ≤1 anillo de
  cualquier celda de test). Cada hexágono se predice **out-of-fold** por un modelo que no
  vio su vecindario. Implementado en `src/models/lookalike_v3.py` y `src/models/spatial_cv.py`.
- **Por qué.** Da una estimación **honesta** de generalización y mide cuánto del desempeño
  de v2 era leakage espacial.
- **Resultados** (ver [v3_spatial_cv_resultados.md](v3_spatial_cv_resultados.md)).
  OOF: **ROC-AUC = 0.7934**, **PR-AUC = 0.5899**, **NDCG@200 = 0.8400**,
  **Precision@200 = 0.815**; clase 1 con recall = 0.718 (no colapsa).
- **Hallazgo honesto (contra la hipótesis inicial).** Esperábamos una **caída** de métricas
  como evidencia de leakage espacial. **No ocurrió**: v3 iguala (incluso supera levemente)
  a v2 (Δ NDCG@200 = +0.0051, Δ ROC-AUC = +0.0132). Interpretación: con un modelo lineal
  sobre features de buffer (campos espaciales suaves), un split aleatorio y uno espacial
  generalizan parecido; la señal no-D1 **se sostiene en zonas no vistas**, no era un
  espejismo del split. Documentar esto —y no forzar la narrativa esperada— es justamente
  la iteración honesta del estándar EUDR.
- **Posible mejora (del paper, futuro).** Ensemble secuencial Lasso-first + 2º modelo sobre
  residuales; útil sobre todo si en el futuro se añade demografía/estrato y aparecen
  no-linealidades.

---

## 5. Limitaciones honestas

1. **El score es de similitud, no de desempeño.** Mide parecido a las celdas con D1, no
   ventas esperadas. Un "score alto" = "vale la pena explorar", no "será rentable".
2. **Supuesto look-alike.** Asume que la estrategia de localización de D1 es buena. Si D1
   se equivoca sistemáticamente, el modelo replica su sesgo.
3. **Sesgo de etiquetado OSM.** El conteo de POIs depende de qué tan bien mapeada está la
   ciudad; zonas sub-mapeadas parecen "vacías" sin estarlo.
4. **Estrato como proxy de ingreso.** El estrato socioeconómico aproxima el ingreso pero
   no lo es; su disponibilidad y vigencia varían por ciudad.
5. **Estática temporal.** El censo es de 2018; OSM es dinámico pero desigual.

---

## 6. Chequeo explícito de leakage

Se distinguen **dos** tipos de leakage, con tratamientos distintos.

### 6.1 Leakage de features / target (descubierto y corregido)

- **Tipo (a) — tautológico, siempre excluido.** La etiqueta `tiene_d1` se define como
  `n_d1_300m >= 1`. Por construcción, las features derivadas de D1 (`n_d1_300m`,
  `n_d1_500m`, `dist_d1_km`) son función directa de la etiqueta. **Nunca** se usan como
  predictores (ni en MCDA ni en la LR); ver `config.MCDA_LEAKAGE_COLS`.
- **Tipo (b) — D1 dentro de "competidores" (descubierto durante v2, corregido).** Las
  features de competencia (`n_supermercados_500m`, `dist_supermercado_km`) se calculaban
  sobre `pois_competidores`, que **incluía a D1**. Como todo positivo tiene un D1 a ≤300 m,
  ese mismo D1 contaba como "supermercado": el **100 %** de los positivos quedaba con
  `dist_supermercado_km ≤ 0.30` (cap mecánico) y `n_supermercados_500m ≥ 1`. En la LR el
  coeficiente de `dist_supermercado_km` se disparaba a **-6.33**, dominando el modelo.
  - **Corrección.** En `src/data/features.py` las subconsultas de competencia ahora filtran
    `COALESCE(es_d1, 0) = 0` (miden solo competidores **no-D1**; D1 es el objetivo
    look-alike, no un competidor a medir).
  - **Evidencia del fix.** Positivos con `dist_supermercado_km ≤ 0.30`: **100 % → 65.1 %**;
    con `n_supermercados_500m ≥ 1`: **100 % → 81.7 %**; coeficiente LR: **-6.33 → -1.09**.
    Caída honesta de métricas: v1 NDCG@200 0.8495→0.8033; v2 ROC-AUC 0.9177→0.7801,
    PR-AUC 0.7724→0.5970. La correlación residual `dist_supermercado`↔`dist_d1`=0.765 es
    co-localización real (señal legítima del look-alike), no leakage.

### 6.2 Leakage por autocorrelación espacial (v2 → v3)

- **Mecanismo (v2).** Celdas H3 vecinas tienen features y etiqueta correlacionadas; un
  split aleatorio las reparte entre train y test → métricas potencialmente optimistas.
- **Corrección (v3).** Spatial CV: bloques H3 a resolución padre 6, `StratifiedGroupKFold`
  de 5 folds y **buffer de 1 anillo** H3 (`grid_disk`) excluido del train alrededor de cada
  celda de test. Predicciones out-of-fold = estimación honesta.
- **Resultado (medido, no esperado).** La caída **no** se materializó: v3 iguala/supera
  levemente a v2. El leakage por autocorrelación espacial era **menor de lo anticipado**
  para este modelo lineal sobre features de buffer. Es un hallazgo honesto: la señal no-D1
  generaliza a zonas no vistas. (No se ajustó el radio de buffer para "fabricar" una caída;
  1 anillo es coherente con celdas de ~174 m de arista y bloques de ~6 km.)

| Métrica (K=200) | v2 (split aleatorio) | v3 (spatial CV, OOF) | Δ (v3 − v2) |
|---|---|---|---|
| ROC-AUC | 0.7801 | 0.7934 | +0.0132 |
| PR-AUC | 0.5970 | 0.5899 | −0.0071 |
| NDCG@200 | 0.8349 | 0.8400 | +0.0051 |
| top-200 hitting | 0.1832 | 0.1854 | +0.0023 |

---

## Referencias

- Lu, *et al.* (2024). *Retail store location screening: A machine learning-based
  approach.* Journal of Retailing and Consumer Services.
- DANE (2018). Censo Nacional de Población y Vivienda (CNPV) — Marco Geoestadístico
  Nacional (MGN). https://geoportal.dane.gov.co/
