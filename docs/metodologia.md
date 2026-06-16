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
  no está cargado). Implementado en `src/mcda.py`; métricas reutilizables en
  `src/metrics.py`.
- **Por qué primero.** Línea base interpretable y barata; referencia contra la cual medir
  si el ML aporta algo.
- **Anti-leakage.** Las features derivadas de D1 (`n_d1_300m`, `n_d1_500m`, `dist_d1_km`)
  se **excluyen del score** (la etiqueta `tiene_d1` es función directa de ellas). La
  etiqueta solo se usa como validación post-hoc, nunca como insumo.
- **Resultados** (ver [v1_mcda_resultados.md](v1_mcda_resultados.md)). Evaluación honesta
  sobre los 3589 hexágonos, K=200:
  - **NDCG@200 = 0.8495**, **Precision@200 = 0.820** (164 de las 200 celdas mejor
    rankeadas ya tienen D1), **top-200 hitting = 0.1866** (techo 0.2275, pues hay 879
    positivos ≫ K=200).
  - Lectura: un baseline sin ML ordena bien las celdas más parecidas a las de D1. v2/v3
    deberán superarlo —o, en v3, revelar cuánto de esto se sostiene sin leakage espacial.

### v2 — Clasificador look-alike naïve
- **Qué.** Clasificador (¿la celda "se parece" a donde hay D1?) con split train/test
  **aleatorio**.
- **Riesgo conocido de antemano.** El split aleatorio mezcla celdas vecinas entre train y
  test → **leakage por autocorrelación espacial** → métricas optimistas.
- **Resultados.** _(pendiente — se espera ver el espejismo de alto desempeño)_

### v3 — Mismo modelo con Spatial CV
- **Qué.** Idéntico modelo, pero validación con **spatial cross-validation**: se excluyen
  **buffers espaciales** alrededor de las celdas de test para que train y test no compartan
  vecindario.
- **Por qué.** Corrige el leakage de v2 y entrega una estimación **honesta** de la
  capacidad de generalización.
- **Posible mejora (del paper).** Ensemble secuencial Lasso-first + 2º modelo sobre
  residuales.
- **Resultados.** _(pendiente — se espera caída de métricas vs. v2 = evidencia del leakage)_

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

- **Mecanismo del leakage (v2).** Autocorrelación espacial: celdas H3 vecinas tienen
  features y etiqueta correlacionadas; un split aleatorio las reparte entre train y test.
- **Corrección (v3).** Spatial CV con exclusión de buffers: _(pendiente — definir radio de
  buffer en función del tamaño de celda H3 y del rango de autocorrelación observado)_.
- **Evidencia esperada.** Comparación de métricas v2 vs. v3 (NDCG, top-K). Una caída
  material al pasar a spatial CV **confirma** que v2 estaba inflado por el leakage.

| Métrica | v2 (split aleatorio) | v3 (spatial CV) |
|---|---|---|
| NDCG@K | _(pendiente)_ | _(pendiente)_ |
| top-K hitting | _(pendiente)_ | _(pendiente)_ |
| top-K loss | _(pendiente)_ | _(pendiente)_ |

---

## Referencias

- Lu, *et al.* (2024). *Retail store location screening: A machine learning-based
  approach.* Journal of Retailing and Consumer Services.
- DANE (2018). Censo Nacional de Población y Vivienda (CNPV) — Marco Geoestadístico
  Nacional (MGN). https://geoportal.dane.gov.co/
