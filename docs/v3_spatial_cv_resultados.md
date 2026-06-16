# Resultados v3 — Clasificador look-alike con Spatial CV

_Generado por `src/models/lookalike_v3.py`. Total de hexagonos: **3589**._

## Que cambia respecto a v2

**Mismo modelo** (Regresion Logistica), **misma definicion de clases** (clase 1 = celda con D1 a <=300m). Lo unico que cambia es la **validacion**: en vez de un split aleatorio, validacion cruzada **espacial**. Cada hexagono se predice *out-of-fold* (OOF) por un modelo que no vio su vecindario -> estimacion honesta de como generaliza el modelo a zonas nuevas de la ciudad.

## Esquema de spatial CV (anti-leakage espacial)

- **Bloques espaciales**: padre H3 a resolucion **6** -> 24 bloques (~36 km2 c/u). Bloques enteros van a train o test.
- **Folds**: StratifiedGroupKFold, **5** folds (respeta bloques, balancea positivos).
- **Buffer**: se excluyen del train las celdas a <=**1** anillo(s) H3 de cualquier celda de test.
- **Predictores** (sin leakage): `n_supermercados_500m`, `dist_supermercado_km`, `n_farmacias_500m`, `n_colegios_500m`, `n_paradas_bus_500m`, `n_bancos_atm_500m`, `densidad_vial`.

**Tamanos por fold:**

| Fold | Train | Test | Test pos. | Buffer removidas |
|---|---|---|---|---|
| 1 | 3018 | 461 | 102 | 110 |
| 2 | 3068 | 448 | 113 | 73 |
| 3 | 2159 | 1226 | 273 | 204 |
| 4 | 2450 | 956 | 245 | 183 |
| 5 | 2986 | 498 | 146 | 105 |

## Diagnostico honesto (predicciones OOF)

- **ROC-AUC**: 0.7934
- **PR-AUC**: 0.5899
- **NDCG@200**: 0.8400 | **Precision@200**: 0.8150 | **top-200 hitting**: 0.1854

**Matriz de confusion OOF** (umbral 0.5; filas = real, columnas = predicho):

| | pred 0 | pred 1 |
|---|---|---|
| **real 0** | 1908 | 802 |
| **real 1** | 248 | 631 |

**Reporte por clase (OOF):**

```
                   precision    recall  f1-score   support

 clase_0 (sin D1)     0.8850    0.7041    0.7842      2710
clase_1 (tipo-D1)     0.4403    0.7179    0.5458       879

         accuracy                         0.7074      3589
        macro avg     0.6627    0.7110    0.6650      3589
     weighted avg     0.7761    0.7074    0.7258      3589
```

## Veredicto de leakage — v2 (split aleatorio) vs v3 (spatial CV)

| Metrica | v2 aleatorio | v3 spatial CV | Δ (v3 - v2) |
|---|---|---|---|
| ROC-AUC (test/OOF) | 0.7801 | 0.7934 | +0.0132 |
| PR-AUC (test/OOF) | 0.5970 | 0.5899 | -0.0071 |
| NDCG@200 | 0.8349 | 0.8400 | +0.0051 |
| top-200 hitting | 0.1832 | 0.1854 | +0.0023 |

> **Veredicto:** el desempeno **se mantiene** bajo spatial CV (Δ NDCG@K +0.0051, Δ ROC-AUC +0.0132; v3 incluso iguala o supera levemente a v2). El leakage por autocorrelacion espacial resulto **menor de lo esperado**: con un modelo lineal sobre features de buffer (campos espaciales suaves), un split aleatorio y uno espacial generalizan parecido. Es un hallazgo valido y honesto — la senal no-D1 (competencia/complementarios/vial) se sostiene en zonas no vistas, no era un espejismo del split.

## Interpretabilidad — coeficientes del modelo final (full-data)

El ranking de produccion usa un modelo reentrenado con **todos** los datos; el desempeno reportado arriba viene de las OOF (no de este ajuste full-data).

| feature              |    coef |   abs_coef |
|:---------------------|--------:|-----------:|
| dist_supermercado_km | -1.1292 |     1.1292 |
| n_farmacias_500m     |  0.3443 |     0.3443 |
| n_supermercados_500m |  0.34   |     0.34   |
| n_colegios_500m      |  0.2136 |     0.2136 |
| n_paradas_bus_500m   |  0.1884 |     0.1884 |
| n_bancos_atm_500m    |  0.1133 |     0.1133 |
| densidad_vial        | -0.0386 |     0.0386 |

## Limitacion

Sigue siendo un score de **similitud / prioridad de exploracion** (no de desempeno), con el supuesto look-alike de que la localizacion de D1 es buena (docs/metodologia.md §5). La spatial CV corrige el leakage espacial, no los sesgos del proxy ni del etiquetado OSM.
