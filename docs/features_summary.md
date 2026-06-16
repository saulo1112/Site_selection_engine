# Resumen de la tabla de features

_Generado por `src/features.py`. Total de hexagonos: **3589**._

## Balance de la etiqueta `tiene_d1`

- Positivos (tiene_d1=1): **879**
- Negativos (tiene_d1=0): **2710**
- Ratio positivos/negativos: **0.3244** (24.49% positivos)

> **Nota de modelado:** dataset desbalanceado. Estrategias a considerar en v2/v3: `class_weight='balanced'`, metricas de ranking (NDCG, top-K) en vez de accuracy, y umbral calibrado. La separacion espacial (spatial CV, v3) reducira aun mas los positivos efectivos.

> **Nota de leakage (critica):** la etiqueta `tiene_d1` se define como `n_d1_300m >= 1`. Por lo tanto las features derivadas de la ubicacion de D1 (`n_d1_300m`, `n_d1_500m`, `dist_d1_km`) son funciones directas de la etiqueta y **NO deben usarse como predictores** en el modelo look-alike (target leakage): su alta correlacion con `tiene_d1` es tautologica, no informativa. El modelo debe aprender de las features de competidores, complementarios, red vial y demografia. Esto es independiente del leakage espacial por autocorrelacion, que se aborda con spatial CV en v3.

## Estadisticas descriptivas por feature

|                      |   count |    mean |    std |   min |    25% |     50% |     75% |     max |   pct_nulos |
|:---------------------|--------:|--------:|-------:|------:|-------:|--------:|--------:|--------:|------------:|
| n_d1_300m            |    3589 |  0.3179 | 0.6331 |     0 | 0      |  0      |  0      |  5      |           0 |
| n_d1_500m            |    3589 |  0.6264 | 0.9628 |     0 | 0      |  0      |  1      |  6      |           0 |
| dist_d1_km           |    3589 |  0.9882 | 1.0178 |     0 | 0.3085 |  0.6565 |  1.289  |  5.9153 |           0 |
| n_supermercados_500m |    3589 |  1.7712 | 2.2796 |     0 | 0      |  1      |  3      | 14      |           0 |
| dist_supermercado_km |    3589 |  0.6232 | 0.8697 |     0 | 0.1349 |  0.359  |  0.7136 |  5.9153 |           0 |
| n_farmacias_500m     |    3589 |  5.3293 | 6.2297 |     0 | 0      |  3      |  9      | 42      |           0 |
| n_colegios_500m      |    3589 |  5.482  | 5.6144 |     0 | 2      |  4      |  7      | 50      |           0 |
| n_paradas_bus_500m   |    3589 | 11.1329 | 8.965  |     0 | 4      | 10      | 16      | 48      |           0 |
| n_bancos_atm_500m    |    3589 |  4.4179 | 7.1617 |     0 | 0      |  1      |  6      | 57      |           0 |
| densidad_vial        |    3589 |  0.0274 | 0.0186 |     0 | 0.013  |  0.0262 |  0.0395 |  0.0855 |           0 |

## Correlacion de cada feature con `tiene_d1`

|                      |   corr_con_tiene_d1 |
|:---------------------|--------------------:|
| n_d1_300m            |              0.8818 |
| n_d1_500m            |              0.6996 |
| n_supermercados_500m |              0.5987 |
| n_farmacias_500m     |              0.4101 |
| n_bancos_atm_500m    |              0.3738 |
| n_paradas_bus_500m   |              0.3061 |
| n_colegios_500m      |              0.246  |
| densidad_vial        |              0.0889 |
| dist_supermercado_km |             -0.3445 |
| dist_d1_km           |             -0.4792 |

## Features demograficas
_No disponibles en esta corrida: la tabla `manzanas_censo` no estaba cargada. Ver `src/load_censo.py` para habilitarlas._
