# Resumen de la tabla de features

_Generado por `src/data/features.py`. Total de hexagonos: **3589**._

## Balance de la etiqueta `tiene_d1`

- Positivos (tiene_d1=1): **879**
- Negativos (tiene_d1=0): **2710**
- Ratio positivos/negativos: **0.3244** (24.49% positivos)

> **Nota de modelado:** dataset desbalanceado. Estrategias a considerar en v2/v3: `class_weight='balanced'`, metricas de ranking (NDCG, top-K) en vez de accuracy, y umbral calibrado. La separacion espacial (spatial CV, v3) reducira aun mas los positivos efectivos.

> **Nota de leakage (critica):** la etiqueta `tiene_d1` se define como `n_d1_300m >= 1`. Por lo tanto las features derivadas de la ubicacion de D1 (`n_d1_300m`, `n_d1_500m`, `dist_d1_km`) son funciones directas de la etiqueta y **NO deben usarse como predictores** en el modelo look-alike (target leakage): su alta correlacion con `tiene_d1` es tautologica, no informativa. El modelo debe aprender de las features de competidores, complementarios, red vial y demografia. Esto es independiente del leakage espacial por autocorrelacion, que se aborda con spatial CV en v3.

> **Nota de competencia (no-D1):** `n_supermercados_500m` y `dist_supermercado_km` miden solo competidores **distintos de D1** (`es_d1 = 0`). Incluir a D1 introduciria leakage: todo positivo tendria un 'supermercado' (el propio D1) a <=300m. Ver docs/metodologia.md §6.

> **Correlacion residual `dist_supermercado_km` <-> `dist_d1_km`**: **0.7653**. Se interpreta como co-localizacion real (zonas con comercio denso tienden a tener tanto D1 como otros supermercados cerca), no como leakage: ya se excluyo a D1 de `dist_supermercado_km` (nota anterior). Ver docs/metodologia.md §6.1.

## Estadisticas descriptivas por feature

|                      |   count |     mean |      std |   min |      25% |      50% |       75% |        max |   pct_nulos |
|:---------------------|--------:|---------:|---------:|------:|---------:|---------:|----------:|-----------:|------------:|
| n_d1_300m            |    3589 |   0.3179 |   0.6331 |     0 |   0      |   0      |    0      |     5      |         0   |
| n_d1_500m            |    3589 |   0.6264 |   0.9628 |     0 |   0      |   0      |    1      |     6      |         0   |
| dist_d1_km           |    3589 |   0.9882 |   1.0178 |     0 |   0.3085 |   0.6565 |    1.289  |     5.9153 |         0   |
| n_supermercados_500m |    3589 |   1.1449 |   1.5826 |     0 |   0      |   1      |    2      |    11      |         0   |
| dist_supermercado_km |    3589 |   0.7697 |   1.0694 |     0 |   0.1911 |   0.4533 |    0.8756 |     7.4769 |         0   |
| n_farmacias_500m     |    3589 |   5.3293 |   6.2297 |     0 |   0      |   3      |    9      |    42      |         0   |
| n_colegios_500m      |    3589 |   5.482  |   5.6144 |     0 |   2      |   4      |    7      |    50      |         0   |
| n_paradas_bus_500m   |    3589 |  11.1329 |   8.965  |     0 |   4      |  10      |   16      |    48      |         0   |
| n_bancos_atm_500m    |    3589 |   4.4179 |   7.1617 |     0 |   0      |   1      |    6      |    57      |         0   |
| densidad_vial        |    3589 |   0.0274 |   0.0186 |     0 |   0.013  |   0.0262 |    0.0395 |     0.0855 |         0   |
| viviendas_estimadas  |    3589 | 672.202  | 583.033  |     0 | 155.487  | 613.103  | 1032.42   | 10107.5    |         0   |
| estrato_promedio     |    3205 |   2.9795 |   1.2313 |     1 |   2      |   3      |    3.6649 |     6      |        10.7 |

## Correlacion de cada feature con `tiene_d1`

|                      |   corr_con_tiene_d1 |
|:---------------------|--------------------:|
| n_d1_300m            |              0.8818 |
| n_d1_500m            |              0.6996 |
| n_supermercados_500m |              0.4367 |
| n_farmacias_500m     |              0.4101 |
| n_bancos_atm_500m    |              0.3738 |
| n_paradas_bus_500m   |              0.3061 |
| estrato_promedio     |              0.2722 |
| n_colegios_500m      |              0.246  |
| viviendas_estimadas  |              0.1081 |
| densidad_vial        |              0.0889 |
| dist_supermercado_km |             -0.2619 |
| dist_d1_km           |             -0.4792 |

## Features demograficas — cobertura

Censo DANE (CNPV/MGN 2018, poblacion/viviendas) + estrato IDECA, prorrateados por area de interseccion manzana<->hexagono. Las manzanas no cubren todo el grid (zonas no residenciales/rurales; estrato 0 = sin estrato, tratado como nulo), por lo que parte de los hexagonos queda **sin dato** (NULL). El modelo v4 imputa la **mediana** para esos casos en vez de descartarlos.

_Total de hexagonos: **3589**._

| Feature | Hex con dato | % con dato | % NULL |
|---|---|---|---|
| `viviendas_estimadas` | 3589 | 100.0% | 0.0% |
| `estrato_promedio` | 3205 | 89.3% | 10.7% |

> **Hipotesis look-alike (a verificar en v4):** D1 es hard-discount con foco en estratos bajos -> se espera que `estrato_promedio` tenga relacion **negativa** con `tiene_d1` (a menor estrato, mas probable presencia de D1). El coeficiente de la LR en v4 lo confirmara o no, honestamente.
