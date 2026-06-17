# Resultados v4 — Look-alike con demografia (censo DANE + estrato IDECA)

_Generado por `src/models/lookalike_v4.py`. Total de hexagonos: **3589**._

## Que cambia respecto a v3

**Mismo modelo** (Regresion Logistica) y **misma validacion** (spatial CV con bloques H3 res-6 + buffer 1 anillo). Lo unico nuevo: se anaden features **demograficas** prorrateadas por manzana — poblacion y viviendas (censo DANE CNPV 2018) y estrato socioeconomico (IDECA Bogota).

> **AVISO:** no se detectaron features demograficas con datos en `features.parquet` (censo/estrato no cargados). v4 degenera a v3. Carga las capas (`src/data/load_censo.py`, `src/data/load_estrato.py`), recalcula features y vuelve a correr v4 para una comparacion real.

## Aporte de la demografia — BASE (sin demo, = v3) vs FULL (con demo, = v4)

Ambos evaluados con **identico** esquema de spatial CV (predicciones OOF). Asi el delta aisla el aporte de la demografia, no del metodo de validacion.

- BASE  (7 preds): `n_supermercados_500m`, `dist_supermercado_km`, `n_farmacias_500m`, `n_colegios_500m`, `n_paradas_bus_500m`, `n_bancos_atm_500m`, `densidad_vial`
- FULL  (7 preds): `n_supermercados_500m`, `dist_supermercado_km`, `n_farmacias_500m`, `n_colegios_500m`, `n_paradas_bus_500m`, `n_bancos_atm_500m`, `densidad_vial`

| Metrica | BASE (v3) | FULL (v4) | Δ (v4 - v3) |
|---|---|---|---|
| ROC-AUC (OOF) | 0.7934 | 0.7934 | +0.0000 |
| PR-AUC (OOF) | 0.5899 | 0.5899 | +0.0000 |
| NDCG@200 | 0.8400 | 0.8400 | +0.0000 |
| Precision@200 | 0.8150 | 0.8150 | +0.0000 |
| top-200 hitting | 0.1854 | 0.1854 | +0.0000 |

> **Veredicto:** sin demografia cargada, v4 == v3 (no hay nada que comparar).

## Diagnostico honesto v4 (predicciones OOF)

- **ROC-AUC**: 0.7934 | **PR-AUC**: 0.5899
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

## Interpretabilidad — coeficientes del modelo final (full-data)

Sobre features estandarizadas (comparables). Signo = direccion del efecto sobre `P(tipo-D1)`; magnitud = importancia relativa. El ranking de produccion usa el modelo full-data; las metricas de arriba vienen de las OOF.

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

Sigue siendo un score de **similitud / prioridad de exploracion** (no de desempeno), con el supuesto look-alike de que la localizacion de D1 es buena (docs/metodologia.md §5). La demografia anade contexto de mercado, no convierte el score en una prediccion de ventas.
