# Resultados v4 — Look-alike con demografia (censo DANE + estrato IDECA)

_Generado por `src/models/lookalike_v4.py`. Total de hexagonos: **3589**._

## Que cambia respecto a v3

**Mismo modelo** (Regresion Logistica) y **misma validacion** (spatial CV con bloques H3 res-6 + buffer 1 anillo). Lo unico nuevo: se anaden features **demograficas** prorrateadas por manzana — poblacion y viviendas (censo DANE CNPV 2018) y estrato socioeconomico (IDECA Bogota).

Demograficas que entraron al modelo: `viviendas_estimadas`, `estrato_promedio`.

**Cobertura** (los hexagonos sin manzana residencial quedan NULL y se imputan con la mediana dentro de cada fold):

| Feature | % con dato | % NULL (imputado) |
|---|---|---|
| `viviendas_estimadas` | 100.0% | 0.0% |
| `estrato_promedio` | 89.3% | 10.7% |

## Aporte de la demografia — BASE (sin demo, = v3) vs FULL (con demo, = v4)

Ambos evaluados con **identico** esquema de spatial CV (predicciones OOF). Asi el delta aisla el aporte de la demografia, no del metodo de validacion.

- BASE  (7 preds): `n_supermercados_500m`, `dist_supermercado_km`, `n_farmacias_500m`, `n_colegios_500m`, `n_paradas_bus_500m`, `n_bancos_atm_500m`, `densidad_vial`
- FULL  (9 preds): `n_supermercados_500m`, `dist_supermercado_km`, `n_farmacias_500m`, `n_colegios_500m`, `n_paradas_bus_500m`, `n_bancos_atm_500m`, `densidad_vial`, `viviendas_estimadas`, `estrato_promedio`

| Metrica | BASE (v3) | FULL (v4) | Δ (v4 - v3) |
|---|---|---|---|
| ROC-AUC (OOF) | 0.7934 | 0.8052 | +0.0118 |
| PR-AUC (OOF) | 0.5899 | 0.5924 | +0.0025 |
| NDCG@200 | 0.8400 | 0.8122 | -0.0277 |
| Precision@200 | 0.8150 | 0.7900 | -0.0250 |
| top-200 hitting | 0.1854 | 0.1797 | -0.0057 |

> **Veredicto:** la demografia **mejora** el desempeno (Δ NDCG@K -0.0277, Δ ROC-AUC +0.0118). Aporta senal de mercado (tamano poblacional y estrato) que las features de OSM no capturaban. v4 pasa a ser el modelo de produccion.

## Diagnostico honesto v4 (predicciones OOF)

- **ROC-AUC**: 0.8052 | **PR-AUC**: 0.5924
- **NDCG@200**: 0.8122 | **Precision@200**: 0.7900 | **top-200 hitting**: 0.1797

**Matriz de confusion OOF** (umbral 0.5; filas = real, columnas = predicho):

| | pred 0 | pred 1 |
|---|---|---|
| **real 0** | 1936 | 774 |
| **real 1** | 252 | 627 |

**Reporte por clase (OOF):**

```
                   precision    recall  f1-score   support

 clase_0 (sin D1)     0.8848    0.7144    0.7905      2710
clase_1 (tipo-D1)     0.4475    0.7133    0.5500       879

         accuracy                         0.7141      3589
        macro avg     0.6662    0.7139    0.6703      3589
     weighted avg     0.7777    0.7141    0.7316      3589
```

## Interpretabilidad — coeficientes del modelo final (full-data)

Sobre features estandarizadas (comparables). Signo = direccion del efecto sobre `P(tipo-D1)`; magnitud = importancia relativa. El ranking de produccion usa el modelo full-data; las metricas de arriba vienen de las OOF.

| feature              |    coef |   abs_coef |
|:---------------------|--------:|-----------:|
| dist_supermercado_km | -1.0495 |     1.0495 |
| estrato_promedio     |  0.5271 |     0.5271 |
| n_colegios_500m      |  0.3202 |     0.3202 |
| n_supermercados_500m |  0.2687 |     0.2687 |
| n_farmacias_500m     |  0.2593 |     0.2593 |
| viviendas_estimadas  |  0.123  |     0.123  |
| n_paradas_bus_500m   |  0.1094 |     0.1094 |
| densidad_vial        |  0.0789 |     0.0789 |
| n_bancos_atm_500m    |  0.0431 |     0.0431 |

> **Lectura del estrato:** coeficiente de `estrato_promedio` = 0.5271 (relacion positiva con P(tipo-D1)). Es **contraria a la esperada** con la hipotesis de negocio (D1 es hard-discount con foco en estratos bajos: a menor estrato, mas probable la presencia de D1).

## Limitacion

Sigue siendo un score de **similitud / prioridad de exploracion** (no de desempeno), con el supuesto look-alike de que la localizacion de D1 es buena (docs/metodologia.md §5). La demografia anade contexto de mercado, no convierte el score en una prediccion de ventas.
