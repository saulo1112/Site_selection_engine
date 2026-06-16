# Resultados v2 — Clasificador look-alike (Regresion Logistica)

_Generado por `src/models/lookalike.py`. Total de hexagonos: **3589**._

## Clases a predecir

Clasificacion **binaria** sobre la etiqueta `tiene_d1` (calculada en la ETAPA 4):

- **Clase 1 (positiva, ~24.5%):** el hexagono YA tiene >=1 tienda D1 a <=300m ("celda tipo-D1", un sitio que D1 ya eligio).
- **Clase 0 (negativa):** el hexagono no tiene D1 cercano.

El modelo estima `P(clase=1)` a partir de las features **no-D1** y ese probabilistico es el **score look-alike** con el que se rankean los hexagonos.

## Predictores (anti-leakage)

Se usan **7** features: `n_supermercados_500m`, `dist_supermercado_km`, `n_farmacias_500m`, `n_colegios_500m`, `n_paradas_bus_500m`, `n_bancos_atm_500m`, `densidad_vial`.

Se **excluyen** las derivadas de D1 (`n_d1_300m`, `n_d1_500m`, `dist_d1_km`): la etiqueta es funcion directa de ellas, usarlas seria leakage tautologico.
Las demograficas (`poblacion_estimada`, `viviendas_estimadas`) no estan disponibles (censo DANE no cargado, ver `src/data/load_censo.py`) y quedan fuera.

## Particion (v2 = split aleatorio, naive)

Split **aleatorio estratificado** 75/25 (`random_state=42`). **Advertencia:** el split aleatorio reparte hexagonos vecinos (espacialmente autocorrelacionados) entre train y test, por lo que las metricas probablemente esten **infladas por leakage espacial**. v3 lo corrige con spatial CV y compara (ver docs/metodologia.md §6).

## Diagnostico de clasificacion (conjunto de test)

- **ROC-AUC**: 0.7801
- **PR-AUC** (average precision, mas honesta con clases desbalanceadas): 0.5970
- Test: 898 hexagonos, 24.5% positivos.

**Matriz de confusion** (umbral 0.5; filas = real, columnas = predicho):

| | pred 0 | pred 1 |
|---|---|---|
| **real 0** | 472 | 206 |
| **real 1** | 69 | 151 |

**Reporte por clase** (precision / recall / F1):

```
                   precision    recall  f1-score   support

 clase_0 (sin D1)     0.8725    0.6962    0.7744       678
clase_1 (tipo-D1)     0.4230    0.6864    0.5234       220

         accuracy                         0.6938       898
        macro avg     0.6477    0.6913    0.6489       898
     weighted avg     0.7623    0.6938    0.7129       898
```

## Interpretabilidad — coeficientes de la LR

Sobre features estandarizadas (comparables entre si). Signo = direccion del efecto sobre `P(tipo-D1)`; magnitud = importancia relativa.

| feature              |    coef |   abs_coef |
|:---------------------|--------:|-----------:|
| dist_supermercado_km | -1.09   |     1.09   |
| n_farmacias_500m     |  0.3775 |     0.3775 |
| n_supermercados_500m |  0.345  |     0.345  |
| n_colegios_500m      |  0.2697 |     0.2697 |
| n_paradas_bus_500m   |  0.1982 |     0.1982 |
| n_bancos_atm_500m    |  0.0781 |     0.0781 |
| densidad_vial        | -0.0156 |     0.0156 |

## Metricas de ranking (sobre todo el grid)

- **NDCG@200**: 0.8349
- **Precision@200**: 0.8050
- **top-200 hitting**: 0.1832 / **loss**: 0.8168

## Comparacion v1 (MCDA) vs v2 (LR)

| Metrica | v1 MCDA | v2 LR |
|---|---|---|
| NDCG@200 | 0.8033 | 0.8349 |
| Precision@200 | 0.7650 | 0.8050 |
| top-200 hitting | 0.1741 | 0.1832 |

> **Lectura honesta:** si v2 no supera materialmente a v1, es un resultado valido: el MCDA ya captura casi toda la senal lineal disponible. Y recordar que cualquier ventaja de v2 aqui puede ser, en parte, leakage espacial -> v3 dira cuanto se sostiene.

## Limitacion

Problema **positive-unlabeled**: las negativas incluyen buenos sitios donde D1 aun no llega. El score es de **similitud / prioridad de exploracion**, no de desempeno; hereda el supuesto de que la estrategia de localizacion de D1 es buena (docs/metodologia.md §5).
