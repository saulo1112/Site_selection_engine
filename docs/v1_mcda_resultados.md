# Resultados v1 — MCDA baseline

_Generado por `src/models/mcda.py`. Total de hexagonos: **3589**._

## Que es esta version

Score ponderado e **interpretable**, sin ML: normalizacion min-max por feature y combinacion lineal con pesos a priori definidos por razonamiento de negocio (no ajustados a la etiqueta). Es la **linea base** contra la cual se mediran v2 (clasificador look-alike) y v3 (spatial CV).

## Anti-leakage

Las features derivadas de la ubicacion de D1 (`n_d1_300m`, `n_d1_500m`, `dist_d1_km`) se **excluyen del score**: la etiqueta `tiene_d1` se define como `n_d1_300m >= 1`, por lo que usarlas seria leakage tautologico. La etiqueta se usa **solo despues**, como validacion honesta (metricas de ranking abajo), nunca como insumo del score.

> **Nota:** las features demograficas (`poblacion_estimada`, `viviendas_estimadas`) no estan disponibles en esta corrida (censo DANE no cargado, ver `src/data/load_censo.py`). Su peso de grupo se redistribuyo proporcionalmente entre los grupos presentes.

## Pesos efectivos por grupo

| Grupo | Peso efectivo |
|---|---|
| competencia | 0.4118 |
| complementarios | 0.4118 |
| accesibilidad_vial | 0.1765 |

## Pesos por feature

| Feature | Grupo | Peso |
|---|---|---|
| `n_supermercados_500m` | competencia | 0.2059 |
| `dist_supermercado_km` | competencia | 0.2059 |
| `n_farmacias_500m` | complementarios | 0.1029 |
| `n_colegios_500m` | complementarios | 0.1029 |
| `n_paradas_bus_500m` | complementarios | 0.1029 |
| `n_bancos_atm_500m` | complementarios | 0.1029 |
| `densidad_vial` | accesibilidad_vial | 0.1765 |

## Evaluacion honesta post-hoc (ranking vs. `tiene_d1`)

Estas metricas **no** se usaron para elegir pesos; solo miden, a posteriori, que tan bien el ranking MCDA recupera las celdas donde D1 ya esta presente.

- **NDCG@200**: 0.8033 (calidad del orden en el top-K; 1.0 = perfecto).
- **Precision@200**: 0.7650 (153 de las 200 celdas mejor rankeadas ya tienen D1).
- **top-200 hitting** (recall de positivos en el top-K): 0.1741 (techo = 0.2275, porque hay 879 positivos > K=200: ni un ranking perfecto puede capturarlos todos en solo K celdas).
- **top-200 loss** (positivos fuera del top-K): 0.8259
- Positivos en el top-200: **153** de 879 positivos totales (3589 hexagonos).

> **Lectura honesta:** el `hitting` parece bajo solo porque hay muchos mas positivos (879) que celdas seleccionadas (K=200). La metrica mas informativa aqui es **Precision@200 = 0.765** y el **NDCG@200 = 0.803**: un baseline sin ML que ordena bien las celdas mas parecidas a las de D1. v2/v3 deberan superarlo (o, en v3, revelar cuanto de esto era leakage espacial).

## Limitacion

El score es de **similitud / prioridad de exploracion**, no de desempeno: mide parecido a las celdas donde D1 ya opera, no ventas esperadas. Hereda el supuesto look-alike de que la estrategia de localizacion de D1 es buena (ver docs/metodologia.md §5).
