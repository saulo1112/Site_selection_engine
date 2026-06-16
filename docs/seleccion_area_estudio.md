# Selección del área de estudio

> Replica el enfoque del chequeo de AOI del proyecto EUDR: la ciudad de estudio se
> decide con datos reales, no a ojo. Se mide la señal disponible en dos frentes:
> (1) densidad de la marca de referencia (Tiendas D1) en OpenStreetMap, y (2)
> disponibilidad de datos demográficos/socioeconómicos del DANE y portales
> municipales.

## 1. Objetivo del chequeo

El modelo look-alike (v2/v3) necesita, por ciudad candidata, suficientes hexágonos
positivos (con presencia de D1) para que, tras aplicar **spatial CV** (exclusión de
buffers espaciales en v3), queden positivos viables en el conjunto de entrenamiento y
prueba. Si una ciudad tiene pocas tiendas D1 etiquetadas, el split espacial puede dejar
una muestra demasiado pequeña o desbalanceada. Por eso el chequeo mide la señal D1
**antes** de comprometerse con una ciudad.

## 2. Metodología

- **Delimitación**: cada ciudad se delimita por su **frontera administrativa en OSM**
  (relación del municipio/distrito), no por bounding box, para obtener un conteo
  honesto dentro de los límites urbanos reales y evitar spillover de municipios
  vecinos (p. ej. Soledad en el área metropolitana de Barranquilla).
- **Resolución del área**: el relation id de cada ciudad se resuelve en runtime vía
  Nominatim (con un id de respaldo hardcoded en `src/config.py` si Nominatim falla).
  El script registra el id efectivamente usado en cada corrida.
- **Consultas Overpass** (`out count;`) por ciudad:
  - `D1 (brand)`: `shop=supermarket` + `brand` que coincide con D1/Tiendas D1.
  - `D1 (por nombre)`: verificación cruzada por `name~"D1"` (captura tiendas D1 mal
    etiquetadas sin el tag `brand`).
  - `Ara`: `shop=supermarket` + `brand=Ara`, como respaldo si D1 fuera escaso.
  - `Total shop=*`: todos los POIs de tipo tienda, como proxy de densidad general de
    etiquetado OSM en la ciudad (para no confundir "poco D1" con "poco mapeado").
- **Fuentes DANE/estrato**: verificadas manualmente vía búsqueda web (no programático,
  porque el censo no tiene una API de conteo equivalente a Overpass) — ver §4.

Script: `src/data_availability_check.py` (`uv run python -m src.data_availability_check`).
Respuestas crudas de Overpass: `data/raw/overpass_<ciudad>.json`.

## 3. Resultados — señal D1 en OpenStreetMap

<!-- AUTO-GENERATED:OVERPASS_TABLE:START -->

_Generado por `src/data_availability_check.py` el 2026-06-15 23:16 UTC._

| Ciudad       |   relation_id |   D1 (brand) |   D1 (por nombre) |   Ara |   Total shop=* |   Ratio D1/shops (%) | Positivos viables   |
|:-------------|--------------:|-------------:|------------------:|------:|---------------:|---------------------:|:--------------------|
| Bogota       |       7426387 |          129 |               164 |    45 |          21569 |                 0.6  | Si                  |
| Cali         |       1320707 |            6 |                29 |     2 |           1278 |                 0.47 | Riesgo              |
| Medellin     |       1343264 |           38 |                76 |     2 |           2706 |                 1.4  | Riesgo              |
| Barranquilla |       1335179 |           12 |                16 |    35 |            811 |                 1.48 | Riesgo              |

_Umbral de positivos viables: D1 >= 40 (margen para separacion espacial en v3)._
<!-- AUTO-GENERATED:OVERPASS_TABLE:END -->

**Lectura honesta del resultado.** La hipótesis inicial era que Bogotá y Cali podrían
quedar "comparables" en señal D1, en cuyo caso se priorizaría Cali por capacidad de
validación cualitativa local. **Esa hipótesis no se sostuvo**: Bogotá domina por un
orden de magnitud (129 tiendas D1 por `brand` vs. 6 en Cali; incluso contando por
nombre, 164 vs. 29). Cali además tiene un total de `shop=*` bajo (1278 frente a 21569
en Bogotá), lo que sugiere una cobertura de etiquetado OSM más pobre en general, no
solo para D1. Medellín (38) queda justo por debajo del umbral de positivos viables, y
Barranquilla (12) muy por debajo. **Bogotá es la única ciudad con señal D1 suficiente
para garantizar positivos viables tras separación espacial.**

## 4. Disponibilidad de datos DANE y estrato socioeconómico

El Censo Nacional de Población y Vivienda (CNPV 2018) y el Marco Geoestadístico
Nacional (MGN) del DANE cubren las **cuatro ciudades de forma uniforme** a nivel de
manzana/sector censal, vía el geoportal:
`https://geoportal.dane.gov.co/geovisores/sociedad/cnpv2018-detallado/`. Variables
disponibles: población, viviendas, hogares — pero **no estrato** (el estrato
socioeconómico no es una variable del censo; proviene de la estratificación
municipal, que sí varía en disponibilidad y vigencia por ciudad).

| Ciudad | Estrato — fuente | Granularidad | Observación |
|---|---|---|---|
| **Bogotá** | IDECA — datasets "Estratificación Manzana Bogotá D.C." y "Estrato Socioeconómico" (`ideca.gov.co`, `datosabiertos.bogota.gov.co`) | Manzana | Mejor documentado y más descargable de las 4 |
| Cali | IDESC / Datos Abiertos Cali (`datos.cali.gov.co`) — "Estratificación socioeconómica según estrato, por barrio" | Barrio (2015) | Más agregado (barrio, no manzana) y menos reciente |
| Medellín | GeoMedellín / MEData (`medata.gov.co`) — capa de estrato socioeconómico + estadísticas por manzana | Manzana/predio | Buena granularidad, portal menos estandarizado que IDECA |
| Barranquilla | "Panorama Urbano" (certificado de estratificación, alcaldía) | Predio (vía trámite) | No se confirmó dataset abierto descargable a nivel manzana |

Dado que el CNPV/MGN no discrimina entre ciudades, **el diferenciador real en
disponibilidad de datos demográficos es el estrato**, y ahí Bogotá (IDECA) también
queda mejor posicionada que las demás.

## 5. Recomendación

**Ciudad de estudio: Bogotá.**

Justificación:
1. **Señal D1 suficiente** — única ciudad con D1 ≥ 40 tiendas (129), garantizando
   positivos viables en el clasificador look-alike incluso después de excluir buffers
   espaciales en v3. Las demás ciudades (Cali: 6, Medellín: 38, Barranquilla: 12)
   quedan en riesgo de tener muestra insuficiente o desbalanceada tras spatial CV.
2. **Mejor disponibilidad de estrato** — IDECA ofrece el dataset de estratificación
   más granular (manzana) y mejor documentado de las cuatro ciudades.
3. La cláusula de desempate por validación cualitativa local (Cali) **no aplica**:
   los números no quedaron comparables, así que no hay desempate que resolver. Se
   documenta esto como hallazgo, no como decisión forzada.

**Limitación a documentar.** Bogotá tiene también el mayor total de `shop=*`
(21569), por lo que el ratio D1/shops (0.60%) no es el más alto de las cuatro
ciudades — Barranquilla (1.48%) y Medellín (1.40%) tienen mayor *proporción* de D1
sobre el total de comercio mapeado. Sin embargo, para el clasificador look-alike el
factor limitante es el **conteo absoluto** de positivos disponibles para entrenar y
validar con spatial CV, no la proporción relativa — por eso prevalece Bogotá.
