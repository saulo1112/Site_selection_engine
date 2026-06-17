"""Chequeo de disponibilidad de datos para elegir la ciudad de estudio.

Replica el rigor del chequeo de AOI del proyecto EUDR: en vez de decidir "a ojo",
mide la senal disponible con datos reales.

Por cada ciudad candidata (Bogota, Cali, Medellin, Barranquilla):
  1. Resuelve su frontera administrativa OSM (relation id via Nominatim, con
     fallback hardcoded en config) -> transparencia: se loguea el id usado.
  2. Cuenta via Overpass:
       - tiendas D1   (shop=supermarket + brand=D1)        -> senal look-alike
       - tiendas D1 por nombre (verificacion cruzada)
       - tiendas Ara  (respaldo si D1 es escaso)
       - total shop=* (densidad general de etiquetado OSM)
  3. Calcula metricas derivadas (ratio D1/shops, viabilidad de positivos).
  4. Guarda respuestas crudas en data/raw/ y escribe una tabla comparativa en
     docs/seleccion_area_estudio.md (entre marcadores autogenerados).

La verificacion de fuentes DANE/estrato es documental (se hace por web y se
registra a mano en docs/seleccion_area_estudio.md), no programatica: el CNPV/MGN
cubre las 4 ciudades de forma uniforme a nivel manzana, asi que no discrimina.

Uso:
    uv run python -m src.selection.data_availability_check
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from src import config


# --------------------------------------------------------------------------- #
# Utilidades de red
# --------------------------------------------------------------------------- #
def resolve_relation_id(city: str, spec: dict) -> int:
    """Resuelve el OSM relation id de una ciudad via Nominatim.

    Usa el fallback de config si Nominatim falla o no devuelve una relacion.
    """
    params = {
        "q": spec["nominatim_query"],
        "format": "jsonv2",
        "limit": 1,
        "polygon_geojson": 0,
    }
    headers = {"User-Agent": config.USER_AGENT}
    try:
        resp = requests.get(
            config.NOMINATIM_ENDPOINT,
            params=params,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        results = resp.json()
        for r in results:
            if r.get("osm_type") == "relation":
                rid = int(r["osm_id"])
                print(f"  [{city}] Nominatim -> relation {rid} ({r.get('display_name', '')[:60]})")
                return rid
        print(f"  [{city}] Nominatim no devolvio relacion; uso fallback {spec['osm_relation_id']}")
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"  [{city}] Nominatim fallo ({exc}); uso fallback {spec['osm_relation_id']}")
    return int(spec["osm_relation_id"])


def overpass_count(query_body: str) -> int:
    """Ejecuta una consulta Overpass `out count;` y devuelve el conteo total.

    Rota endpoints y reintenta con backoff exponencial ante errores transitorios.
    """
    full_query = f"[out:json][timeout:{config.REQUEST_TIMEOUT}];({query_body});out count;"
    headers = {"User-Agent": config.USER_AGENT}

    last_exc: Exception | None = None
    for attempt in range(config.MAX_RETRIES):
        endpoint = config.OVERPASS_ENDPOINTS[attempt % len(config.OVERPASS_ENDPOINTS)]
        try:
            resp = requests.post(
                endpoint,
                data={"data": full_query},
                headers=headers,
                timeout=config.REQUEST_TIMEOUT + 30,
            )
            resp.raise_for_status()
            payload = resp.json()
            # `out count;` devuelve un elemento type=count con tags.total
            for el in payload.get("elements", []):
                if el.get("type") == "count":
                    return int(el["tags"]["total"])
            # Fallback: si no hay elemento count, contar elementos.
            return len(payload.get("elements", []))
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_exc = exc
            wait = config.BACKOFF_BASE * (2 ** attempt)
            print(f"    intento {attempt + 1}/{config.MAX_RETRIES} fallo en {endpoint} "
                  f"({type(exc).__name__}); reintento en {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Overpass fallo tras {config.MAX_RETRIES} intentos: {last_exc}")


# --------------------------------------------------------------------------- #
# Chequeo por ciudad
# --------------------------------------------------------------------------- #
def check_city(city: str, spec: dict) -> dict:
    print(f"\n=== {city} ===")
    relation_id = resolve_relation_id(city, spec)
    area_id = 3_600_000_000 + relation_id

    counts: dict[str, int] = {}
    raw: dict[str, object] = {"city": city, "relation_id": relation_id, "area_id": area_id}

    for key, template in config.OVERPASS_QUERIES.items():
        body = template.format(area_id=area_id)
        count = overpass_count(body)
        counts[key] = count
        print(f"  {key:14s}: {count}")
        time.sleep(config.SLEEP_BETWEEN_QUERIES)

    raw["counts"] = counts
    raw["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()

    # Guardar crudo
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_RAW / f"overpass_{city.lower()}.json"
    out_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    d1 = counts["d1"]
    shops = counts["shops_total"]
    return {
        "Ciudad": city,
        "relation_id": relation_id,
        "D1 (brand)": d1,
        "D1 (por nombre)": counts["d1_by_name"],
        "Ara": counts["ara"],
        "Total shop=*": shops,
        "Ratio D1/shops (%)": round(100 * d1 / shops, 2) if shops else 0.0,
        "Positivos viables": "Si" if d1 >= config.MIN_D1_VIABLE else "Riesgo",
    }


# --------------------------------------------------------------------------- #
# Escritura del informe
# --------------------------------------------------------------------------- #
MARKER_START = "<!-- AUTO-GENERATED:OVERPASS_TABLE:START -->"
MARKER_END = "<!-- AUTO-GENERATED:OVERPASS_TABLE:END -->"


def write_table_to_doc(df: pd.DataFrame) -> None:
    """Inserta/actualiza la tabla comparativa entre marcadores en el doc."""
    config.DOCS.mkdir(parents=True, exist_ok=True)
    doc_path = config.DOCS / "seleccion_area_estudio.md"

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    table_md = df.to_markdown(index=False)
    block = (
        f"{MARKER_START}\n"
        f"\n_Generado por `src/selection/data_availability_check.py` el {stamp}._\n\n"
        f"{table_md}\n\n"
        f"_Umbral de positivos viables: D1 >= {config.MIN_D1_VIABLE} "
        f"(margen para separacion espacial en v3)._\n"
        f"{MARKER_END}"
    )

    if doc_path.exists() and MARKER_START in doc_path.read_text(encoding="utf-8"):
        text = doc_path.read_text(encoding="utf-8")
        pre = text.split(MARKER_START)[0]
        post = text.split(MARKER_END)[1] if MARKER_END in text else ""
        doc_path.write_text(pre + block + post, encoding="utf-8")
    else:
        # Si el doc aun no tiene el bloque, solo se escribe la tabla;
        # la narrativa la completa el autor alrededor de los marcadores.
        header = "# Seleccion del area de estudio\n\n"
        doc_path.write_text(header + block + "\n", encoding="utf-8")
    print(f"\nTabla escrita en {doc_path}")


def main() -> None:
    rows = []
    for city, spec in config.CITIES.items():
        rows.append(check_city(city, spec))
        time.sleep(config.SLEEP_BETWEEN_CITIES)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    write_table_to_doc(df)


if __name__ == "__main__":
    main()
