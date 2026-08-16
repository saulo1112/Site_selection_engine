"""Data availability check for choosing the study city.

Replicates the rigor of the EUDR project's AOI check: instead of deciding "by eye",
it measures the available signal with real data.

For each candidate city (Bogota, Cali, Medellin, Barranquilla):
  1. Resolves its administrative OSM boundary (relation id via Nominatim, with a
     hardcoded fallback in config) -> transparency: the id used is logged.
  2. Counts via Overpass:
       - D1 stores   (shop=supermarket + brand=D1)        -> look-alike signal
       - D1 stores by name (cross-check)
       - Ara stores  (backup if D1 is scarce)
       - total shop=* (general density of OSM tagging)
  3. Computes derived metrics (D1/shops ratio, viability of positives).
  4. Saves raw responses to data/raw/ and writes a comparison table in
     docs/seleccion_area_estudio.md (between auto-generated markers).

Verification of DANE/stratum sources is documentary (done via the web and recorded
by hand in docs/seleccion_area_estudio.md), not programmatic: CNPV/MGN covers all 4
cities uniformly at the city-block level, so it isn't a discriminating factor.

Usage:
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
# Network utilities
# --------------------------------------------------------------------------- #
def resolve_relation_id(city: str, spec: dict) -> int:
    """Resolves a city's OSM relation id via Nominatim.

    Uses the config fallback if Nominatim fails or doesn't return a relation.
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
        print(f"  [{city}] Nominatim did not return a relation; using fallback {spec['osm_relation_id']}")
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"  [{city}] Nominatim failed ({exc}); using fallback {spec['osm_relation_id']}")
    return int(spec["osm_relation_id"])


def overpass_count(query_body: str) -> int:
    """Runs an Overpass `out count;` query and returns the total count.

    Rotates endpoints and retries with exponential backoff on transient errors.
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
            # `out count;` returns a type=count element with tags.total
            for el in payload.get("elements", []):
                if el.get("type") == "count":
                    return int(el["tags"]["total"])
            # Fallback: if there is no count element, count the elements.
            return len(payload.get("elements", []))
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_exc = exc
            wait = config.BACKOFF_BASE * (2 ** attempt)
            print(f"    attempt {attempt + 1}/{config.MAX_RETRIES} failed on {endpoint} "
                  f"({type(exc).__name__}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Overpass failed after {config.MAX_RETRIES} attempts: {last_exc}")


# --------------------------------------------------------------------------- #
# Per-city check
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

    # Save raw data
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_RAW / f"overpass_{city.lower()}.json"
    out_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    d1 = counts["d1"]
    shops = counts["shops_total"]
    return {
        "City": city,
        "relation_id": relation_id,
        "D1 (brand)": d1,
        "D1 (by name)": counts["d1_by_name"],
        "Ara": counts["ara"],
        "Total shop=*": shops,
        "D1/shops ratio (%)": round(100 * d1 / shops, 2) if shops else 0.0,
        "Viable positives": "Yes" if d1 >= config.MIN_D1_VIABLE else "At risk",
    }


# --------------------------------------------------------------------------- #
# Report writing
# --------------------------------------------------------------------------- #
MARKER_START = "<!-- AUTO-GENERATED:OVERPASS_TABLE:START -->"
MARKER_END = "<!-- AUTO-GENERATED:OVERPASS_TABLE:END -->"


def write_table_to_doc(df: pd.DataFrame) -> None:
    """Inserts/updates the comparison table between markers in the doc."""
    config.DOCS.mkdir(parents=True, exist_ok=True)
    doc_path = config.DOCS / "seleccion_area_estudio.md"

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    table_md = df.to_markdown(index=False)
    block = (
        f"{MARKER_START}\n"
        f"\n_Generated by `src/selection/data_availability_check.py` on {stamp}._\n\n"
        f"{table_md}\n\n"
        f"_Viable-positives threshold: D1 >= {config.MIN_D1_VIABLE} "
        f"(margin for spatial separation in v3)._\n"
        f"{MARKER_END}"
    )

    if doc_path.exists() and MARKER_START in doc_path.read_text(encoding="utf-8"):
        text = doc_path.read_text(encoding="utf-8")
        pre = text.split(MARKER_START)[0]
        post = text.split(MARKER_END)[1] if MARKER_END in text else ""
        doc_path.write_text(pre + block + post, encoding="utf-8")
    else:
        # If the doc doesn't have the block yet, only the table is written;
        # the author fills in the narrative around the markers.
        header = "# Study area selection\n\n"
        doc_path.write_text(header + block + "\n", encoding="utf-8")
    print(f"\nTable written to {doc_path}")


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
