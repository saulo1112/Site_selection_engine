"""Frontend del Site Selection Engine — narrativa de decision para expansion D1.

Cuenta una historia en 3 golpes: (1) donde estan las tiendas D1 hoy, (2) cual es
LA recomendacion #1 para la proxima apertura, (3) por que ese hexagono, comparando
sus features contra el promedio de las zonas que ya tienen D1.

Lee directamente los artefactos locales (parquet + GeoJSON) — sin API, mas simple y
robusto para la demo. El mapa usa pydeck: H3HexagonLayer para el score de fondo,
una capa destacada para el hexagono #1, y ScatterplotLayer para las tiendas actuales.

Ejecutar local:
    uv run streamlit run app/streamlit_app.py

Despliegue: Streamlit Community Cloud (app principal = app/streamlit_app.py). Los
rankings/parquet/GeoJSON van versionados en git (ver docs/despliegue.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Raiz del proyecto en sys.path, independiente del cwd desde el que se lance streamlit.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import pydeck as pdk
import streamlit as st

from src import config

st.set_page_config(page_title="¿Dónde abre D1 su tienda 167?", layout="wide")

SCORE_COL = config.SERVING_SCORE_COL["v3"]   # score_lookalike_v3
RANK_COL = "rank_lookalike_v3"

# Features del panel "por que": (col tecnica, nombre legible, distancia_inversa).
# distancia_inversa=True -> mas cerca es mejor: ✅ si esta POR DEBAJO del promedio D1.
FEATURE_SPEC: list[tuple[str, str, bool]] = [
    ("n_supermercados_500m", "Supermercados en 500m", False),
    ("dist_supermercado_km", "Distancia al supermercado mas cercano", True),
    ("n_farmacias_500m", "Farmacias en 500m", False),
    ("n_colegios_500m", "Colegios en 500m", False),
    ("n_paradas_bus_500m", "Paradas de bus en 500m", False),
    ("n_bancos_atm_500m", "Bancos/ATMs en 500m", False),
    ("densidad_vial", "Densidad de red vial", False),
    ("viviendas_estimadas", "Viviendas estimadas en la zona", False),
    ("estrato_promedio", "Estrato promedio", False),
]


# --------------------------------------------------------------------------- #
# Carga de datos (parquet/GeoJSON local, cacheada)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_ranking() -> pd.DataFrame:
    path = config.LOOKALIKE_V3_RANKING_PARQUET_PATH
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df.rename(columns={SCORE_COL: "score", RANK_COL: "rank"})


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    path = config.FEATURES_PARQUET_PATH
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path).set_index("h3_index")


@st.cache_data(show_spinner=False)
def load_d1_points() -> pd.DataFrame:
    path = config.SERVING_POI_LAYERS["d1"]
    if not path.exists():
        return pd.DataFrame()
    gj = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {"lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1]}
        for f in gj["features"] if f["geometry"]["type"] == "Point"
    ]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def d1_reference(_features: pd.DataFrame) -> pd.Series:
    """Promedio de cada feature sobre las zonas que YA tienen D1 (tiene_d1==1)."""
    cols = [c for c, _, _ in FEATURE_SPEC]
    return _features.loc[_features["tiene_d1"] == 1, cols].mean()


# --------------------------------------------------------------------------- #
# Helpers de presentacion
# --------------------------------------------------------------------------- #
def _score_to_color(scores: pd.Series, alpha: int = 120) -> list[list[int]]:
    """Rampa gris claro -> naranja -> rojo intenso por score normalizado [min,max]."""
    lo, hi = float(scores.min()), float(scores.max())
    rng = (hi - lo) or 1.0
    out = []
    for s in scores:
        t = (s - lo) / rng
        if t < 0.5:  # gris [200,200,200] -> naranja [255,165,0]
            u = t / 0.5
            r, g, b = 200 + 55 * u, 200 - 35 * u, 200 - 200 * u
        else:        # naranja [255,165,0] -> rojo [220,50,50]
            u = (t - 0.5) / 0.5
            r, g, b = 255 - 35 * u, 165 - 115 * u, 50 * u
        out.append([int(r), int(g), int(b), alpha])
    return out


def _fmt(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "s/d"
    av = abs(v)
    if av >= 1000:
        return f"{v:,.0f}"
    if float(v).is_integer():
        return f"{int(v)}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


# --------------------------------------------------------------------------- #
# Carga + validacion
# --------------------------------------------------------------------------- #
ranking = load_ranking()
features = load_features()
d1_points = load_d1_points()

missing = []
if ranking.empty:
    missing.append(f"ranking v3 (`{config.LOOKALIKE_V3_RANKING_PARQUET_PATH.name}`)")
if features.empty:
    missing.append(f"features (`{config.FEATURES_PARQUET_PATH.name}`)")
if missing:
    st.error(
        "Faltan artefactos para correr el dashboard: " + ", ".join(missing) + ". "
        "Corre el pipeline: `uv run python -m src.data.features` y "
        "`uv run python -m src.models.lookalike_v3`."
    )
    st.stop()

ref = d1_reference(features)
hex1_row = ranking.loc[ranking["rank"] == 1].iloc[0]
hex1_id = hex1_row["h3_index"]
hex1_feats = features.loc[hex1_id] if hex1_id in features.index else None
n_total = len(ranking)
n_stores = len(d1_points)

# --------------------------------------------------------------------------- #
# Sidebar (simplificado: v3 fijo de produccion)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Controles")
    st.caption("Modelo: **v3 — Look-alike + spatial CV** (produccion)")
    top_k = st.slider(
        "Hexagonos candidatos coloreados", 50, n_total, min(300, n_total), step=50,
        help="Cuantos de los mejores hexagonos se colorean de fondo. La recomendacion "
             "#1 siempre esta destacada.",
    )
    if "show_alts" not in st.session_state:
        st.session_state.show_alts = False
    if st.button("Ver Top 5 alternativas", width="stretch"):
        st.session_state.show_alts = not st.session_state.show_alts
    st.caption(f"Fuente: parquet local · {n_total} hexagonos · {n_stores} tiendas D1")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title(f"¿Donde deberia abrir D1 su tienda numero {n_stores + 1} en Bogota?")
st.caption(
    f"Modelo look-alike entrenado sobre {n_stores} tiendas D1 existentes · "
    "Score = similitud de entorno, **no** prediccion de ventas."
)

# --------------------------------------------------------------------------- #
# Layout principal 60 / 40
# --------------------------------------------------------------------------- #
col_map, col_panel = st.columns([3, 2], gap="medium")

# --- Mapa ---
with col_map:
    bg = ranking.sort_values("score", ascending=False).head(top_k).copy()
    bg["color"] = _score_to_color(bg["score"], alpha=120)
    bg_layer = pdk.Layer(
        "H3HexagonLayer", data=bg, get_hexagon="h3_index",
        get_fill_color="color", get_line_color=[120, 120, 120, 60],
        line_width_min_pixels=0.5, pickable=True, stroked=True, filled=True,
        extruded=False,
    )

    hl = ranking.loc[ranking["rank"] == 1].copy()
    hl_layer = pdk.Layer(
        "H3HexagonLayer", data=hl, get_hexagon="h3_index",
        get_fill_color=[220, 50, 50, 255], get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=3, pickable=True, stroked=True, filled=True,
        extruded=False,
    )

    layers = [bg_layer, hl_layer]
    if not d1_points.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=d1_points, get_position="[lon, lat]",
            get_fill_color=[30, 100, 220, 200], get_radius=80,
            radius_min_pixels=2, radius_max_pixels=8, pickable=False,
        ))

    tooltip = {
        "html": "<b>Rank:</b> {rank}<br/><b>Score:</b> {score}<br/><b>h3:</b> {h3_index}",
        "style": {"backgroundColor": "#1b1b1b", "color": "white"},
    }
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=float(hex1_row["lat_centroid"]),
            longitude=float(hex1_row["lon_centroid"]),
            zoom=13, pitch=0,
        ),
        map_style="road",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, width="stretch")
    st.caption(
        "🔴 Recomendacion #1   🔵 Tiendas D1 actuales "
        f"({n_stores})   ░ Score bajo → Score alto ░"
    )

    if st.session_state.show_alts:
        st.markdown("**Top 5 alternativas (rank 2–6)**")
        alts = (
            ranking.loc[ranking["rank"].between(2, 6),
                        ["rank", "h3_index", "score", "lat_centroid", "lon_centroid"]]
            .sort_values("rank")
        )
        st.dataframe(alts, width="stretch", hide_index=True)

# --- Panel de recomendacion ---
with col_panel:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#c0392b,#e74c3c);
                    padding:18px 20px;border-radius:12px;color:white;">
          <div style="font-size:14px;letter-spacing:1px;opacity:.9;">🏆 RECOMENDACION #1</div>
          <div style="font-family:monospace;font-size:13px;margin-top:8px;opacity:.95;">
            {hex1_id}</div>
          <div style="font-size:34px;font-weight:700;margin-top:6px;line-height:1;">
            {hex1_row['score']:.3f}<span style="font-size:16px;font-weight:400;"> / 1.00</span>
          </div>
          <div style="font-size:13px;opacity:.9;margin-top:4px;">Score de similitud</div>
          <div style="font-size:13px;opacity:.9;margin-top:8px;">
            Rank <b>1</b> de {n_total} hexagonos candidatos</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### ¿Por que este hexagono?")
    st.caption(
        "Valor del hexagono #1 vs. promedio de las zonas que **ya tienen D1**. "
        "✅ favorable · ⚠️ por debajo del patron D1."
    )

    if hex1_feats is None:
        st.warning("No se encontraron features para el hexagono #1.")
    else:
        for tech, label, inverse in FEATURE_SPEC:
            hv = hex1_feats.get(tech)
            av = ref.get(tech)
            if hv is None or av is None or pd.isna(hv) or pd.isna(av):
                emoji = "•"
            else:
                better = (hv < av) if inverse else (hv > av)
                emoji = "✅" if better else "⚠️"
            st.markdown(
                f"{emoji}&nbsp; **{label}**  \n"
                f"<span style='color:#888'>"
                f"{_fmt(hv)} &nbsp;·&nbsp; promedio D1: {_fmt(av)}</span>",
                unsafe_allow_html=True,
            )

    st.warning(
        "Este score mide **similitud de entorno** con tiendas D1 existentes, no predice "
        "rentabilidad. Usar como punto de partida para analisis de campo, no como "
        "decision final."
    )
