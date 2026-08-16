"""Site Selection Engine frontend — decision narrative for D1 expansion.

Tells a story in 3 beats: (1) where D1 stores are today, (2) what is
THE #1 recommendation for the next opening, (3) why that hexagon, comparing
its features against the average of the zones that already have D1.

Reads local artifacts directly (parquet + GeoJSON) — no API, simpler and
more robust for the demo. The map uses pydeck: H3HexagonLayer for the
background score, a highlighted layer for hexagon #1, and ScatterplotLayer
for the current stores.

Run locally:
    uv run streamlit run app/streamlit_app.py

Deployment: Streamlit Community Cloud (main app = app/streamlit_app.py). The
rankings/parquet/GeoJSON are versioned in git (see docs/despliegue.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Project root on sys.path, independent of the cwd streamlit is launched from.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import pydeck as pdk
import streamlit as st

from src import config

st.set_page_config(page_title="Where should D1 open its 167th store?", layout="wide")

SCORE_COL = config.SERVING_SCORE_COL["v3"]   # score_lookalike_v3
RANK_COL = "rank_lookalike_v3"

# Features for the "why" panel: (technical col, readable name, inverse_distance).
# inverse_distance=True -> closer is better.
FEATURE_SPEC: list[tuple[str, str, bool]] = [
    ("n_supermercados_500m", "Supermarkets within 500m", False),
    ("dist_supermercado_km", "Distance to nearest supermarket", True),
    ("n_farmacias_500m", "Pharmacies within 500m", False),
    ("n_colegios_500m", "Schools within 500m", False),
    ("n_paradas_bus_500m", "Bus stops within 500m", False),
    ("n_bancos_atm_500m", "Banks/ATMs within 500m", False),
    ("densidad_vial", "Road network density", False),
    ("viviendas_estimadas", "Estimated households in the zone", False),
    ("estrato_promedio", "Average socioeconomic stratum", False),
]


# --------------------------------------------------------------------------- #
# Data loading (local parquet/GeoJSON, cached)
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
    """Average of each feature over the zones that ALREADY have D1 (tiene_d1==1)."""
    cols = [c for c, _, _ in FEATURE_SPEC]
    return _features.loc[_features["tiene_d1"] == 1, cols].mean()


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def _score_to_color(scores: pd.Series, alpha: int = 120) -> list[list[int]]:
    """Light gray -> orange -> intense red ramp over the normalized [min,max] score."""
    lo, hi = float(scores.min()), float(scores.max())
    rng = (hi - lo) or 1.0
    out = []
    for s in scores:
        t = (s - lo) / rng
        if t < 0.5:  # gray [200,200,200] -> orange [255,165,0]
            u = t / 0.5
            r, g, b = 200 + 55 * u, 200 - 35 * u, 200 - 200 * u
        else:        # orange [255,165,0] -> red [220,50,50]
            u = (t - 0.5) / 0.5
            r, g, b = 255 - 35 * u, 165 - 115 * u, 50 * u
        out.append([int(r), int(g), int(b), alpha])
    return out


def _fmt(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    av = abs(v)
    if av >= 1000:
        return f"{v:,.0f}"
    if float(v).is_integer():
        return f"{int(v)}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


# --------------------------------------------------------------------------- #
# Load + validate
# --------------------------------------------------------------------------- #
ranking = load_ranking()
features = load_features()
d1_points = load_d1_points()

missing = []
if ranking.empty:
    missing.append(f"v3 ranking (`{config.LOOKALIKE_V3_RANKING_PARQUET_PATH.name}`)")
if features.empty:
    missing.append(f"features (`{config.FEATURES_PARQUET_PATH.name}`)")
if missing:
    st.error(
        "Missing artifacts to run the dashboard: " + ", ".join(missing) + ". "
        "Run the pipeline: `uv run python -m src.data.features` and "
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
# Sidebar (simplified: v3 fixed as production model)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Controls")
    st.caption("Model: **v3 — Look-alike + spatial CV** (production)")
    top_k = st.slider(
        "Colored candidate hexagons", 50, n_total, min(300, n_total), step=50,
        help="How many of the top hexagons are colored in the background. The #1 "
             "recommendation is always highlighted.",
    )
    with st.expander("Top 5 alternatives (rank 2–6)"):
        alts = (
            ranking.loc[ranking["rank"].between(2, 6),
                        ["rank", "h3_index", "score", "lat_centroid", "lon_centroid"]]
            .sort_values("rank")
        )
        st.dataframe(alts, width="stretch", hide_index=True)
    st.caption(f"Source: local parquet · {n_total} hexagons · {n_stores} D1 stores")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title(f"Where should D1 open its store number {n_stores + 1} in Bogota?")
st.caption(
    f"Look-alike model trained on {n_stores} existing D1 stores · "
    "Score = environment similarity, **not** a sales prediction."
)

# --------------------------------------------------------------------------- #
# Main layout 60 / 40
# --------------------------------------------------------------------------- #
col_map, col_panel = st.columns([3, 2], gap="medium")

# --- Map ---
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
        "🔴 #1 recommendation   🔵 Current D1 stores "
        f"({n_stores})   ░ Low score → High score ░"
    )

# --- Recommendation panel ---
with col_panel:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#c0392b,#e74c3c);
                    padding:18px 20px;border-radius:12px;color:white;">
          <div style="font-size:14px;letter-spacing:1px;opacity:.9;">🏆 #1 RECOMMENDATION</div>
          <div style="font-family:monospace;font-size:13px;margin-top:8px;opacity:.95;">
            {hex1_id}</div>
          <div style="font-size:34px;font-weight:700;margin-top:6px;line-height:1;">
            {hex1_row['score']:.3f}<span style="font-size:16px;font-weight:400;"> / 1.00</span>
          </div>
          <div style="font-size:13px;opacity:.9;margin-top:4px;">Similarity score</div>
          <div style="font-size:13px;opacity:.9;margin-top:8px;">
            Rank <b>1</b> of {n_total} candidate hexagons</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Why this hexagon?")
    st.caption(
        "Value of hexagon #1 vs. average of the zones that **already have D1**. "
        "✅ favorable · ⚠️ below the D1 pattern."
    )

    if hex1_feats is None:
        st.warning("No features were found for hexagon #1.")
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
                f"{_fmt(hv)} &nbsp;·&nbsp; D1 average: {_fmt(av)}</span>",
                unsafe_allow_html=True,
            )

    st.warning(
        "This score measures **environment similarity** to existing D1 stores, it does "
        "not predict profitability. Use it as a starting point for field analysis, not "
        "as a final decision."
    )
