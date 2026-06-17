"""Frontend del Site Selection Engine — mapa interactivo de hexagonos look-alike.

Arquitectura HIBRIDA: consume la API FastAPI (src/api) si `API_BASE_URL` esta definido;
si no responde (o no esta configurado), cae a leer los artefactos locales directamente
(robusto para la demo). El mapa usa pydeck H3HexagonLayer, que renderiza los hexagonos
nativamente desde el `h3_index`.

Ejecutar local:
    uv run streamlit run app/streamlit_app.py

Despliegue: Streamlit Community Cloud (app principal = app/streamlit_app.py).
Configurar el secret API_BASE_URL con la URL publica de la API.
"""

from __future__ import annotations

import os

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

st.set_page_config(page_title="Site Selection Engine — Bogota", layout="wide")

# API_BASE_URL desde secret de Streamlit o variable de entorno (vacio -> fallback local).
# st.secrets lanza si no existe secrets.toml, por eso se protege con try/except.
def _get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


API_BASE_URL = _get_secret("API_BASE_URL") or os.environ.get("API_BASE_URL", "")

MODEL_LABELS = {
    "v4": "v4 — Look-alike + demografia (DANE/IDECA)",
    "v3": "v3 — Look-alike + spatial CV",
    "v2": "v2 — Look-alike (split aleatorio)",
    "mcda": "v1 — MCDA (sin ML)",
}


# --------------------------------------------------------------------------- #
# Carga de datos: API primero, fallback a parquet local
# --------------------------------------------------------------------------- #
def _api_get(path: str, params: dict | None = None, timeout: float = 8.0):
    resp = requests.get(f"{API_BASE_URL.rstrip('/')}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300, show_spinner=False)
def get_available_models() -> tuple[list[str], str]:
    """(modelos disponibles, fuente)."""
    if API_BASE_URL:
        try:
            data = _api_get("/models")
            return data["available"], "api"
        except requests.RequestException:
            pass
    # Fallback local
    from src import config
    avail = [m for m, p in config.SERVING_RANKINGS.items() if p.exists()]
    return avail, "local"


@st.cache_data(ttl=300, show_spinner=False)
def load_ranking(model: str, source: str) -> pd.DataFrame:
    """Ranking normalizado: h3_index, lat_centroid, lon_centroid, score, rank, tiene_d1."""
    if source == "api":
        data = _api_get("/hexes", {"model": model})
        return pd.DataFrame(data["items"])
    # Fallback local (mismo normalizador que la API)
    from src.api import service
    return service.get_ranking(model)


def _score_to_color(scores: pd.Series) -> list[list[int]]:
    """Rampa amarillo->rojo (RGBA) por score normalizado [min,max]."""
    lo, hi = float(scores.min()), float(scores.max())
    rng = (hi - lo) or 1.0
    colors = []
    for s in scores:
        t = (s - lo) / rng
        r = int(255 * (0.2 + 0.8 * t))
        g = int(255 * (0.9 - 0.7 * t))
        b = int(60 * (1 - t))
        colors.append([r, g, b, 170])
    return colors


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("🗺️ Site Selection Engine — Bogota")
st.caption(
    "Ranking de hexagonos H3 por **similitud look-alike a Tiendas D1**. "
    "El score es `P(tipo-D1)`: prioridad de exploracion, **no** prediccion de ventas."
)

models, source = get_available_models()
if not models:
    st.error(
        "No hay rankings disponibles. Corre el pipeline de modelos "
        "(`uv run python -m src.models.lookalike_v3` / `lookalike_v4`) o configura "
        "`API_BASE_URL`."
    )
    st.stop()

with st.sidebar:
    st.header("Controles")
    st.caption(f"Fuente de datos: **{'API' if source == 'api' else 'parquet local'}**"
               + (f" ({API_BASE_URL})" if source == "api" else ""))
    default_idx = models.index("v4") if "v4" in models else 0
    model = st.selectbox(
        "Modelo", models, index=default_idx,
        format_func=lambda m: MODEL_LABELS.get(m, m),
    )
    df = load_ranking(model, source)

    top_k = st.slider("Top-K hexagonos a mostrar", 50, len(df), min(300, len(df)), step=50)
    smin, smax = float(df["score"].min()), float(df["score"].max())
    min_score = st.slider("Score minimo", smin, smax, smin, step=(smax - smin) / 100 or 0.01)
    show_labels = st.checkbox("Resaltar hexagonos con D1 actual", value=False)

# Filtro
view = df[df["score"] >= min_score].sort_values("score", ascending=False).head(top_k).copy()
view["color"] = _score_to_color(view["score"])
if show_labels and "tiene_d1" in view.columns:
    # Borde/tinte para los que ya tienen D1 (verdad de terreno).
    view["color"] = [
        [40, 120, 255, 200] if (td == 1) else c
        for c, td in zip(view["color"], view["tiene_d1"].fillna(0))
    ]

# Mapa
col_map, col_info = st.columns([3, 1])
with col_map:
    layer = pdk.Layer(
        "H3HexagonLayer",
        data=view,
        get_hexagon="h3_index",
        get_fill_color="color",
        get_line_color=[60, 60, 60, 80],
        line_width_min_pixels=1,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
    )
    tooltip = {
        "html": "<b>Rank:</b> {rank}<br/><b>Score:</b> {score}<br/>"
                "<b>h3:</b> {h3_index}",
        "style": {"backgroundColor": "#1b1b1b", "color": "white"},
    }
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=4.65, longitude=-74.10, zoom=10.5, pitch=0),
        map_style="road",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, width="stretch")

with col_info:
    st.metric("Hexagonos mostrados", len(view))
    if "tiene_d1" in view.columns:
        pos = int(view["tiene_d1"].fillna(0).sum())
        st.metric(f"De ellos con D1 actual", pos)
        st.caption(f"Precision en la vista: {pos / len(view):.1%}" if len(view) else "")
    st.caption(
        "Azul = ya tiene D1 (si el resaltado esta activo). Amarillo→rojo = score "
        "look-alike creciente."
    )

st.subheader("Top hexagonos")
st.dataframe(
    view[["rank", "h3_index", "lat_centroid", "lon_centroid", "score"]
         + (["tiene_d1"] if "tiene_d1" in view.columns else [])].head(50),
    width="stretch", hide_index=True,
)
