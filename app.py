import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from src.data import get_expiries
from src.iv_grid import combined_smile, filter_smile, filter_by_price, time_to_expiry_years

st.set_page_config(layout="wide", page_title="SPY Volatility Surface")
st.title("SPY Implied Volatility Surface")
st.caption("Drag to rotate · scroll to zoom · hover for values")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_FILE = RESULTS_DIR / "surface_data.csv"


def compute_surface(spot, r, min_price, iv_min, iv_max, max_expiries):
    """Compute surface from live yfinance data. Falls back to empty if network fails."""
    try:
        expiries = get_expiries()[:max_expiries]
    except Exception as e:
        st.warning(f"Could not fetch expiries from Yahoo: {e}")
        return pd.DataFrame()

    frames = []
    for exp in expiries:
        try:
            s = combined_smile(exp, spot, r)
        except Exception:
            continue
        if s.empty:
            continue
        s = filter_smile(s, iv_min=iv_min, iv_max=iv_max)
        s = filter_by_price(s, min_price=min_price)
        if s.empty:
            continue
        s = s.copy()
        s["expiry"] = exp
        s["T_years"] = time_to_expiry_years(exp)
        frames.append(s[["expiry", "T_years", "strike", "iv", "side"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def load_cached():
    """Load the precomputed surface shipped with the repo."""
    if CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE)
    return pd.DataFrame()


with st.sidebar:
    st.header("Parameters")
    spot = st.number_input("Spot price", value=763.0, step=1.0)
    r = st.slider("Risk-free rate", 0.0, 0.10, 0.045, 0.005)
    min_price = st.slider("Min option price ($)", 0.0, 1.0, 0.05, 0.01)
    iv_min = st.slider("Min IV", 0.0, 0.2, 0.02, 0.01)
    iv_max = st.slider("Max IV", 0.5, 3.0, 1.5, 0.1)

    st.divider()
    if st.button("🔄 Refresh from Yahoo (slow, may fail)"):
        with st.spinner("Fetching live data..."):
            fresh = compute_surface(spot, r, min_price, iv_min, iv_max, 6)
            if not fresh.empty:
                fresh.to_csv(CACHE_FILE, index=False)
                st.success(f"Refreshed with {len(fresh)} points")
                st.rerun()
            else:
                st.error("Fetch failed. Using cached data.")

# Default: use cached CSV
surf = load_cached()

# Apply current filters to cached data
if not surf.empty:
    surf = surf[(surf["iv"] >= iv_min) & (surf["iv"] <= iv_max)]
    surf = surf.reset_index(drop=True)

if surf.empty:
    st.error(
        "No surface data available. Either the cache file is missing, "
        "or you need to click Refresh in the sidebar (network required)."
    )
else:
    st.write(f"**{len(surf)} strikes** across **{surf['expiry'].nunique()} expiries**")
    st.caption("Data source: precomputed cache (results/surface_data.csv)")

    fig3d = go.Figure(data=[
        go.Mesh3d(
            x=surf["strike"],
            y=surf["T_years"] * 365,
            z=surf["iv"],
            intensity=surf["iv"],
            colorscale="Viridis",
            opacity=0.9,
        )
    ])
    fig3d.update_layout(
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Days to expiry",
            zaxis_title="Implied vol",
        ),
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.subheader("Smile family by expiry")
    fig2d = go.Figure()
    for exp in sorted(surf["expiry"].unique()):
        s = surf[surf["expiry"] == exp]
        fig2d.add_trace(go.Scatter(
            x=s["strike"], y=s["iv"], mode="lines+markers", name=str(exp),
        ))
    fig2d.add_vline(x=spot, line_dash="dash", line_color="red",
                    annotation_text=f"Spot {spot:.0f}")
    fig2d.update_layout(
        xaxis_title="Strike",
        yaxis_title="Implied vol",
        height=450,
    )
    st.plotly_chart(fig2d, use_container_width=True)