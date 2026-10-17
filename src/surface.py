import numpy as np
import pandas as pd
from pathlib import Path
from src.iv_grid import combined_smile, filter_smile, filter_by_price, time_to_expiry_years

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TICKER = "SPY"


def build_surface(expiries, spot, r, ticker=TICKER, min_price=0.05, iv_min=0.02, iv_max=1.5):
    """
    Stack per-expiry smiles into a single long-format DataFrame with columns:
        expiry, T_years, strike, iv, side
    """
    frames = []
    for expiry in expiries:
        s = combined_smile(expiry, spot, r, ticker=ticker)
        if s.empty:
            print(f"Skipping {expiry}: no usable quotes")
            continue
        s = filter_smile(s, iv_min=iv_min, iv_max=iv_max)
        s = filter_by_price(s, min_price=min_price)
        if s.empty:
            print(f"Skipping {expiry}: empty after filtering")
            continue
        s = s.copy()
        s["expiry"] = expiry
        s["T_years"] = time_to_expiry_years(expiry)
        frames.append(s[["expiry", "T_years", "strike", "iv", "side"]])
        print(f"{expiry}: {len(s)} strikes, T={s['T_years'].iloc[0]:.3f}y")
    if not frames:
        return pd.DataFrame()
    surf = pd.concat(frames, ignore_index=True)
    surf = surf.sort_values(["T_years", "strike"]).reset_index(drop=True)
    return surf


def smile_family_plot(surface, spot, save_path=None):
    """2D: one smile curve per expiry (shows smile + term structure together)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    expiries = sorted(surface["expiry"].unique())
    cmap = plt.cm.viridis
    for i, exp in enumerate(expiries):
        s = surface[surface["expiry"] == exp]
        color = cmap(i / max(len(expiries) - 1, 1))
        ax.plot(s["strike"], s["iv"], "o-", markersize=3, color=color, label=exp)
    ax.axvline(spot, color="red", linestyle="--", alpha=0.6, label=f"Spot {spot:.2f}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility")
    ax.set_title("SPY volatility smile: term structure family")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()
    return fig


def surface_3d_plot(surface, spot, save_path=None):
    """3D trisurf of IV across (strike, T)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    x = surface["strike"].values
    y = surface["T_years"].values * 365.0
    z = surface["iv"].values

    if len(surface) > 3:
        ax.plot_trisurf(x, y, z, cmap="viridis", alpha=0.85, linewidth=0.2)
    else:
        ax.scatter(x, y, z, c="blue")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Days to expiry")
    ax.set_zlabel("Implied vol")
    ax.set_title("SPY implied volatility surface")
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()
    return fig