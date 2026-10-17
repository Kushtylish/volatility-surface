import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
from src.implied_vol import implied_vol

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TICKER = "SPY"


def load_chain_file(expiry, side, ticker=TICKER):
    """Load one saved chain file (calls or puts). Returns DataFrame or None."""
    path = DATA_DIR / f"{ticker}_{expiry}_{side}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def clean_chain(df):
    """
    Filter out unusable quotes:
    - require bid > 0 (real market makers quoting)
    - require ask > bid (not crossed)
    - require positive open interest
    - require strike > 0
    """
    d = df.copy()
    d = d[(d["bid"] > 0) & (d["ask"] > d["bid"])]
    d = d[d["openInterest"] > 0]
    d = d[d["strike"] > 0]
    return d.reset_index(drop=True)


def time_to_expiry_years(expiry_str):
    """Calendar days to expiry divided by 365."""
    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    today = date.today()
    days = (expiry - today).days
    return max(days, 0) / 365.0


def compute_iv_grid(expiry, spot, r, side="calls", ticker=TICKER, price_col="mid"):
    """
    Compute IV for each strike in the chain.

    price_col:
      'mid'  -> (bid + ask) / 2
      'last' -> lastPrice
    """
    df = load_chain_file(expiry, side, ticker)
    if df is None:
        return pd.DataFrame()

    df = clean_chain(df)
    if df.empty:
        return pd.DataFrame()

    T = time_to_expiry_years(expiry)
    if T <= 0:
        return pd.DataFrame()

    if price_col == "mid":
        prices = ((df["bid"] + df["ask"]) / 2).values
    else:
        prices = df["lastPrice"].values

    option_type = "call" if side == "calls" else "put"
    strikes = df["strike"].values

    ivs = []
    for price, K in zip(prices, strikes):
        try:
            iv = implied_vol(float(price), spot, float(K), T, r, option_type)
        except (ValueError, ZeroDivisionError):
            iv = np.nan
        ivs.append(iv)

    out = pd.DataFrame({
        "strike": strikes,
        "price": prices,
        "iv": ivs,
        "volume": df["volume"].values,
        "openInterest": df["openInterest"].values,
        "T": T,
        "expiry": expiry,
        "side": side,
    })
    out = out.dropna(subset=["iv"])
    return out