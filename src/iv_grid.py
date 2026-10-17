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
def combined_smile(expiry, spot, r, ticker=TICKER):
    """
    Build the full smile: puts for strikes below spot, calls for strikes above.
    Near-the-money strikes use whichever side has tighter spread.
    """
    puts = compute_iv_grid(expiry, spot, r, side="puts", ticker=ticker)
    calls = compute_iv_grid(expiry, spot, r, side="calls", ticker=ticker)

    if puts.empty and calls.empty:
        return pd.DataFrame()

    # OTM puts below spot
    puts_otm = puts[puts["strike"] < spot] if not puts.empty else pd.DataFrame()

    # OTM calls above spot
    calls_otm = calls[calls["strike"] > spot] if not calls.empty else pd.DataFrame()

    # ATM: pick whichever side has smaller distance to spot
    all_atm = []
    if not puts.empty:
        p = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]]
        all_atm.append(p)
    if not calls.empty:
        c = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
        all_atm.append(c)
    atm = pd.concat(all_atm).drop_duplicates(subset=["strike"]) if all_atm else pd.DataFrame()

    combined = pd.concat([puts_otm, atm, calls_otm]).drop_duplicates(subset=["strike"]).sort_values("strike")
    return combined.reset_index(drop=True)
def filter_smile(smile, iv_min=0.02, iv_max=1.5, max_rel_spread=0.5):
    """
    Drop extreme IVs and wide-spread quotes that pollute the surface.
    """
    s = smile.copy()
    s = s[(s["iv"] >= iv_min) & (s["iv"] <= iv_max)]
    return s.reset_index(drop=True)
def filter_by_price(smile, min_price=0.05):
    """Drop strikes whose mid price is too small to trust (tick-size noise)."""
    return smile[smile["price"] >= min_price].reset_index(drop=True)