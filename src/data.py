import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TICKER = "SPY"


def get_spot(ticker=TICKER):
    """Last close price of the underlying."""
    t = yf.Ticker(ticker)
    hist = t.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"Could not fetch history for {ticker}")
    return float(hist["Close"].iloc[-1])


def get_expiries(ticker=TICKER):
    """List of available option expiry dates as strings 'YYYY-MM-DD'."""
    t = yf.Ticker(ticker)
    return list(t.options)


def get_chain(expiry, ticker=TICKER):
    """Return (calls, puts) DataFrames for a given expiry."""
    t = yf.Ticker(ticker)
    chain = t.option_chain(expiry)
    return chain.calls.copy(), chain.puts.copy()


def save_chain(expiry, ticker=TICKER):
    """Save calls and puts for one expiry to data/, return the DataFrames.
    Returns (None, None) if the chain is empty (e.g., front-dated expiry)."""
    DATA_DIR.mkdir(exist_ok=True)
    calls, puts = get_chain(expiry, ticker)
    if calls.empty or puts.empty:
        print(f"Skipping {expiry} (empty chain)")
        return None, None
    calls.to_csv(DATA_DIR / f"{ticker}_{expiry}_calls.csv", index=False)
    puts.to_csv(DATA_DIR / f"{ticker}_{expiry}_puts.csv", index=False)
    return calls, puts