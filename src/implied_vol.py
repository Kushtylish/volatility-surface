import numpy as np
from scipy.optimize import brentq
from src.black_scholes import bs_price, bs_greeks


def implied_vol(market_price, S, K, T, r, option_type="call", q=0.0, tol=1e-8, max_iter=100):
    """
    Solve for implied volatility using Newton-Raphson with brentq fallback.

    Parameters
    ----------
    market_price : float - observed option price
    S, K, T, r, q : float - standard BS inputs
    option_type : 'call' or 'put'

    Returns
    -------
    float - implied volatility (annualized)
    """
    if T <= 0:
        raise ValueError("T must be positive to compute implied vol")

    # Arbitrage bounds check
    if option_type == "call":
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        upper = S * np.exp(-q * T)
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
        upper = K * np.exp(-r * T)

    if market_price < intrinsic - 1e-8:
        raise ValueError(
            f"Price {market_price} below intrinsic {intrinsic:.4f} - no valid IV"
        )
    if market_price > upper + 1e-8:
        raise ValueError(
            f"Price {market_price} above upper bound {upper:.4f} - no valid IV"
        )

    # Newton-Raphson
    sigma = 0.2  # initial guess
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type, q)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        vega = bs_greeks(S, K, T, r, sigma, option_type, q)["vega"]
        if vega < 1e-10:
            break  # fall through to brentq
        sigma = sigma - diff / vega
        if sigma <= 0:
            sigma = 1e-4  # keep positive, keep iterating

    # Fallback: brentq on [1e-6, 5.0]
    def objective(s):
        return bs_price(S, K, T, r, s, option_type, q) - market_price

    try:
        return brentq(objective, 1e-6, 5.0, xtol=1e-10, maxiter=200)
    except ValueError as e:
        raise ValueError(f"Could not bracket implied vol: {e}")