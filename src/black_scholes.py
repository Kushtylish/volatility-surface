import numpy as np
from scipy.stats import norm


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes price for a European option.

    Parameters
    ----------
    S : float - spot price
    K : float - strike
    T : float - time to expiry in years
    r : float - risk-free rate (annualized, continuous)
    sigma : float - volatility (annualized)
    option_type : 'call' or 'put'
    q : float - continuous dividend yield

    Returns
    -------
    float - option price
    """
    if T <= 0:
        # At expiry, value is intrinsic
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    return price


def bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Analytic Greeks for a European option under Black-Scholes.

    Returns dict with keys: delta, gamma, vega, theta, rho.
    Vega is per 1.0 change in sigma (not per 1%).
    Theta is per year (not per day).
    Rho is per 1.0 change in r.
    """
    if T <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def put_call_parity_gap(S, K, T, r, sigma, q=0.0):
    """
    Verify put-call parity: C - P = S*exp(-qT) - K*exp(-rT).
    Returns the residual (should be ~0).
    """
    C = bs_price(S, K, T, r, sigma, "call", q)
    P = bs_price(S, K, T, r, sigma, "put", q)
    lhs = C - P
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    return lhs - rhs