"""
Black-Scholes analytical pricing for European vanilla options.

Formulas (call/put) and the standard Greeks (delta, gamma, vega, theta, rho).
All functions are vectorized with numpy so they accept scalars or arrays.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """Compute d1 and d2 used throughout the Black-Scholes formulas.

    S: spot price
    K: strike price
    T: time to maturity (in years)
    r: risk-free rate (annualized, continuous compounding)
    sigma: volatility (annualized)
    q: continuous dividend yield (default 0)
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Black-Scholes price of a European call or put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price


def delta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Sensitivity of the option price to a $1 move in the underlying."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    elif option_type == "put":
        return np.exp(-q * T) * (norm.cdf(d1) - 1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r, sigma, q=0.0):
    """Sensitivity of delta to a $1 move in the underlying (same for call/put)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """Sensitivity of the option price to a 1-point change in volatility.

    Returned per unit of sigma (i.e. divide by 100 for a '1 vol point' quote).
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Time decay of the option price, expressed per year (divide by 365 for per day)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    term1 = -np.exp(-q * T) * S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
        return term1 + term2 + term3
    elif option_type == "put":
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        return term1 + term2 + term3
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def rho(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Sensitivity of the option price to a 1-point change in the risk-free rate."""
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


if __name__ == "__main__":
    # Quick sanity check
    S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.2
    print(f"Call price: {bs_price(S, K, T, r, sigma, option_type='call'):.4f}")
    print(f"Put price:  {bs_price(S, K, T, r, sigma, option_type='put'):.4f}")
    print(f"Delta (call): {delta(S, K, T, r, sigma, option_type='call'):.4f}")
    print(f"Gamma: {gamma(S, K, T, r, sigma):.4f}")
    print(f"Vega: {vega(S, K, T, r, sigma):.4f}")
    print(f"Theta (call, per day): {theta(S, K, T, r, sigma, option_type='call') / 365:.4f}")
