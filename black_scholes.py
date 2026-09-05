# Black-Scholes : pricing d'options vanilles europeennes + les Greeks
# Formules classiques, cf. Hull "Options, Futures and Other Derivatives" chap 15-19

import numpy as np
from scipy.stats import norm


def d1_d2(S, K, T, r, sigma, q=0.0):
    # S = spot, K = strike, T = maturite (en annees), r = taux sans risque, sigma = vol
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type doit etre 'call' ou 'put'")


def delta(S, K, T, r, sigma, q=0.0, option_type="call"):
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    elif option_type == "put":
        return np.exp(-q * T) * (norm.cdf(d1) - 1)
    else:
        raise ValueError("option_type doit etre 'call' ou 'put'")


def gamma(S, K, T, r, sigma, q=0.0):
    # gamma est le meme pour un call et un put
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    # idem, pas de distinction call/put. valeur pour 1 unite de sigma (diviser par 100 si besoin)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, option_type="call"):
    # valeur par an -> diviser par 365 pour avoir la perte de valeur "par jour"
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
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
        raise ValueError("option_type doit etre 'call' ou 'put'")


def rho(S, K, T, r, sigma, q=0.0, option_type="call"):
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type doit etre 'call' ou 'put'")


if __name__ == "__main__":
    # petit test rapide pour verifier que les formules sont coherentes
    S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.2
    print("Call price:", round(bs_price(S, K, T, r, sigma, option_type="call"), 4))
    print("Put price: ", round(bs_price(S, K, T, r, sigma, option_type="put"), 4))
    print("Delta call:", round(delta(S, K, T, r, sigma, option_type="call"), 4))
    print("Gamma:", round(gamma(S, K, T, r, sigma), 4))
    print("Vega:", round(vega(S, K, T, r, sigma), 4))
    print("Theta call (par jour):", round(theta(S, K, T, r, sigma, option_type="call") / 365, 4))
