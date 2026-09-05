"""
Delta-hedging simulation for a short European option position.

Idea: a trader sells one option, receives the Black-Scholes premium,
and dynamically hedges by holding `delta` shares of the underlying,
rebalanced at discrete time steps. In a perfect continuous-time world
this replication is exact (zero P&L). With discrete rebalancing, a
residual hedging error appears -- this simulation quantifies it and
shows how it shrinks as rebalancing frequency increases.
"""

import numpy as np

from black_scholes import bs_price, delta as bs_delta


def simulate_stock_paths(S0, T, r, sigma, q, n_steps, n_paths, seed=None):
    """Simulate GBM stock paths under the risk-neutral measure.

    Returns an array of shape (n_paths, n_steps + 1) including t=0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


def simulate_delta_hedge(S0, K, T, r, sigma, q=0.0, option_type="call",
                          n_steps=52, n_paths=10_000, seed=None):
    """Simulate a delta-hedged short option position.

    Returns a dict with the hedging P&L per path, plus diagnostics.

    P&L convention: the trader SELLS the option at the Black-Scholes
    premium and hedges. P&L = final hedge portfolio value - option payoff.
    A P&L of exactly 0 means perfect replication.
    """
    dt = T / n_steps
    paths = simulate_stock_paths(S0, T, r, sigma, q, n_steps, n_paths, seed)

    premium = bs_price(S0, K, T, r, sigma, q, option_type)

    # Time to maturity at each rebalancing date
    times_to_maturity = T - np.arange(n_steps + 1) * dt
    times_to_maturity[-1] = 1e-8  # avoid division by zero at maturity in delta formula

    S0_col = paths[:, 0]
    deltas = np.zeros((n_paths, n_steps + 1))
    for i in range(n_steps):  # delta at maturity (last column) is not needed for hedging
        deltas[:, i] = bs_delta(paths[:, i], K, times_to_maturity[i], r, sigma, q, option_type)

    # Initial hedge: sell option, receive premium, buy delta_0 shares,
    # remainder sits in (or is borrowed from) the cash account.
    cash = premium - deltas[:, 0] * S0_col

    for i in range(1, n_steps):
        cash = cash * np.exp(r * dt) - (deltas[:, i] - deltas[:, i - 1]) * paths[:, i]

    # Close out the hedge at maturity: sell remaining shares, settle cash.
    cash = cash * np.exp(r * dt) + deltas[:, n_steps - 1] * paths[:, n_steps]

    ST = paths[:, -1]
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    pnl = cash - payoff

    return {
        "pnl": pnl,
        "premium": premium,
        "mean_pnl": pnl.mean(),
        "std_pnl": pnl.std(),
        "paths": paths,
    }


def rebalancing_frequency_study(S0, K, T, r, sigma, q=0.0, option_type="call",
                                 steps_list=None, n_paths=5_000, seed=42):
    """Show how hedging error (std of P&L) shrinks as rebalancing frequency increases."""
    if steps_list is None:
        steps_list = [4, 12, 26, 52, 104, 252]

    stds = []
    for n_steps in steps_list:
        result = simulate_delta_hedge(S0, K, T, r, sigma, q, option_type,
                                       n_steps=n_steps, n_paths=n_paths, seed=seed)
        stds.append(result["std_pnl"])

    return {"steps_list": np.array(steps_list), "pnl_std": np.array(stds)}


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.2
    result = simulate_delta_hedge(S, K, T, r, sigma, option_type="call",
                                   n_steps=52, n_paths=20_000, seed=7)
    print(f"Premium received: {result['premium']:.4f}")
    print(f"Mean hedging P&L (weekly rebalancing): {result['mean_pnl']:.4f}")
    print(f"Std hedging P&L (weekly rebalancing):  {result['std_pnl']:.4f}")
