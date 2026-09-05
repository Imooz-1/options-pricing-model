# Simulation de couverture delta-neutre pour une position courte sur option
#
# Idee : je vends une option, j'encaisse la prime BS, et je couvre en detenant
# `delta` actions du sous-jacent, rebalance a intervalles reguliers. En continu
# la replication est parfaite (PnL nul). En discret il reste une erreur -
# c'est ca qu'on regarde ici, et comment elle diminue quand on rebalance plus souvent.

import numpy as np

from black_scholes import bs_price, delta as bs_delta


def simulate_stock_paths(S0, T, r, sigma, q, n_steps, n_paths, seed=None):
    # simulation GBM sous la mesure risque-neutre, retourne (n_paths, n_steps+1)
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
    """
    PnL = valeur finale du portefeuille de couverture - payoff de l'option.
    PnL = 0 -> replication parfaite.
    """
    dt = T / n_steps
    paths = simulate_stock_paths(S0, T, r, sigma, q, n_steps, n_paths, seed)

    premium = bs_price(S0, K, T, r, sigma, q, option_type)

    times_to_maturity = T - np.arange(n_steps + 1) * dt
    times_to_maturity[-1] = 1e-8  # evite division par zero dans la formule du delta

    S0_col = paths[:, 0]
    deltas = np.zeros((n_paths, n_steps + 1))
    for i in range(n_steps):
        deltas[:, i] = bs_delta(paths[:, i], K, times_to_maturity[i], r, sigma, q, option_type)

    # t=0 : on vend l'option, on achete delta_0 actions, le reste va sur le compte cash
    cash = premium - deltas[:, 0] * S0_col

    for i in range(1, n_steps):
        cash = cash * np.exp(r * dt) - (deltas[:, i] - deltas[:, i - 1]) * paths[:, i]

    # a maturite on solde la position
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
    # montre que l'ecart-type du PnL baisse quand on rebalance plus souvent
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
    print("Prime encaissee:", round(result["premium"], 4))
    print("PnL moyen (rebalancement hebdo):", round(result["mean_pnl"], 4))
    print("Ecart-type du PnL (rebalancement hebdo):", round(result["std_pnl"], 4))
