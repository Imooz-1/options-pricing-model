"""
Monte Carlo pricing for European vanilla options under Black-Scholes dynamics.

Two estimators are provided:
- Naive Monte Carlo (plain simulation)
- Antithetic variates (variance reduction technique)

Both simulate the terminal stock price directly via the closed-form
solution of geometric Brownian motion, so there is no discretization
bias (this is exact simulation, not an Euler scheme).
"""

import numpy as np


def _terminal_price(S, T, r, sigma, q, Z):
    """Exact GBM terminal price given standard normal draws Z."""
    return S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)


def mc_price(S, K, T, r, sigma, q=0.0, option_type="call", n_paths=100_000,
             antithetic=True, seed=None):
    """Monte Carlo price of a European option.

    Returns (price, standard_error).

    If antithetic=True, uses antithetic variates: for each standard normal
    draw Z we also use -Z, which reduces variance because payoffs from
    symmetric draws are negatively correlated for monotonic payoff functions.
    """
    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal(half)
        Z_full = np.concatenate([Z, -Z])
    else:
        Z_full = rng.standard_normal(n_paths)

    ST = _terminal_price(S, T, r, sigma, q, Z_full)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoff

    if antithetic:
        # Pair up (Z, -Z) payoffs and average within pairs before computing
        # the variance, which is the correct way to estimate the standard
        # error of an antithetic estimator (pairs, not raw paths, are the
        # i.i.d. unit here).
        n = len(discounted) // 2
        pair_avg = 0.5 * (discounted[:n] + discounted[n:])
        price = pair_avg.mean()
        se = pair_avg.std(ddof=1) / np.sqrt(n)
    else:
        price = discounted.mean()
        se = discounted.std(ddof=1) / np.sqrt(len(discounted))

    return price, se


def convergence_study(S, K, T, r, sigma, q=0.0, option_type="call",
                       path_counts=None, n_repeats=30, seed=42):
    """Run MC pricing (naive and antithetic) across a range of path counts.

    Each path count is repeated `n_repeats` times with independent seeds and
    the mean absolute error is reported, which gives a much smoother and more
    representative convergence curve than a single noisy realization.

    Returns a dict of arrays: path_counts, naive_errors, anti_errors
    (error = mean |MC price - analytical BS price| across repeats).
    """
    from black_scholes import bs_price

    if path_counts is None:
        path_counts = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

    rng = np.random.default_rng(seed)
    true_price = bs_price(S, K, T, r, sigma, q, option_type)

    naive_errors, anti_errors = [], []

    for n in path_counts:
        naive_run_errors, anti_run_errors = [], []
        for _ in range(n_repeats):
            s = rng.integers(0, 1_000_000)
            p_naive, _ = mc_price(S, K, T, r, sigma, q, option_type, n_paths=n,
                                   antithetic=False, seed=s)
            p_anti, _ = mc_price(S, K, T, r, sigma, q, option_type, n_paths=n,
                                  antithetic=True, seed=s)
            naive_run_errors.append(abs(p_naive - true_price))
            anti_run_errors.append(abs(p_anti - true_price))
        naive_errors.append(np.mean(naive_run_errors))
        anti_errors.append(np.mean(anti_run_errors))

    return {
        "path_counts": np.array(path_counts),
        "true_price": true_price,
        "naive_errors": np.array(naive_errors),
        "anti_errors": np.array(anti_errors),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from black_scholes import bs_price

    S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.2

    bs = bs_price(S, K, T, r, sigma, option_type="call")
    mc_naive, se_naive = mc_price(S, K, T, r, sigma, option_type="call",
                                   n_paths=100_000, antithetic=False, seed=1)
    mc_anti, se_anti = mc_price(S, K, T, r, sigma, option_type="call",
                                 n_paths=100_000, antithetic=True, seed=1)

    print(f"Black-Scholes analytical: {bs:.4f}")
    print(f"MC naive:      {mc_naive:.4f}  (SE: {se_naive:.4f})")
    print(f"MC antithetic: {mc_anti:.4f}  (SE: {se_anti:.4f})")
    print(f"Variance reduction factor (SE naive / SE antithetic): {se_naive / se_anti:.2f}x")
