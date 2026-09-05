# Pricing Monte Carlo pour options europeennes, avec et sans reduction de variance
# On simule directement le prix terminal (solution exacte du GBM, pas de discretisation)

import numpy as np


def terminal_price(S, T, r, sigma, q, Z):
    return S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)


def mc_price(S, K, T, r, sigma, q=0.0, option_type="call", n_paths=100_000,
             antithetic=True, seed=None):
    """
    Retourne (prix, erreur standard).
    antithetic=True -> pour chaque tirage Z on utilise aussi -Z (variables antithetiques),
    ca reduit la variance de l'estimateur sans rien couter de plus.
    """
    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal(half)
        Z_full = np.concatenate([Z, -Z])
    else:
        Z_full = rng.standard_normal(n_paths)

    ST = terminal_price(S, T, r, sigma, q, Z_full)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type doit etre 'call' ou 'put'")

    discounted = np.exp(-r * T) * payoff

    if antithetic:
        # il faut moyenner les paires (Z, -Z) avant de calculer l'ecart-type,
        # sinon on sous-estime l'erreur standard (les paires sont l'unite i.i.d, pas les paths bruts)
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
    """
    Pour chaque taille d'echantillon, on repete n_repeats fois avec des seeds
    differentes et on prend l'erreur absolue moyenne. Une seule simulation par
    point donne une courbe trop bruitee pour etre lisible.
    """
    from black_scholes import bs_price

    if path_counts is None:
        path_counts = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

    rng = np.random.default_rng(seed)
    true_price = bs_price(S, K, T, r, sigma, q, option_type)

    naive_errors, anti_errors = [], []

    for n in path_counts:
        naive_run, anti_run = [], []
        for _ in range(n_repeats):
            s = rng.integers(0, 1_000_000)
            p_naive, _ = mc_price(S, K, T, r, sigma, q, option_type, n_paths=n,
                                   antithetic=False, seed=s)
            p_anti, _ = mc_price(S, K, T, r, sigma, q, option_type, n_paths=n,
                                  antithetic=True, seed=s)
            naive_run.append(abs(p_naive - true_price))
            anti_run.append(abs(p_anti - true_price))
        naive_errors.append(np.mean(naive_run))
        anti_errors.append(np.mean(anti_run))

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

    print("Black-Scholes (analytique):", round(bs, 4))
    print(f"MC naif:      {mc_naive:.4f}  (erreur std: {se_naive:.4f})")
    print(f"MC antithetique: {mc_anti:.4f}  (erreur std: {se_anti:.4f})")
    print(f"gain sur l'erreur std (naif/antithetique): {se_naive / se_anti:.2f}x")
