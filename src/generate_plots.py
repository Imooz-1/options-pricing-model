# Genere les 4 graphes utilises dans le README :
# 1. convergence Monte Carlo (naif vs antithetique) vs prix BS
# 2. Greeks en fonction du spot
# 3. distribution du PnL de couverture
# 4. erreur de couverture vs frequence de rebalancement

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from black_scholes import bs_price, delta, gamma, vega, theta
from monte_carlo import convergence_study
from hedging import simulate_delta_hedge, rebalancing_frequency_study

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

S0, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.2


# 1. convergence Monte Carlo -------------------------------------------------
result = convergence_study(S0, K, T, r, sigma, option_type="call")

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(result["path_counts"], result["naive_errors"], "o-", label="Monte Carlo naïf", color="#d62728")
ax.loglog(result["path_counts"], result["anti_errors"], "s-", label="Monte Carlo (variables antithétiques)", color="#1f77b4")
ax.loglog(result["path_counts"], 1 / np.sqrt(result["path_counts"]) * result["naive_errors"][0] * np.sqrt(result["path_counts"][0]),
           "--", color="gray", alpha=0.6, label="Référence O(1/√n)")
ax.set_xlabel("Nombre de trajectoires simulées")
ax.set_ylabel("Erreur absolue vs prix Black-Scholes")
ax.set_title("Convergence Monte Carlo vs prix analytique Black-Scholes")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "mc_convergence.png"), dpi=150)
plt.close(fig)


# 2. Greeks vs spot -----------------------------------------------------------
S_range = np.linspace(50, 150, 200)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(S_range, delta(S_range, K, T, r, sigma, option_type="call"), label="Call", color="#1f77b4")
axes[0, 0].plot(S_range, delta(S_range, K, T, r, sigma, option_type="put"), label="Put", color="#d62728")
axes[0, 0].axvline(K, color="gray", linestyle=":", alpha=0.6)
axes[0, 0].set_title("Delta")
axes[0, 0].set_xlabel("Prix du sous-jacent")
axes[0, 0].legend()

axes[0, 1].plot(S_range, gamma(S_range, K, T, r, sigma), color="#2ca02c")
axes[0, 1].axvline(K, color="gray", linestyle=":", alpha=0.6)
axes[0, 1].set_title("Gamma (identique call/put)")
axes[0, 1].set_xlabel("Prix du sous-jacent")

axes[1, 0].plot(S_range, vega(S_range, K, T, r, sigma), color="#9467bd")
axes[1, 0].axvline(K, color="gray", linestyle=":", alpha=0.6)
axes[1, 0].set_title("Vega (identique call/put)")
axes[1, 0].set_xlabel("Prix du sous-jacent")

axes[1, 1].plot(S_range, theta(S_range, K, T, r, sigma, option_type="call") / 365, label="Call", color="#1f77b4")
axes[1, 1].plot(S_range, theta(S_range, K, T, r, sigma, option_type="put") / 365, label="Put", color="#d62728")
axes[1, 1].axvline(K, color="gray", linestyle=":", alpha=0.6)
axes[1, 1].set_title("Theta (par jour)")
axes[1, 1].set_xlabel("Prix du sous-jacent")
axes[1, 1].legend()

fig.suptitle(f"Greeks en fonction du spot (K={K}, T={T}an, r={r}, σ={sigma})", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "greeks.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# 3. distribution du PnL de couverture ----------------------------------------
hedge_result = simulate_delta_hedge(S0, K, T, r, sigma, option_type="call",
                                     n_steps=52, n_paths=20_000, seed=7)

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(hedge_result["pnl"], bins=80, color="#1f77b4", alpha=0.75, edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.axvline(hedge_result["mean_pnl"], color="#d62728", linestyle="-", linewidth=1.5,
           label=f"P&L moyen = {hedge_result['mean_pnl']:.3f}")
ax.set_xlabel("P&L de couverture à maturité")
ax.set_ylabel("Fréquence")
ax.set_title("Distribution du P&L de couverture delta-neutre\n(rebalancement hebdomadaire, 52 pas)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "hedging_pnl_distribution.png"), dpi=150)
plt.close(fig)


# 4. erreur de couverture vs frequence de rebalancement -----------------------
freq_result = rebalancing_frequency_study(S0, K, T, r, sigma, option_type="call")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(freq_result["steps_list"], freq_result["pnl_std"], "o-", color="#1f77b4")
ax.set_xlabel("Nombre de rebalancements sur la durée de vie de l'option")
ax.set_ylabel("Écart-type du P&L de couverture")
ax.set_title("Erreur de couverture discrète vs fréquence de rebalancement")
ax.set_xscale("log")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "hedging_error_vs_frequency.png"), dpi=150)
plt.close(fig)

print("figures generees dans", PLOTS_DIR)
print("- mc_convergence.png")
print("- greeks.png")
print("- hedging_pnl_distribution.png")
print("- hedging_error_vs_frequency.png")
