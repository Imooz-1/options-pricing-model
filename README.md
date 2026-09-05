# Options Pricing Engine — Black-Scholes, Monte Carlo & Delta Hedging

Moteur de pricing pour options vanilles européennes, combinant la formule fermée de **Black-Scholes**, une implémentation **Monte Carlo avec réduction de variance** (variables antithétiques), le calcul des **Greeks**, et une **simulation de couverture delta-neutre** pour quantifier le risque résiduel d'un rebalancement discret.

## Pourquoi ce projet

Ce projet ne se limite pas à implémenter la formule de Black-Scholes : l'objectif est de couvrir les trois briques qu'un desk de trading ou de structuration utilise réellement au quotidien — pricer, quantifier les risques (Greeks), et vérifier que la couverture théorique tient en pratique quand on ne peut rebalancer qu'à intervalles discrets.

## Structure du repo

```
options-pricing/
├── README.md
├── requirements.txt
├── run_all.py                  # lance tous les modules + régénère les figures
├── src/
│   ├── black_scholes.py        # prix analytique + Greeks (delta, gamma, vega, theta, rho)
│   ├── monte_carlo.py           # pricing MC (naïf + variables antithétiques) et étude de convergence
│   ├── hedging.py               # simulation de couverture delta-neutre discrète
│   └── generate_plots.py        # génère les 4 figures ci-dessous
└── plots/
    ├── mc_convergence.png
    ├── greeks.png
    ├── hedging_pnl_distribution.png
    └── hedging_error_vs_frequency.png
```

## Méthodologie

### 1. Pricing analytique (Black-Scholes)
Implémentation directe des formules fermées pour call et put européens, avec dividende continu optionnel (`q`). Les cinq Greeks principaux (delta, gamma, vega, theta, rho) sont dérivés analytiquement plutôt qu'approximés par différences finies.

### 2. Pricing par simulation (Monte Carlo)
Le prix terminal du sous-jacent est simulé exactement (solution fermée du mouvement brownien géométrique, pas de biais de discrétisation). Deux estimateurs sont comparés :
- **Monte Carlo naïf** : moyenne simple des payoffs actualisés
- **Variables antithétiques** : pour chaque tirage `Z`, on utilise aussi `-Z`. Cette technique de réduction de variance exploite la corrélation négative entre les payoffs de tirages symétriques pour une fonction monotone, réduisant l'erreur standard sans biais supplémentaire.

### 3. Simulation de couverture delta-neutre
Un trader vend une option, encaisse la prime Black-Scholes, et couvre sa position en détenant `delta` actions du sous-jacent, rebalancées à intervalles discrets (hebdomadaire par défaut). En théorie continue, cette réplication est parfaite (P&L nul). En pratique, le rebalancement discret laisse une erreur résiduelle — ce projet la quantifie et montre comment elle diminue avec la fréquence de rebalancement.

## Résultats

### Convergence Monte Carlo
Erreur moyenne (sur 30 runs indépendants) entre le prix Monte Carlo et le prix Black-Scholes, en fonction du nombre de trajectoires simulées.

![Convergence Monte Carlo](plots/mc_convergence.png)

La méthode par variables antithétiques converge systématiquement plus vite que la méthode naïve, à nombre de tirages égal — pour un budget de calcul donné, elle est donc préférable.

### Greeks
Sensibilité du prix de l'option au spot, à la volatilité et au temps, pour K=100, T=1 an, r=3%, σ=20%.

![Greeks](plots/greeks.png)

### Distribution du P&L de couverture
Sur 20 000 simulations, avec un rebalancement hebdomadaire (52 pas sur l'année) :

![Distribution du P&L de couverture](plots/hedging_pnl_distribution.png)

Le P&L moyen est proche de zéro (la couverture est non biaisée), mais sa dispersion illustre le risque résiduel dû au rebalancement discret plutôt que continu.

### Erreur de couverture vs fréquence de rebalancement
L'écart-type du P&L de couverture diminue quand on rebalance plus souvent — le compromis classique entre coût de transaction (non modélisé ici) et qualité de la couverture.

![Erreur de couverture vs fréquence](plots/hedging_error_vs_frequency.png)

## Utilisation

```bash
git clone <url-du-repo>
cd options-pricing
pip install -r requirements.txt
python run_all.py
```

Chaque module peut aussi être utilisé indépendamment :

```python
import sys
sys.path.insert(0, "src")

from black_scholes import bs_price, delta, gamma, vega, theta
from monte_carlo import mc_price
from hedging import simulate_delta_hedge

price = bs_price(S=100, K=100, T=1.0, r=0.03, sigma=0.2, option_type="call")
mc_estimate, std_error = mc_price(S=100, K=100, T=1.0, r=0.03, sigma=0.2,
                                   option_type="call", n_paths=100_000)
hedge_result = simulate_delta_hedge(S0=100, K=100, T=1.0, r=0.03, sigma=0.2,
                                     option_type="call", n_steps=52, n_paths=10_000)
```

## Limites et extensions possibles

- Volatilité supposée constante (pas de smile/skew) — une extension naturelle serait un modèle à volatilité stochastique (Heston) ou local (Dupire)
- Coûts de transaction non modélisés dans la simulation de couverture
- Options européennes uniquement — le pricing américain nécessiterait un arbre binomial ou Longstaff-Schwartz (Monte Carlo américain)
