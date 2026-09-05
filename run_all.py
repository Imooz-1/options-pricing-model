# lance tout le projet d'un coup : tests des 3 modules + regeneration des graphes
# usage : python run_all.py

import subprocess
import sys
import os

SRC = os.path.join(os.path.dirname(__file__), "src")

scripts = ["black_scholes.py", "monte_carlo.py", "hedging.py", "generate_plots.py"]

for script in scripts:
    print(f"\n--- {script} ---")
    subprocess.run([sys.executable, script], cwd=SRC, check=True)
