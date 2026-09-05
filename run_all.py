"""
Entry point: runs the sanity checks for each module and regenerates every
figure used in the README. Run with:

    python run_all.py
"""

import subprocess
import sys
import os

SRC = os.path.join(os.path.dirname(__file__), "src")

scripts = ["black_scholes.py", "monte_carlo.py", "hedging.py", "generate_plots.py"]

for script in scripts:
    print(f"\n{'=' * 60}\nRunning {script}\n{'=' * 60}")
    subprocess.run([sys.executable, script], cwd=SRC, check=True)
