# src/run_experiments.py


from __future__ import annotations
import os
import time
import subprocess
from pathlib import Path
from typing import List

# ---- Paths ----
BASE = Path(__file__).resolve().parents[1]
SRC  = BASE / "src"
DATA = BASE / "data"
DATA.mkdir(parents=True, exist_ok=True)

# ---- Experiment matrix (edit to your needs) ----
# Instances: keep small first; later add real dataset slices
INSTANCES: List[str] = ["toy01", "toy02", "toy03"]   # you can replace with your real cases
SOLVERS = ["ILP", "SAT", "HYBRID"]                       # later: ["ILP", "SAT", "HYBRID"]
VARIANTS:  List[str] = ["baseline"]                  # later: ["baseline", "heuristic"]
SEEDS:     List[int] = [0, 1, 2]                     # later: expand to 5 or 10 if you have time

# Time budget per run (seconds). Keep consistent across solvers for fair comparison.
TIME_LIMITS = {
    "ILP": 600,          # 10 minutes
    "SAT": 300,          # 5 minutes
    "HYBRID": 600        # e.g., SAT warm-start then ILP
}

# Optional: extra parameters you want to pass to your solvers
EXTRA = {
    "ILP":   {"variant_note": ""},
    "SAT":   {"variant_note": ""},
    "HYBRID":{"variant_note": "sat_warmstart_then_ilp"}
}

# ---- Helper to call a solver script via subprocess ----
def run_one(instance: str, solver: str, variant: str, seed: int) -> int:

    env = os.environ.copy()
    env["EXP_INSTANCE"] = instance
    env["EXP_SOLVER"]   = solver        # "ILP" / "SAT" / "HYBRID"
    env["EXP_VARIANT"]  = variant       # "baseline"/"heuristic"/...
    env["EXP_SEED"]     = str(seed)
    env["EXP_TLIMIT"]   = str(TIME_LIMITS.get(solver, 300))
    env["EXP_NOTE"]     = EXTRA.get(solver, {}).get("variant_note", "")

    # Choose script to run based on solver
    if solver == "ILP":
        script = SRC / "solve_toy_ilp.py"
    elif solver == "SAT":
        script = SRC / "solve_toy_sat.py"      # add this later when ready
    elif solver == "HYBRID":
        # Option A: a dedicated hybrid script
        script = SRC / "solve_hybrid.py"       # implement later
        # Option B: or call ILP with a warm-start file produced by a prior SAT call
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Call the solver as a subprocess; stdout/stderr are shown in the console
    print(f"[RUN] inst={instance} | solver={solver} | variant={variant} | seed={seed}")
    t0 = time.time()
    try:
        # Use your current Python interpreter
        subprocess.run(
            ["python", str(script)],
            check=True,
            env=env,
            cwd=str(BASE)  # ensure relative paths inside solver are resolved from project root
        )
        ret = 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Subprocess failed with return code {e.returncode}")
        ret = e.returncode
    dt = time.time() - t0
    print(f"[DONE] ({dt:.1f}s) inst={instance}, solver={solver}, variant={variant}, seed={seed}")
    return ret

def main():


    total = 0
    fail  = 0
    for inst in INSTANCES:
        for solver in SOLVERS:
            for variant in VARIANTS:
                for seed in SEEDS:
                    total += 1
                    rc = run_one(inst, solver, variant, seed)
                    if rc != 0:
                        fail += 1
    print(f"\n[SUMMARY] total runs: {total}, failed: {fail}")
    print(f"Outputs appended to: {DATA/'runs.csv'} and {DATA/'timeline.csv'}")

if __name__ == "__main__":
    main()
