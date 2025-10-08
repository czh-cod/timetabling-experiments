from .ilp_solver import run_ilp_solver
from .sat_solver import run_sat_solver

def run_hybrid_solver(data: dict):
    ilp = run_ilp_solver(data)
    if ilp["status"] in ("Optimal", "Feasible") and ilp["assignments"]:
        return {"status": "HYBRID_ILP", "assignments": ilp["assignments"], "solve_time": ilp["solve_time"]}
    sat = run_sat_solver(data)
    return {"status": "HYBRID_SAT_" + sat["status"], "assignments": sat["assignments"], "solve_time": sat["solve_time"]}
