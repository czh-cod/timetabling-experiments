import csv
from pathlib import Path
import time

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
DATA.mkdir(parents=True, exist_ok=True)

RUNS_FILE = DATA / "runs.csv"
TIMELINE_FILE = DATA / "timeline.csv"

# 表头
RUNS_HEADER = [
    "instance","solver","variant","seed","feasible",
    "time_to_first_feasible","time_total","restarts_or_nodes",
    "total_penalty","soft_teacher_pref","soft_room_pref",
    "soft_student_gap","soft_timeslot_spread",
    "var_count_sat","clause_count_sat","var_count_ilp","constr_count_ilp"
]

TIMELINE_HEADER = [
    "instance","solver","variant","seed","timestamp_sec","best_penalty_so_far"
]


def init_files():
    if not RUNS_FILE.exists():
        with open(RUNS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(RUNS_HEADER)

    if not TIMELINE_FILE.exists():
        with open(TIMELINE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(TIMELINE_HEADER)


def log_run(row):
    with open(RUNS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def log_timeline(instance, solver, variant, seed, start_time, best_penalty):
    now = time.time() - start_time
    with open(TIMELINE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([instance, solver, variant, seed, int(now), best_penalty])
