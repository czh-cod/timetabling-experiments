from __future__ import annotations
import os, time
from pathlib import Path
from collections import defaultdict
import pandas as pd
import pulp
from src import logger

INSTANCE = os.getenv("EXP_INSTANCE", "toy01")
SOLVER   = os.getenv("EXP_SOLVER", "ILP")
VARIANT  = os.getenv("EXP_VARIANT", "baseline")
SEED     = int(os.getenv("EXP_SEED", "0"))
TIME_LIM = int(os.getenv("EXP_TLIMIT", "600"))
NOTE     = os.getenv("EXP_NOTE", "")

INST2PREFIX = {
    "toy01": "pu-spr07-cs",
    "toy02": "pu-spr07-cs",
    "toy03": "pu-spr07-cs",
}
DATASET_PREFIX = INST2PREFIX.get(INSTANCE, "pu-spr07-cs")

BASE = Path(__file__).resolve().parents[1]
DATASETS = BASE / "datasets"
OUTDIR   = BASE / "data"
OUTDIR.mkdir(parents=True, exist_ok=True)

import random, numpy as np
random.seed(SEED); np.random.seed(SEED)

logger.init_files()

def _read_csv_must(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path)

df_rooms = _read_csv_must(DATASETS / f"{DATASET_PREFIX}_rooms.csv")
df_insts = _read_csv_must(DATASETS / f"{DATASET_PREFIX}_instructors.csv")
df_classes = _read_csv_must(DATASETS / f"{DATASET_PREFIX}_classes.csv")

for col_old, col_new in [("class_id", "id"), ("teacher", "instructor")]:
    if col_old in df_classes.columns and "id" not in df_classes.columns:
        df_classes.rename(columns={col_old: col_new}, inplace=True)

if "capacity" not in df_rooms.columns and "cap" in df_rooms.columns:
    df_rooms.rename(columns={"cap": "capacity"}, inplace=True)

df_classes["instructor"] = df_classes.get("instructor", pd.Series(["NA"]*len(df_classes))).fillna("NA").astype(str)
df_rooms["capacity"] = pd.to_numeric(df_rooms.get("capacity", 0), errors="coerce").fillna(0).astype(int)

N, R, S = 200, 20, 30
classes = df_classes.head(N).copy()
rooms   = df_rooms.head(R).copy()
slots   = list(range(1, S + 1))

candidate = []
for _, c in classes.iterrows():
    e = str(c["id"])
    inst = str(c["instructor"])
    for _, r in rooms.iterrows():
        room_id = str(r["id"])
        for s in slots:
            candidate.append((e, room_id, s, inst))

model = pulp.LpProblem("ToyTimetabling", pulp.LpMinimize)
y = pulp.LpVariable.dicts("y",
                          ((e, r, s) for (e, r, s, _) in candidate),
                          lowBound=0, upBound=1, cat="Binary")

model += 0


for e in classes["id"].astype(str).tolist():
    model += pulp.lpSum(y[(e, r, s)] for (ee, r, s, _) in candidate if ee == e) == 1, f"assign_once_{e}"


room_slot_groups = defaultdict(list)
for (e, r, s, _) in candidate:
    room_slot_groups[(r, s)].append((e, r, s))
for (r, s), vars_ in room_slot_groups.items():
    model += pulp.lpSum(y[(e, r, s)] for (e, r, s) in vars_) <= 1, f"room_slot_{r}_{s}"


inst_slot_groups = defaultdict(list)
for (e, r, s, inst) in candidate:
    if inst != "NA":
        inst_slot_groups[(inst, s)].append((e, r, s))
for (inst, s), vars_ in inst_slot_groups.items():
    model += pulp.lpSum(y[(e, r, s)] for (e, r, s) in vars_) <= 1, f"inst_slot_{inst}_{s}"


if "num_students" in classes.columns:
    cls_size = dict(zip(classes["id"].astype(str), pd.to_numeric(classes["num_students"], errors="coerce").fillna(0).astype(int)))
    room_cap = dict(zip(rooms["id"].astype(str), rooms["capacity"]))
    for (e, r, s, _) in candidate:
        # 若房间容量不够，则强制该 y=0
        if room_cap.get(r, 0) < cls_size.get(e, 0):
            model += y[(e, r, s)] == 0, f"cap_{e}_{r}_{s}"


print(f"Solving ILP ... inst={INSTANCE} N={len(classes)} R={len(rooms)} S={len(slots)} TL={TIME_LIM}s")
t0 = time.time()
status = model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_LIM))
solve_time = time.time() - t0
status_str = pulp.LpStatus[status]
print("Status:", status_str, f"({solve_time:.2f}s)")


feasible = "Y" if status_str in ("Optimal", "Feasible") else "N"
time_to_first = solve_time if feasible == "Y" else None
best_penalty  = 0  # 目前仅可行性，soft 先置 0


schedule = []
if feasible == "Y":
    for (e, r, s, _) in candidate:
        if pulp.value(y[(e, r, s)]) > 0.5:
            inst = classes.loc[classes["id"].astype(str) == e, "instructor"].values[0]
            schedule.append({"event": e, "room": r, "slot": s, "instructor": inst})
df_sched = pd.DataFrame(schedule).sort_values(["slot", "room", "event"])
out_schedule = OUTDIR / f"schedule_ilp_{INSTANCE}.csv"
df_sched.to_csv(out_schedule, index=False)
print(f"[OK] Saved {out_schedule.name} with {len(df_sched)} assignments.")


start = time.time() - solve_time  # 近似：开始时间=当前时间-求解时长
logger.log_timeline(INSTANCE, SOLVER, VARIANT, SEED, start, best_penalty)
row = [
    INSTANCE, SOLVER, VARIANT, SEED, feasible,
    time_to_first, solve_time, len(candidate),  # node数用候选数近似
    best_penalty, 0, 0, 0, 0,                   # soft penalty 先 0
    0, 0, len(classes), len(rooms)              # 末尾统计
]
logger.log_run(row)

print("实验完成！结果已写入 data/runs.csv、data/timeline.csv ✅")
