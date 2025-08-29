import os
print("当前工作目录:", os.getcwd())

# src/solve_toy_ilp.py
import pandas as pd
import pulp
import time
from collections import defaultdict
from src import logger
import os

INSTANCE = os.getenv("EXP_INSTANCE", "toy01")
SOLVER   = os.getenv("EXP_SOLVER", "ILP")
VARIANT  = os.getenv("EXP_VARIANT", "baseline")
SEED     = int(os.getenv("EXP_SEED", "0"))
TIME_LIM = int(os.getenv("EXP_TLIMIT", "600"))
NOTE     = os.getenv("EXP_NOTE", "")

import random, numpy as np
random.seed(SEED)
np.random.seed(SEED)


logger.init_files()

instance = "toy01"
solver = "ILP"
variant = "baseline"
seed = 0

start = time.time()
time_to_first = None
best_penalty = float("inf")


from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
DATASETS = BASE / "datasets"

df_rooms = pd.read_csv(DATASETS / "rooms.csv")
df_insts = pd.read_csv(DATASETS / "instructors.csv")
df_classes = pd.read_csv(DATASETS / "classes.csv")


N = 200
R = 20
S = 30
classes = df_classes.head(N).copy()
rooms = df_rooms.head(R).copy()
slots = list(range(1, S + 1))


classes["instructor"] = classes["instructor"].fillna("NA")
rooms["capacity"] = pd.to_numeric(rooms["capacity"], errors="coerce").fillna(0).astype(int)


candidate = []
for _, c in classes.iterrows():
    e = str(c["id"])
    inst = str(c["instructor"])
    for _, r in rooms.iterrows():
        room_id = str(r["id"])
        for s in slots:
            candidate.append((e, room_id, s, inst))


model = pulp.LpProblem("ToyTimetabling", pulp.LpMinimize)

# 决策变量 y[e,r,s] ∈ {0,1}
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

print(f"Solving toy ILP ... (N={N}, R={R}, S={S})")
status = model.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=300))
print("Status:", pulp.LpStatus[status])

if pulp.LpStatus[status] == "Optimal" or pulp.LpStatus[status] == "Feasible":
    feasible = "Y"
    time_to_first = time.time() - start
    best_penalty = 0
    logger.log_timeline(instance, solver, variant, seed, start, best_penalty)
else:
    feasible = "N"

schedule = []
for (e, r, s, _) in candidate:
    if pulp.value(y[(e, r, s)]) > 0.5:
        inst = classes.loc[classes["id"].astype(str) == e, "instructor"].values[0]
        schedule.append({"event": e, "room": r, "slot": s, "instructor": inst})

df_sched = pd.DataFrame(schedule).sort_values(["slot", "room", "event"])
df_sched.to_csv("schedule_toy.csv", index=False)
print("Saved schedule_toy.csv with", len(df_sched), "assignments.")

time_total = time.time() - start
row = [
    instance, solver, variant, seed, feasible,
    time_to_first, time_total, len(candidate),   # 用候选变量数代替节点数
    best_penalty, 0, 0, 0, 0,                   # soft constraint penalty（先写 0）
    0, 0, len(classes), len(rooms)              # SAT=0，ILP 用课程数/房间数做示例
]
logger.log_run(row)

print("实验完成！结果已写入 runs.csv 和 timeline.csv ✅")
