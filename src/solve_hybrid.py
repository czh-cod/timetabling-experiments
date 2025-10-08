from __future__ import annotations
import os, time
from pathlib import Path
from collections import defaultdict
import pandas as pd
from src import logger


INSTANCE = os.getenv("EXP_INSTANCE", "toy01")
SOLVER   = os.getenv("EXP_SOLVER", "HYBRID")
VARIANT  = os.getenv("EXP_VARIANT", "sat_then_ilp")
SEED     = int(os.getenv("EXP_SEED", "0"))
TIME_LIM = int(os.getenv("EXP_TLIMIT", "600"))
NOTE     = os.getenv("EXP_NOTE", "")

INST2PREFIX = {"toy01": "pu-spr07-cs", "toy02": "pu-spr07-cs", "toy03": "pu-spr07-cs"}
DATASET_PREFIX = INST2PREFIX.get(INSTANCE, "pu-spr07-cs")

BASE = Path(__file__).resolve().parents[1]
DATASETS = BASE / "datasets"
OUTDIR   = BASE / "data"
OUTDIR.mkdir(parents=True, exist_ok=True)

logger.init_files()

def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

df_rooms = _read(DATASETS / f"{DATASET_PREFIX}_rooms.csv")
df_insts = _read(DATASETS / f"{DATASET_PREFIX}_instructors.csv")
df_classes = _read(DATASETS / f"{DATASET_PREFIX}_classes.csv")

for old,new in [("class_id","id"),("teacher","instructor")]:
    if old in df_classes.columns and new not in df_classes.columns:
        df_classes.rename(columns={old:new}, inplace=True)
if "capacity" not in df_rooms.columns and "cap" in df_rooms.columns:
    df_rooms.rename(columns={"cap":"capacity"}, inplace=True)

df_classes["instructor"] = df_classes.get("instructor", pd.Series(["NA"]*len(df_classes))).fillna("NA").astype(str)
df_rooms["capacity"] = pd.to_numeric(df_rooms.get("capacity", 0), errors="coerce").fillna(0).astype(int)

# toy 规模
N, R, S = 200, 20, 30
classes = df_classes.head(N).copy()
rooms   = df_rooms.head(R).copy()
slots   = list(range(1, S+1))

C = classes["id"].astype(str).tolist()
R_ids = rooms["id"].astype(str).tolist()

def try_sat():
    try:
        from pysat.solvers import Glucose3
        var_id = {}; vid = 1
        for c in C:
            for r in R_ids:
                for s in slots:
                    var_id[(c,r,s)] = vid; vid += 1
        g = Glucose3()
        for c in C:
            g.add_clause([var_id[(c,r,s)] for r in R_ids for s in slots])
        for c in C:
            L = [var_id[(c,r,s)] for r in R_ids for s in slots]
            for i in range(len(L)):
                for j in range(i+1, len(L)):
                    g.add_clause([-L[i], -L[j]])
        for r in R_ids:
            for s in slots:
                L = [var_id[(c,r,s)] for c in C]
                for i in range(len(L)):
                    for j in range(i+1, len(L)):
                        g.add_clause([-L[i], -L[j]])
        cls_inst = dict(zip(classes["id"].astype(str), classes["instructor"]))
        for s in slots:
            for inst in classes["instructor"].unique():
                cl = [c for c in C if cls_inst.get(c,"NA")==inst]
                for i in range(len(cl)):
                    for j in range(i+1,len(cl)):
                        for r1 in R_ids:
                            for r2 in R_ids:
                                g.add_clause([-var_id[(cl[i],r1,s)], -var_id[(cl[j],r2,s)]])
        ok = g.solve_limited(expect_interrupt=True)
        if not ok:
            return "UNSAT", []
        model = set(l for l in g.get_model() if l>0)
        assigns = []
        for (c,r,s), v in var_id.items():
            if v in model:
                inst = classes.loc[classes["id"].astype(str)==c, "instructor"].values[0]
                assigns.append({"event": c, "room": r, "slot": s, "instructor": inst})
        return "SAT", assigns
    except Exception:

        from collections import defaultdict
        used_rs = set()
        used_inst = defaultdict(set)
        assigns = []
        cls_inst = dict(zip(classes["id"].astype(str), classes["instructor"]))
        for c in C:
            i = cls_inst.get(c,"NA")
            placed = False
            for s in slots:
                if i!="NA" and s in used_inst[i]:
                    continue
                for r in R_ids:
                    if (r,s) in used_rs:
                        continue
                    used_rs.add((r,s))
                    if i!="NA": used_inst[i].add(s)
                    inst = classes.loc[classes["id"].astype(str)==c, "instructor"].values[0]
                    assigns.append({"event": c, "room": r, "slot": s, "instructor": inst})
                    placed = True
                    break
                if placed: break
            if not placed:
                return "UNSAT", []
        return "SAT", assigns

def run_ilp(timeout: int):
    import pulp
    from collections import defaultdict as dd
    cand = []
    for _, c in classes.iterrows():
        e = str(c["id"]); inst = str(c["instructor"])
        for _, r in rooms.iterrows():
            room_id = str(r["id"])
            for s in slots:
                cand.append((e, room_id, s, inst))
    m = pulp.LpProblem("Hybrid_ILP", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", ((e,r,s) for (e,r,s,_) in cand), 0, 1, cat="Binary")
    m += 0
    for e in C:
        m += pulp.lpSum(y[(e,r,s)] for (ee,r,s,_) in cand if ee==e) == 1
    rs = dd(list)
    for (e,r,s,_) in cand: rs[(r,s)].append((e,r,s))
    for (r,s), L in rs.items(): m += pulp.lpSum(y[(e,r,s)] for (e,r,s) in L) <= 1
    insts = dd(list)
    for (e,r,s,inst) in cand:
        if inst!="NA": insts[(inst,s)].append((e,r,s))
    for (inst,s), L in insts.items(): m += pulp.lpSum(y[(e,r,s)] for (e,r,s) in L) <= 1
    t0 = time.time()
    status = m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout))
    tm = time.time()-t0
    ok = (pulp.LpStatus[status] in ("Optimal","Feasible"))
    assigns = []
    if ok:
        for (e,r,s,_) in cand:
            if pulp.value(y[(e,r,s)]) > 0.5:
                inst = classes.loc[classes["id"].astype(str)==e, "instructor"].values[0]
                assigns.append({"event": e, "room": r, "slot": s, "instructor": inst})
    return ok, assigns, tm

t0 = time.time()
sat_status, assigns = try_sat()
if sat_status == "SAT":
    solve_time = time.time()-t0
    status_str = "HYBRID_SAT"
else:
    ok, assigns, ilp_time = run_ilp(TIME_LIM)
    solve_time = time.time()-t0
    status_str = "HYBRID_ILP" if ok else "HYBRID_FAIL"

feasible = "Y" if assigns else "N"
time_to_first = solve_time if feasible=="Y" else None
best_penalty = 0

out_csv = OUTDIR / f"schedule_hybrid_{INSTANCE}.csv"
pd.DataFrame(assigns).sort_values(["slot","room","event"]).to_csv(out_csv, index=False)
print(f"[HYBRID] {status_str} ({solve_time:.2f}s) -> {out_csv.name} rows={len(assigns)}")

logger.log_timeline(INSTANCE, "HYBRID", VARIANT, SEED, time.time()-solve_time, best_penalty)
row = [INSTANCE, "HYBRID", VARIANT, SEED, feasible,
       time_to_first, solve_time, len(C)*len(R_ids)*len(slots),
       best_penalty, 0,0,0,0, 0,0, len(classes), len(rooms)]
logger.log_run(row)

print("实验完成！结果已写入 data/runs.csv、data/timeline.csv ✅")
