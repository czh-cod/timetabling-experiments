import time
from itertools import product
import pandas as pd

try:
    import pulp as pl
except ImportError as e:
    raise RuntimeError("PuLP is required for ILP solver. Install via: pip install pulp") from e


def _parse_slots(schedule_df):
    return list(schedule_df.apply(lambda r: f"{r['day']}-{r['slot']}", axis=1))


def _parse_instructor_avail(inst_df):
    avail = {}
    for _, r in inst_df.iterrows():
        raw = str(r.get("available_slots", "")).replace("|", ",")
        slots = set(s.strip().replace(" ", "").replace("-", "")
                    for s in raw.split(",") if s.strip())
        avail[r["instructor_id"]] = slots
    return avail


def run_ilp_solver(data: dict, timeout_sec: int = 60):
    """
    data: {
      'classes': DataFrame[class_id, course_name, instructor_id, num_students],
      'rooms':   DataFrame[room_id, capacity],
      'instructors': DataFrame[instructor_id, name, available_slots],
      'schedule': DataFrame[day, slot]
    }
    returns: dict(status, assignments, solve_time)
    """
    start = time.time()

    classes = data["classes"].copy()
    rooms = data["rooms"].copy()
    insts = data["instructors"].copy()
    sched = data["schedule"].copy()

    slots = [s.replace("-", "") for s in _parse_slots(sched)]
    inst_avail = _parse_instructor_avail(insts)

    C = list(classes["class_id"])
    R = list(rooms["room_id"])
    S = slots

    cls_inst = dict(zip(classes["class_id"], classes["instructor_id"]))
    cls_size = dict(zip(classes["class_id"], classes.get("num_students", pd.Series([1]*len(C)))))

    room_cap = dict(zip(rooms["room_id"], rooms["capacity"]))

    prob = pl.LpProblem("Timetabling_ILP", pl.LpMinimize)
    X = pl.LpVariable.dicts("x", (C, R, S), lowBound=0, upBound=1, cat=pl.LpBinary)

    prob += 0

    for c in C:
        prob += pl.lpSum(X[c][r][s] for r, s in product(R, S)) == 1

    for c in C:
        for r in R:
            for s in S:
                prob += X[c][r][s] <= int(room_cap[r] >= int(cls_size[c]))

    for r in R:
        for s in S:
            prob += pl.lpSum(X[c][r][s] for c in C) <= 1

    for s in S:
        for i in insts["instructor_id"]:
            # 老师可用
            allow = (s in inst_avail.get(i, set())) if len(inst_avail) else True
            for c in C:
                if cls_inst[c] == i and not allow:
                    for r in R:
                        prob += X[c][r][s] == 0
        # 不冲突
        for i in insts["instructor_id"]:
            prob += pl.lpSum(X[c][r][s]
                             for c in C if cls_inst[c] == i
                             for r in R) <= 1

    # 设置求解器
    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=timeout_sec)
    status = prob.solve(solver)

    solve_time = time.time() - start
    status_str = pl.LpStatus[status]

    assignments = []
    if status_str == "Optimal" or status_str == "Feasible":
        for c in C:
            for r in R:
                for s in S:
                    if pl.value(X[c][r][s]) > 0.5:
                        assignments.append({"class_id": c, "room": r, "slot": s})

    return {
        "status": status_str,
        "assignments": assignments,
        "solve_time": round(solve_time, 3)
    }
