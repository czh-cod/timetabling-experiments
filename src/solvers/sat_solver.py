# src/solvers/sat_solver.py
import time
from collections import defaultdict

def run_sat_solver(data: dict, time_limit: int = 30):
    """
    轻量 SAT 求解接口：
    - 若安装 python-sat，则构造简单布尔模型；
    - 否则回退到贪心构造（演示用途）。
    返回: dict(status, assignments, solve_time)
    """
    start = time.time()

    try:
        from pysat.solvers import Glucose3  # Optional dependency
        has_pysat = True
    except Exception:
        has_pysat = False

    classes = data["classes"]
    rooms = data["rooms"]
    insts = data["instructors"]
    sched = data["schedule"]

    slots = [f"{d}-{t}".replace("-", "") for d, t in zip(sched["day"], sched["slot"])]
    C = list(classes["class_id"])
    R = list(rooms["room_id"])

    # ------- 回退：贪心可行化（无外部依赖） -------
    if not has_pysat:
        used = set()  # (r, s)
        inst_at = defaultdict(set)  # i -> {s}
        cls_inst = dict(zip(classes["class_id"], classes["instructor_id"]))
        assignments = []
        for c in C:
            i = cls_inst[c]
            placed = False
            for s in slots:
                if s in inst_at[i]:
                    continue
                for r in R:
                    if (r, s) in used:
                        continue
                    used.add((r, s))
                    inst_at[i].add(s)
                    assignments.append({"class_id": c, "room": r, "slot": s})
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                # 放不下就返回不可满足
                return {"status": "UNSAT", "assignments": [], "solve_time": round(time.time() - start, 3)}
        return {"status": "SAT", "assignments": assignments, "solve_time": round(time.time() - start, 3)}

    # ------- 有 PySAT：示意性 CNF 编码 -------
    # 变量编码：v(c,r,s) -> id
    var_id = {}
    vid = 1
    for c in C:
        for r in R:
            for s in slots:
                var_id[(c, r, s)] = vid
                vid += 1

    g = Glucose3()
    # 每课至少一个 (r,s)
    for c in C:
        g.add_clause([var_id[(c, r, s)] for r in R for s in slots])
    # 每课至多一个
    for c in C:
        idx = [var_id[(c, r, s)] for r in R for s in slots]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                g.add_clause([-idx[i], -idx[j]])

    # 房间冲突
    for r in R:
        for s in slots:
            vars_rs = [var_id[(c, r, s)] for c in C]
            for i in range(len(vars_rs)):
                for j in range(i + 1, len(vars_rs)):
                    g.add_clause([-vars_rs[i], -vars_rs[j]])

    # 老师冲突
    cls_inst = dict(zip(classes["class_id"], classes["instructor_id"]))
    for s in slots:
        for i in insts["instructor_id"]:
            cls_i = [c for c in C if cls_inst[c] == i]
            for r1 in R:
                for r2 in R:
                    for m in range(len(cls_i)):
                        for n in range(m + 1, len(cls_i)):
                            g.add_clause([-var_id[(cls_i[m], r1, s)], -var_id[(cls_i[n], r2, s)]])

    ok = g.solve_limited(expect_interrupt=True)
    if not ok:
        return {"status": "UNSAT", "assignments": [], "solve_time": round(time.time() - start, 3)}

    model = set(l for l in g.get_model() if l > 0)
    assignments = []
    for (c, r, s), v in var_id.items():
        if v in model:
            assignments.append({"class_id": c, "room": r, "slot": s})

    return {"status": "SAT", "assignments": assignments, "solve_time": round(time.time() - start, 3)}
