from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT  = BASE / "tables"
OUT.mkdir(parents=True, exist_ok=True)

runs_path = DATA / "runs.csv"
runs = pd.read_csv(runs_path)

required_cols = {
    "instance","solver","variant","seed","feasible",
    "time_to_first_feasible","time_total","restarts_or_nodes",
    "total_penalty","soft_teacher_pref","soft_room_pref",
    "soft_student_gap","soft_timeslot_spread"
}
missing = required_cols - set(runs.columns)
if missing:
    raise ValueError(f"runs.csv 缺少这些列: {missing}")

feas = runs.groupby(["instance","solver","variant"], as_index=False).agg(
    Feasible=("feasible", lambda s: "Y" if (s.astype(str).str.upper()=="Y").any() else "N"),
    T_first_s=("time_to_first_feasible","mean"),
    T_total_s=("time_total","mean"),
    Restarts_Nodes=("restarts_or_nodes","mean")
)
feas = feas.round({"T_first_s":2,"T_total_s":2,"Restarts_Nodes":2})
feas_ren = feas.rename(columns={
    "instance":"Instance","solver":"Solver","variant":"Variant",
    "T_first_s":"T\\_first(s)","T_total_s":"T\\_total(s)"
})
with open(OUT/"feasibility.tex","w",encoding="utf-8") as f:
    f.write("\\begin{table}[ht]\n\\centering\n")
    f.write("\\caption{Feasibility and Efficiency on Benchmarks}\n")
    f.write("\\label{tab:feasibility}\n")
    f.write(feas_ren.to_latex(index=False, escape=False))
    f.write("\\end{table}\n")

qual = runs.groupby(["instance","solver","variant"], as_index=False).agg(
    Total=("total_penalty","mean"),
    TeacherPref=("soft_teacher_pref","mean"),
    RoomPref=("soft_room_pref","mean"),
    StudentGaps=("soft_student_gap","mean"),
    Spread=("soft_timeslot_spread","mean"),
).round(2)
qual_ren = qual.rename(columns={
    "instance":"Instance","solver":"Solver","variant":"Variant"
})
with open(OUT/"quality.tex","w",encoding="utf-8") as f:
    f.write("\\begin{table}[ht]\n\\centering\n")
    f.write("\\caption{Solution Quality: Soft-Constraint Penalties (Lower is Better)}\n")
    f.write("\\label{tab:quality}\n")
    f.write(qual_ren.to_latex(index=False, escape=False))
    f.write("\\end{table}\n")

sat_base = runs[(runs["solver"]=="SAT") & (runs["variant"].str.lower()=="baseline")]
sat_prop = runs[(runs["solver"]=="SAT") & (runs["variant"].str.contains("heuristic", case=False, na=False))]

def summarize_ablation(metric: str) -> pd.DataFrame:
    base = sat_base.groupby("instance")[metric].mean()
    prop = sat_prop.groupby("instance")[metric].mean()
    df = pd.concat([base.rename("Baseline"), prop.rename("Proposed")], axis=1)
    df["Gain"] = df["Baseline"] - df["Proposed"]
    df["Gain(\\%)"] = 100.0 * df["Gain"] / df["Baseline"]
    df = df.reset_index().rename(columns={"instance":"Instance"}).round(2)
    df.insert(0, "Metric", metric)
    return df

ablation = pd.concat([
    summarize_ablation("total_penalty"),
    summarize_ablation("time_total")
], ignore_index=True)

with open(OUT/"ablation.tex","w",encoding="utf-8") as f:
    f.write("\\begin{table}[ht]\n\\centering\n")
    f.write("\\caption{Ablation: Effect of Heuristics (SAT Baseline vs +Heuristic)}\n")
    f.write("\\label{tab:ablation}\n")
    f.write(ablation.to_latex(index=False, escape=False))
    f.write("\\end{table}\n")

print("LaTeX tables written to:", OUT)
