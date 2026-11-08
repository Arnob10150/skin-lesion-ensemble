"""Export result CSVs to LaTeX booktabs tables.

Design choices:
- Internal and external main tables are SPLIT into a "primary metrics" part and
  a "secondary metrics" part to avoid overflow on the 2-column Wiley page.
- Paired / ablation / calibration / compute / per-class tables stay single.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT  = os.path.abspath(os.path.join(HERE, "..", "paper", "tables"))
os.makedirs(OUT, exist_ok=True)

def latex_table(df, numcols_bf=None):
    cols = df.columns.tolist()
    aligns = ["l"] + ["c"]*(len(cols)-1)
    best_idx = {}
    if numcols_bf:
        for c in numcols_bf:
            if c in cols:
                try:
                    vals = pd.to_numeric(df[c].astype(str).str.split(" ").str[0].str.replace("±", ""), errors="coerce")
                    best_idx[c] = int(vals.idxmax()) if vals.notna().any() else None
                except Exception: pass
    lines = ["\\begin{tabular}{" + "".join(aligns) + "}", "\\toprule",
             " & ".join(str(c) for c in cols) + " \\\\", "\\midrule"]
    for i, row in df.iterrows():
        cells = []
        for c in cols:
            v = str(row[c])
            if c in best_idx and best_idx[c] == i: v = f"\\textbf{{{v}}}"
            cells.append(v)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)

# --- Primary / secondary split for the two main tables ---
PRIMARY_COLS = ["Accuracy", "Balanced acc.", "Macro-F1", "AUROC"]
SECONDARY_COLS = ["Top-3 acc.", "Kappa", "MCC", "ECE"]

def split_main(csv_name, prefix):
    p = os.path.join(RES, csv_name)
    if not os.path.exists(p): return
    df = pd.read_csv(p)
    df = df.loc[:, [c for c in df.columns if not c.startswith("_")]]
    primary = df[["Model"] + [c for c in PRIMARY_COLS if c in df.columns]]
    secondary = df[["Model"] + [c for c in SECONDARY_COLS if c in df.columns]]
    open(os.path.join(OUT, f"{prefix}_a.tex"), "w", encoding="utf8").write(
        latex_table(primary, numcols_bf=PRIMARY_COLS) + "\n")
    open(os.path.join(OUT, f"{prefix}_b.tex"), "w", encoding="utf8").write(
        latex_table(secondary, numcols_bf=SECONDARY_COLS) + "\n")
    print(f"wrote {prefix}_a.tex and {prefix}_b.tex")

# compact single-piece tables
JOBS = [
    ("table_paired.csv",       "table_paired.tex",       []),
    ("table_ablation.csv",     "table_ablation.tex",     ["Macro-F1", "AUROC"]),
    ("table_perclass.csv",     "table_perclass.tex",     ["F1", "Precision", "Recall"]),
    ("table_calibration.csv",  "table_calibration.tex",  []),
    ("table_compute.csv",      "table_compute.tex",      []),
]

split_main("table_internal.csv", "table_internal")
split_main("table_external.csv", "table_external")

for csv, tex, bfcols in JOBS:
    p = os.path.join(RES, csv)
    if not os.path.exists(p): continue
    df = pd.read_csv(p)
    df = df.loc[:, [c for c in df.columns if not c.startswith("_")]]
    if csv == "table_paired.csv":
        df = df.rename(columns={c: r"$\Delta$ macro-F1 (95\% CI)"
                                for c in df.columns if "Macro-F1" in c})
    # trim ablation table — collapse single-backbone rows ending in "(soft)" and drop secondary cols
    if csv == "table_ablation.csv" and "Model" in df.columns:
        df = df[df["Model"].apply(lambda s: ("+" in s) or ("All" in s))]
        df = df[["Model"] + [c for c in ["Accuracy", "Balanced acc.", "Macro-F1", "AUROC"] if c in df.columns]]
    code = latex_table(df, numcols_bf=bfcols)
    open(os.path.join(OUT, tex), "w", encoding="utf8").write(code + "\n")
    print(f"wrote {tex}")

# For backward compatibility, also keep legacy table_internal.tex / table_external.tex
for src, dst in [("table_internal_a.tex", "table_internal.tex"),
                 ("table_external_a.tex", "table_external.tex")]:
    sp, dp = os.path.join(OUT, src), os.path.join(OUT, dst)
    if os.path.exists(sp):
        with open(sp, 'r', encoding='utf8') as f: s = f.read()
        with open(dp, 'w', encoding='utf8') as f: f.write(s)

print("done.")
