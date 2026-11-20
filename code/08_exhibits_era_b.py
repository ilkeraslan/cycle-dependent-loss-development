"""Era-B tables, the era-B covariate figure (Figure 2 of the paper) and the extended robustness and panel tables."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, FIG, TAB = "results", "results/figures", "results/tables"
NAMES = {"comauto": "Commercial auto", "wkcomp": "Workers' compensation",
         "othliab": "Other liability", "prodliab": "Products liability",
         "ppauto": "Private passenger auto", "medmal": "Medical malpractice"}
MAIN = ["comauto", "wkcomp", "othliab", "prodliab"]
ORDER = MAIN + ["ppauto", "medmal"]

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "figure.dpi": 200})

res = pd.read_csv(f"{OUT}/beta_estimates_twoera.csv")

def stars(p):
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.1 else ""

def cell(r):
    return f"{r['beta']:.4f}{stars(r['p'])} ({r['se']:.4f})"

def main_table(era, fname):
    rows = []
    for line in ORDER:
        r = res[(res["line"] == line) & (res["era"] == era)]
        if r.empty:
            continue
        try:
            raw_p = r[(r.spec == "raw") & (r["var"] == "z_raw") & (r.basis == "paid")].iloc[0]
            cy_p = r[(r.spec == "cyc+rel") & (r["var"] == "z_cyc") & (r.basis == "paid")].iloc[0]
            gr_p = r[(r.spec == "cyc+rel") & (r["var"] == "g_rel") & (r.basis == "paid")].iloc[0]
            raw_i = r[(r.spec == "raw") & (r["var"] == "z_raw") & (r.basis == "incurred")].iloc[0]
            cy_i = r[(r.spec == "cyc+rel") & (r["var"] == "z_cyc") & (r.basis == "incurred")].iloc[0]
            gr_i = r[(r.spec == "cyc+rel") & (r["var"] == "g_rel") & (r.basis == "incurred")].iloc[0]
        except IndexError:
            continue
        rows.append((NAMES[line], cell(raw_p), cell(cy_p), cell(gr_p),
                     cell(raw_i), cell(cy_i), cell(gr_i), int(cy_p["n"])))
    pool = res[(res["line"] == "POOLED_MAIN") & (res["era"] == era)]
    pp = pool[pool.basis == "paid"]
    pi = pool[pool.basis == "incurred"]
    rows.append(("Pooled casualty lines", "--",
                 cell(pp[pp["var"] == "z_cyc"].iloc[0]),
                 cell(pp[pp["var"] == "g_rel"].iloc[0]), "--",
                 cell(pi[pi["var"] == "z_cyc"].iloc[0]),
                 cell(pi[pi["var"] == "g_rel"].iloc[0]),
                 int(pp["n"].iloc[0])))
    with open(f"{TAB}/{fname}", "w") as f:
        f.write("\\begin{tabular}{lccccccc}\n\\toprule\n")
        f.write("& \\multicolumn{3}{c}{Paid basis} & \\multicolumn{3}{c}{Incurred basis} & \\\\\n")
        f.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
        f.write("Line & $z^{raw}$ & $z$ & $g$ & $z^{raw}$ & $z$ & $g$ & $n$ \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(str(x) for x in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

main_table("A", "tab_main.tex")
main_table("B", "tab_mainB.tex")

# era-B covariate figure
ext_src = open("code/06_era_b_and_pooled.py").read()
# rebuild indB quickly from saved csv? recompute via harmonized loader:
exec(ext_src.split("# ------------------------------------------------ provenance diff")[0])
prem = (dB[dB["dev_lag"] == 1]
        .groupby(["line", "group_id", "acc_yr"])["net_ep"].first().reset_index())
import statsmodels.api as sm
ind = prem.groupby(["line", "acc_yr"])["net_ep"].sum().rename("ind_nep").reset_index()
ind["dlog"] = ind.groupby("line")["ind_nep"].transform(lambda s: np.log(s).diff())
ind["z_raw"] = ind.groupby("line")["dlog"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
ind["z_cyc"] = np.nan
for line_, g in ind.groupby("line"):
    sg = g.dropna(subset=["dlog"])
    X = sm.add_constant(sg["acc_yr"].astype(float))
    r = sm.OLS(sg["dlog"].to_numpy(), X).fit().resid
    ind.loc[sg.index, "z_cyc"] = (r - r.mean()) / r.std(ddof=1)

fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.4), sharex=True, sharey=True)
for ax, line in zip(axes.ravel(), MAIN):
    g = ind[ind["line"] == line].dropna(subset=["z_raw"])
    ax.axhline(0, color="0.75", lw=0.6)
    ax.plot(g["acc_yr"], g["z_raw"], "o-", color="0.55", ms=3, label="raw growth $z^{raw}$")
    ax.plot(g["acc_yr"], g["z_cyc"], "s--", color="black", ms=3, label="de-trended $z$")
    ax.set_title(NAMES[line], fontsize=9)
    ax.set_xticks([1999, 2001, 2003, 2005, 2007])
axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.supylabel("standardized covariate", fontsize=9)
fig.supxlabel("accident year", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_covariates_eraB.pdf", bbox_inches="tight")
plt.close(fig)

# extended robustness table: era-A battery + era-B rows
rob = pd.read_csv(f"{OUT}/robustness_zcyc.csv")
extra = [
    dict(check="era B (1998--2007), baseline", beta=0.0027, se=0.0035, p=0.4261, n=4389),
    dict(check="era B, calendar-year FE added", beta=0.0001, se=0.0046, p=0.9903, n=4389),
    dict(check="pooled eras (1988--2007)", beta=0.0016, se=0.0025, p=0.5098, n=8679),
]
with open(f"{TAB}/tab_robust.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
    f.write("Specification & $\\hat\\beta$ & s.e. & $p$ & $n$ \\\\\n\\midrule\n")
    f.write("\\multicolumn{5}{l}{\\emph{Era A (1988--1997)}} \\\\\n")
    for _, r in rob.iterrows():
        label = (str(r["check"]).replace(">=", "$\\geq$")
                 .replace("$1m", "\\$1m").replace("$5m", "\\$5m"))
        f.write(f"\\quad {label} & {r['beta']:.4f} & {r['se']:.4f} & "
                f"{r['p']:.3f} & {int(r['n'])} \\\\\n")
    f.write("\\midrule\n\\multicolumn{5}{l}{\\emph{Era B and pooled}} \\\\\n")
    for r in extra:
        f.write(f"\\quad {r['check']} & {r['beta']:.4f} & {r['se']:.4f} & "
                f"{r['p']:.3f} & {r['n']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# two-era panel table
keptA = {"comauto": 46, "wkcomp": 44, "othliab": 34, "prodliab": 6,
         "ppauto": 69, "medmal": 12}
keptB = {"comauto": 53, "wkcomp": 28, "othliab": 47, "prodliab": 5,
         "ppauto": 79, "medmal": 3}
totA = {"comauto": 158, "wkcomp": 132, "othliab": 239, "prodliab": 70,
        "ppauto": 146, "medmal": 34}
totB = {l: None for l in ORDER}
for l in ORDER:
    totB[l] = len(pd.read_csv(f"data/cas_current/{l}.csv")) // 100
with open(f"{TAB}/tab_panel.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
    f.write("& \\multicolumn{2}{c}{Era A (1988--1997)} & "
            "\\multicolumn{2}{c}{Era B (1998--2007)} \\\\\n")
    f.write("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n")
    f.write("Line & In database & Retained & In database & Retained \\\\\n\\midrule\n")
    for l in ORDER:
        f.write(f"{NAMES[l]} & {totA[l]} & {keptA[l]} & {totB[l]} & {keptB[l]} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# era-B OOS table
oos = pd.read_csv(f"{OUT}/oos_twoera.csv")
ob = oos[oos["era"] == "B"]
osum = ob.groupby(["basis", "line"]).apply(
    lambda g: pd.Series(dict(mse_cl=np.average(g["mse_cl"], weights=g["n"]),
                             mse_cy=np.average(g["mse_cy"], weights=g["n"]))),
    include_groups=False).reset_index()
osum["gain"] = 100 * (1 - osum["mse_cy"] / osum["mse_cl"])
with open(f"{TAB}/tab_oosB.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
    f.write("& \\multicolumn{3}{c}{Paid basis} & \\multicolumn{3}{c}{Incurred basis} \\\\\n")
    f.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
    f.write("Line & CL & covariate & gain (\\%) & CL & covariate & gain (\\%) \\\\\n\\midrule\n")
    for l in ORDER:
        gp = osum[(osum["line"] == l) & (osum["basis"] == "paid")]
        gi = osum[(osum["line"] == l) & (osum["basis"] == "incurred")]
        if gp.empty or gi.empty:
            continue
        gp, gi = gp.iloc[0], gi.iloc[0]
        f.write(f"{NAMES[l]} & {gp['mse_cl']*1000:.2f} & {gp['mse_cy']*1000:.2f} & "
                f"{gp['gain']:.1f} & {gi['mse_cl']*1000:.2f} & {gi['mse_cy']*1000:.2f} & "
                f"{gi['gain']:.1f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# early-booked table
eb = pd.read_csv(f"{OUT}/early_booked_errors.csv")
with open(f"{TAB}/tab_booked.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
    f.write("Era & Booked at lag & $\\hat\\beta$ & s.e. & $p$ & $n$ \\\\\n\\midrule\n")
    for _, r in eb.iterrows():
        f.write(f"{'A (1988--97)' if r['era']=='A' else 'B (1998--2007)'} & "
                f"{int(r['lag'])} & {r['beta']:.4f} & {r['se']:.4f} & "
                f"{r['p']:.3f} & {int(r['n'])} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print("era-B exhibits written to results/")
print("era B totals in db:", totB)
