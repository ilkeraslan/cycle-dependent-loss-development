"""Figure 1 (fig1_covariates) and Figure 3 (fig2_lag_profile) and the era-A tables (main estimates, out-of-sample test, panel), from results/."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"
FIG = "results/figures"
TAB = "results/tables"
import os
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1, "figure.dpi": 200,
})

LINE_NAMES = {"comauto": "Commercial auto", "wkcomp": "Workers' compensation",
              "othliab": "Other liability", "prodliab": "Products liability",
              "ppauto": "Private passenger auto", "medmal": "Medical malpractice"}
MAIN = ["comauto", "wkcomp", "othliab", "prodliab"]

# ---------------------------------------------------------------- figure 1
ind = pd.read_csv(f"{OUT}/cycle_covariate.csv")
fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.4), sharex=True, sharey=True)
for ax, line in zip(axes.ravel(), MAIN):
    g = ind[ind["line"] == line].dropna(subset=["z_raw"])
    ax.axhline(0, color="0.75", lw=0.6)
    ax.plot(g["acc_yr"], g["z_raw"], "o-", color="0.55", ms=3,
            label="raw growth $z^{raw}$")
    ax.plot(g["acc_yr"], g["z_cyc"], "s--", color="black", ms=3,
            label="de-trended $z$")
    ax.set_title(LINE_NAMES[line], fontsize=9)
    ax.set_xticks([1989, 1991, 1993, 1995, 1997])
axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.supylabel("standardized covariate", fontsize=9)
fig.supxlabel("accident year", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_covariates.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 2
prof = pd.read_csv(f"{OUT}/incurred_lag_profile.csv")
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.axhline(0, color="0.75", lw=0.6)
ax.errorbar(prof["lag"], prof["beta"], yerr=1.96 * prof["se"], fmt="o",
            color="black", ms=4, capsize=3, lw=1)
ax.set_xlabel("development lag $j$ (factor $j\\rightarrow j{+}1$)")
ax.set_ylabel(r"$\hat\beta_j$ on de-trended $z$")
ax.set_xticks(prof["lag"])
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_lag_profile.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- table: main
res = pd.read_csv(f"{OUT}/beta_estimates_era_a.csv")

def stars(p):
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.1 else ""

def cell(b, s, p):
    return f"{b:.4f}{stars(p)} ({s:.4f})"

order = MAIN + ["ppauto", "medmal"]
rows = []
for line in order:
    r = res[(res["line"] == line)]
    raw = r[(r["spec"] == "raw") & (r["var"] == "z_raw") & (r["basis"] == "paid")].iloc[0]
    cy_p = r[(r["spec"] == "cyc+rel") & (r["var"] == "z_cyc") & (r["basis"] == "paid")].iloc[0]
    gr_p = r[(r["spec"] == "cyc+rel") & (r["var"] == "g_rel") & (r["basis"] == "paid")].iloc[0]
    raw_i = r[(r["spec"] == "raw") & (r["var"] == "z_raw") & (r["basis"] == "incurred")].iloc[0]
    cy_i = r[(r["spec"] == "cyc+rel") & (r["var"] == "z_cyc") & (r["basis"] == "incurred")].iloc[0]
    gr_i = r[(r["spec"] == "cyc+rel") & (r["var"] == "g_rel") & (r["basis"] == "incurred")].iloc[0]
    rows.append((LINE_NAMES[line], cell(raw["beta"], raw["se"], raw["p"]),
                 cell(cy_p["beta"], cy_p["se"], cy_p["p"]),
                 cell(gr_p["beta"], gr_p["se"], gr_p["p"]),
                 cell(raw_i["beta"], raw_i["se"], raw_i["p"]),
                 cell(cy_i["beta"], cy_i["se"], cy_i["p"]),
                 cell(gr_i["beta"], gr_i["se"], gr_i["p"]),
                 int(cy_p["n"])))
pool_p = res[(res["line"] == "POOLED_MAIN") & (res["basis"] == "paid")]
pool_i = res[(res["line"] == "POOLED_MAIN") & (res["basis"] == "incurred")]
rows.append(("Pooled casualty lines", "--",
             cell(*pool_p[pool_p['var']=='z_cyc'][['beta','se','p']].iloc[0]),
             cell(*pool_p[pool_p['var']=='g_rel'][['beta','se','p']].iloc[0]),
             "--",
             cell(*pool_i[pool_i['var']=='z_cyc'][['beta','se','p']].iloc[0]),
             cell(*pool_i[pool_i['var']=='g_rel'][['beta','se','p']].iloc[0]),
             int(pool_p['n'].iloc[0])))

with open(f"{TAB}/tab_main.tex", "w") as f:
    f.write("\\begin{tabular}{lccccccc}\n\\toprule\n")
    f.write("& \\multicolumn{3}{c}{Paid basis} & \\multicolumn{3}{c}{Incurred basis} & \\\\\n")
    f.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
    f.write("Line & $z^{raw}$ & $z$ & $g$ & $z^{raw}$ & $z$ & $g$ & $n$ \\\\\n\\midrule\n")
    for r in rows:
        f.write(" & ".join(str(x) for x in r) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- table: OOS
osum = pd.read_csv(f"{OUT}/oos_summary_era_a.csv")
with open(f"{TAB}/tab_oos.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
    f.write("& \\multicolumn{3}{c}{Paid basis} & \\multicolumn{3}{c}{Incurred basis} \\\\\n")
    f.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
    f.write("Line & CL & covariate & gain (\\%) & CL & covariate & gain (\\%) \\\\\n\\midrule\n")
    for line in order:
        gp = osum[(osum["line"] == line) & (osum["basis"] == "paid")]
        gi = osum[(osum["line"] == line) & (osum["basis"] == "incurred")]
        if gp.empty or gi.empty:
            continue
        gp, gi = gp.iloc[0], gi.iloc[0]
        f.write(f"{LINE_NAMES[line]} & {gp['mse_cl']*1000:.2f} & {gp['mse_cy']*1000:.2f} & "
                f"{gp['gain_pct']:.1f} & {gi['mse_cl']*1000:.2f} & {gi['mse_cy']*1000:.2f} & "
                f"{gi['gain_pct']:.1f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- table: panel
scr = pd.read_csv(f"{OUT}/screening_log.csv")
ind_nep = ind.groupby("line")["ind_nep"].mean()
with open(f"{TAB}/tab_panel.tex", "w") as f:
    f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
    f.write("Line & Groups in database & Groups retained & Mean industry NEP (\\$m) \\\\\n\\midrule\n")
    for line in order:
        s = scr[scr["line"] == line].iloc[0]
        f.write(f"{LINE_NAMES[line]} & {int(s['total'])} & {int(s['keep'])} & "
                f"{ind_nep[line]/1000:,.0f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print("exhibits written")
print(open(f"{TAB}/tab_main.tex").read())
