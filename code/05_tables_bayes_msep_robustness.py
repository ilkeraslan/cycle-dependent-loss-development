"""Tables of the Bayesian estimates, robustness checks and reserve MSEP, from results/."""
import pandas as pd

TAB = "results/tables"

bayes = pd.read_csv("results/bayes_posterior.csv")
lab = {"beta_lag1": r"$\beta_1$", "beta_lag2": r"$\beta_2$",
       "beta_lag3": r"$\beta_3$", "beta_lag4": r"$\beta_4$",
       "beta_lag5": r"$\beta_5$", "beta_lag6": r"$\beta_6$",
       "mu_beta": r"$\mu_\beta$", "gamma": r"$\gamma$",
       "tau": r"$\tau$", "omega": r"$\omega_\beta$"}
with open(f"{TAB}/tab_bayes.tex", "w") as f:
    f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
    f.write("Parameter & Posterior mean & Posterior s.d. & $\\Pr(\\cdot<0)$ \\\\\n\\midrule\n")
    for _, r in bayes.iterrows():
        p = "--" if pd.isna(r["p_neg"]) else f"{r['p_neg']:.2f}"
        f.write(f"{lab[r['param']]} & {r['post_mean']:.4f} & {r['post_sd']:.4f} & {p} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

rob = pd.read_csv("results/robustness_zcyc.csv")
with open(f"{TAB}/tab_robust.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
    f.write("Specification & $\\hat\\beta$ & s.e. & $p$ & $n$ \\\\\n\\midrule\n")
    for _, r in rob.iterrows():
        label = (str(r["check"]).replace(">=", "$\\geq$")
                 .replace("$1m", "\\$1m").replace("$5m", "\\$5m"))
        f.write(f"{label} & {r['beta']:.4f} & {r['se']:.4f} & "
                f"{r['p']:.3f} & {int(r['n'])} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

ms = pd.read_csv("results/msep_summary.csv")
names = {"comauto": "Commercial auto", "wkcomp": "Workers' compensation",
         "othliab": "Other liability", "prodliab": "Products liability"}
with open(f"{TAB}/tab_msep.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
    f.write("& \\multicolumn{3}{c}{Cycle-blind log-normal CL} & "
            "\\multicolumn{3}{c}{Covariate model} \\\\\n")
    f.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
    f.write("Line & Reserve & $\\sqrt{\\mathrm{MSEP}}$ & CV & "
            "Reserve & $\\sqrt{\\mathrm{MSEP}}$ & CV \\\\\n\\midrule\n")
    for _, r in ms.iterrows():
        f.write(f"{names[r['line']]} & {r['reserve_cl']/1000:,.0f} & "
                f"{r['se_cl']/1000:,.0f} & {r['cv_cl']:.3f} & "
                f"{r['reserve_cy']/1000:,.0f} & {r['se_cy']/1000:,.0f} & "
                f"{r['cv_cy']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

pit = pd.read_csv("results/pit_deciles.csv")
n = pit[["cl", "cov"]].sum()
print("PIT shares CL :", (pit["cl"] / n["cl"]).round(3).tolist())
print("PIT shares COV:", (pit["cov"] / n["cov"]).round(3).tolist())
loco = pd.read_csv("results/loco_othliab.csv")
print("LOCO beta:", loco["beta"].min().round(4), loco["beta"].max().round(4),
      "max p:", loco["p"].max().round(4))
print("tables written")
