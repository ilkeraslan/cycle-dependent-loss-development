"""Era A (CAS accident years 1988-1997, valuation 1997): screening of company groups,
cycle covariates and paid-loss development regressions.

- screening: complete 10 x 10 triangles, positive premium, median net earned premium of at
  least USD 2 million, stable premium, positive paid losses
- covariates: z_raw, standardised log growth of industry net earned premium; z_cyc, z_raw
  de-trended within line (residual of dlog premium on a linear accident-year trend);
  g_rel, company premium growth less line premium growth (winsorised at 1% and 99%)
- placebo: private passenger auto should show no effect of z_cyc
- booked-ultimate errors (booked ultimate at year-end 1997 against paid losses at lag 10)
  regressed on the cycle covariate

Reads data/cas_full_square.csv (built by prepare_data.py); writes to results/.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "data/cas_full_square.csv"
OUT = "results"
MAIN_LINES = ["comauto", "wkcomp", "othliab", "prodliab"]
ALL_LINES = MAIN_LINES + ["ppauto", "medmal"]
MAX_LAG_REG = 6
MIN_MEDIAN_NEP = 2000
VAL_YEAR = 1997

df = pd.read_csv(DATA)

# ---------------------------------------------------------------- screening
def screen(g):
    full = len(g) == 100 and g["cum_paid_loss"].notna().all()
    nep = g.groupby("acc_yr")["net_ep"].first()
    pos_prem = (nep > 0).all()
    med_ok = nep.median() >= MIN_MEDIAN_NEP
    stable = (nep.std() / nep.mean() < 1.0) if nep.mean() > 0 else False
    pos_paid = (g.loc[g["dev_lag"] >= 1, "cum_paid_loss"] > 0).all()
    dev_ok = (g.loc[g["dev_lag"] == 10, "cum_paid_loss"].sum()
              > g.loc[g["dev_lag"] == 1, "cum_paid_loss"].sum())
    return dict(full=full, pos_prem=pos_prem, med_ok=med_ok, stable=stable,
                pos_paid=pos_paid, dev_ok=dev_ok,
                keep=full and pos_prem and med_ok and stable and pos_paid and dev_ok)

flags = (df.groupby(["line", "group_id", "company"])
           .apply(screen, include_groups=False).apply(pd.Series).reset_index())
screen_log = flags.groupby("line")[["full", "pos_prem", "med_ok", "stable",
                                    "pos_paid", "dev_ok", "keep"]].sum()
screen_log["total"] = flags.groupby("line").size()
screen_log.to_csv(f"{OUT}/screening_log.csv")
kept = flags.loc[flags["keep"], ["line", "group_id"]]
d = df.merge(kept, on=["line", "group_id"])

# ----------------------------------------------------------- covariates
prem = (d[d["dev_lag"] == 1]
        .groupby(["line", "group_id", "acc_yr"])["net_ep"].first().reset_index())
ind = prem.groupby(["line", "acc_yr"])["net_ep"].sum().rename("ind_nep").reset_index()
ind["dlog"] = ind.groupby("line")["ind_nep"].transform(lambda s: np.log(s).diff())
ind["z_raw"] = ind.groupby("line")["dlog"].transform(
    lambda s: (s - s.mean()) / s.std(ddof=1))

def detrend(g):
    s = g.dropna(subset=["dlog"]).copy()
    X = sm.add_constant(s["acc_yr"].astype(float))
    r = sm.OLS(s["dlog"], X).fit()
    g.loc[s.index, "resid"] = r.resid
    return g

ind = ind.groupby("line", group_keys=False).apply(detrend, include_groups=False
      ).join(ind[["line"]])
ind["z_cyc"] = ind.groupby("line")["resid"].transform(
    lambda s: (s - s.mean()) / s.std(ddof=1))
ind.to_csv(f"{OUT}/cycle_covariate.csv", index=False)

prem = prem.merge(ind[["line", "acc_yr", "dlog", "z_raw", "z_cyc"]],
                  on=["line", "acc_yr"], how="left")
prem["g_co"] = prem.groupby(["line", "group_id"])["net_ep"].transform(
    lambda s: np.log(s).diff())
prem["g_rel_raw"] = prem["g_co"] - prem["dlog"]
lo, hi = prem["g_rel_raw"].quantile([0.01, 0.99])
prem["g_rel_w"] = prem["g_rel_raw"].clip(lo, hi)
prem["g_rel"] = prem.groupby("line")["g_rel_w"].transform(
    lambda s: (s - s.mean()) / s.std(ddof=1))

# ------------------------------------------------------ development factors
p = d.pivot_table(index=["line", "group_id", "acc_yr"],
                  columns="dev_lag", values="cum_paid_loss")
rows = []
for (line, gid, ay), r in p.iterrows():
    for j in range(1, 10):
        c0, c1 = r.get(j, np.nan), r.get(j + 1, np.nan)
        if np.isfinite(c0) and np.isfinite(c1) and c0 > 0 and c1 > 0:
            rows.append(dict(line=line, group_id=gid, acc_yr=ay, lag=j,
                             cal_yr=ay + j, f=c1 / c0, w=c0))
fac = pd.DataFrame(rows)
fac["lnf"] = np.log(fac["f"])
fac = fac.merge(prem[["line", "group_id", "acc_yr", "z_raw", "z_cyc", "g_rel"]],
                on=["line", "group_id", "acc_yr"], how="left")

est = fac[(fac["cal_yr"] <= VAL_YEAR) & fac["z_cyc"].notna()
          & (fac["lag"] <= MAX_LAG_REG)].copy()

# --------------------------------------------------------------- regressions
def run_wls(s, xcols, cluster):
    Xd = pd.get_dummies(s["lag"], prefix="lag").astype(float)
    X = pd.concat([Xd, s[xcols]], axis=1)
    return sm.WLS(s["lnf"], X, weights=s["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": cluster})

spec_rows = []
for line in ALL_LINES:
    s = est[(est["line"] == line) & est["g_rel"].notna()].copy()
    if s.empty:
        continue
    for name, xcols in [("raw", ["z_raw"]), ("cyc", ["z_cyc"]),
                        ("cyc+rel", ["z_cyc", "g_rel"])]:
        m = run_wls(s, xcols, s["acc_yr"])
        for c in xcols:
            spec_rows.append(dict(line=line, spec=name, var=c,
                                  beta=m.params[c], se=m.bse[c],
                                  p=m.pvalues[c], n=int(m.nobs),
                                  n_ay=s["acc_yr"].nunique()))
    # per-lag z_cyc profile
    for j in sorted(s["lag"].unique()):
        sj = s[s["lag"] == j]
        if sj["acc_yr"].nunique() < 5:
            continue
        X = sm.add_constant(sj[["z_cyc", "g_rel"]])
        m = sm.WLS(sj["lnf"], X, weights=sj["w"]).fit(
            cov_type="cluster", cov_kwds={"groups": sj["acc_yr"]})
        spec_rows.append(dict(line=line, spec=f"lag{j}", var="z_cyc",
                              beta=m.params["z_cyc"], se=m.bse["z_cyc"],
                              p=m.pvalues["z_cyc"], n=int(m.nobs),
                              n_ay=sj["acc_yr"].nunique()))
        spec_rows.append(dict(line=line, spec=f"lag{j}", var="g_rel",
                              beta=m.params["g_rel"], se=m.bse["g_rel"],
                              p=m.pvalues["g_rel"], n=int(m.nobs),
                              n_ay=sj["acc_yr"].nunique()))
res = pd.DataFrame(spec_rows)
res.to_csv(f"{OUT}/beta_estimates_paid_era_a.csv", index=False)

# pooled across the four main casualty lines (line FE), placebo separately
s = est[est["line"].isin(MAIN_LINES) & est["g_rel"].notna()].copy()
Xd = pd.get_dummies(s["lag"], prefix="lag").astype(float)
Xl = pd.get_dummies(s["line"], prefix="ln", drop_first=True).astype(float)
X = pd.concat([Xd, Xl, s[["z_cyc", "g_rel"]]], axis=1)
m_pool = sm.WLS(s["lnf"], X, weights=s["w"]).fit(
    cov_type="cluster", cov_kwds={"groups": s["line"] + s["acc_yr"].astype(str)})
pooled = pd.DataFrame(dict(beta=m_pool.params[["z_cyc", "g_rel"]],
                           se=m_pool.bse[["z_cyc", "g_rel"]],
                           p=m_pool.pvalues[["z_cyc", "g_rel"]]))
pooled["n"] = int(m_pool.nobs)
pooled.to_csv(f"{OUT}/pooled_main_lines_paid_era_a.csv")

# --------------------------------------------- booked-ultimate motivating fact
bk = d[d["dev_yr"] == VAL_YEAR][["line", "group_id", "acc_yr", "booked_ult_loss"]]
act = d[d["dev_lag"] == 10][["line", "group_id", "acc_yr", "cum_paid_loss"]]\
        .rename(columns={"cum_paid_loss": "actual_ult"})
mot = bk.merge(act, on=["line", "group_id", "acc_yr"])
mot = mot.groupby(["line", "acc_yr"])[["booked_ult_loss", "actual_ult"]].sum().reset_index()
mot["err"] = mot["booked_ult_loss"] / mot["actual_ult"] - 1
mot = mot.merge(ind[["line", "acc_yr", "z_raw", "z_cyc"]], on=["line", "acc_yr"])
mm = mot[mot["line"].isin(MAIN_LINES)].dropna(subset=["z_cyc"])
Xl = pd.get_dummies(mm["line"], prefix="ln").astype(float)
X = pd.concat([Xl, mm[["z_cyc"]]], axis=1)
m_mot = sm.OLS(mm["err"], X).fit(cov_type="cluster",
                                 cov_kwds={"groups": mm["acc_yr"]})
print("BOOKED-ULT ERROR on z_cyc (main lines, line FE):",
      f"beta={m_mot.params['z_cyc']:.4f} se={m_mot.bse['z_cyc']:.4f} "
      f"p={m_mot.pvalues['z_cyc']:.4f} n={int(m_mot.nobs)}")

print("\nscreening:\n", screen_log)
print("\nkept:", kept.groupby("line").size().to_dict())
print("\nz_cyc by line/AY:\n",
      ind.pivot_table(index="acc_yr", columns="line", values="z_cyc").round(2))
print("\nmain specs:\n",
      res[res["spec"].isin(["raw", "cyc", "cyc+rel"])].round(4).to_string())
print("\npooled main lines (z_cyc, g_rel):\n", pooled.round(4).to_string())
print("\nper-lag z_cyc profile (main lines):\n",
      res[(res["spec"].str.startswith("lag")) & (res["var"] == "z_cyc")
          & (res["line"].isin(MAIN_LINES))].round(4).to_string())
