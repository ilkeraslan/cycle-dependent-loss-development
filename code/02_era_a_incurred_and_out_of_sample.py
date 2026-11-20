"""Era A: incurred-loss development factors (case-reserve channel), the one-year-ahead
out-of-sample test of the next diagonal's development factors (fit on calendar years up to
T-1, scored on T, T = 1994-1997, paid and incurred bases) and the booked-error tests.
Writes to results/.
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

def screen(g):
    full = len(g) == 100 and g["cum_paid_loss"].notna().all()
    nep = g.groupby("acc_yr")["net_ep"].first()
    pos_prem = (nep > 0).all()
    med_ok = nep.median() >= MIN_MEDIAN_NEP
    stable = (nep.std() / nep.mean() < 1.0) if nep.mean() > 0 else False
    pos_paid = (g.loc[g["dev_lag"] >= 1, "cum_paid_loss"] > 0).all()
    dev_ok = (g.loc[g["dev_lag"] == 10, "cum_paid_loss"].sum()
              > g.loc[g["dev_lag"] == 1, "cum_paid_loss"].sum())
    return dict(keep=full and pos_prem and med_ok and stable and pos_paid and dev_ok)

flags = (df.groupby(["line", "group_id", "company"])
           .apply(screen, include_groups=False).apply(pd.Series).reset_index())
kept = flags.loc[flags["keep"], ["line", "group_id"]]
d = df.merge(kept, on=["line", "group_id"])

prem = (d[d["dev_lag"] == 1]
        .groupby(["line", "group_id", "acc_yr"])["net_ep"].first().reset_index())
ind = prem.groupby(["line", "acc_yr"])["net_ep"].sum().rename("ind_nep").reset_index()
ind["dlog"] = ind.groupby("line")["ind_nep"].transform(lambda s: np.log(s).diff())
ind["z_raw"] = ind.groupby("line")["dlog"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))

def detrend(g):
    s = g.dropna(subset=["dlog"]).copy()
    X = sm.add_constant(s["acc_yr"].astype(float))
    g.loc[s.index, "resid"] = sm.OLS(s["dlog"], X).fit().resid
    return g

ind = ind.groupby("line", group_keys=False).apply(detrend, include_groups=False
      ).join(ind[["line"]])
ind["z_cyc"] = ind.groupby("line")["resid"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))

prem = prem.merge(ind[["line", "acc_yr", "dlog", "z_raw", "z_cyc"]],
                  on=["line", "acc_yr"], how="left")
prem["g_co"] = prem.groupby(["line", "group_id"])["net_ep"].transform(lambda s: np.log(s).diff())
prem["g_rel_raw"] = prem["g_co"] - prem["dlog"]
lo, hi = prem["g_rel_raw"].quantile([0.01, 0.99])
prem["g_rel"] = prem.groupby("line")["g_rel_raw"].transform(
    lambda s: (s.clip(lo, hi) - s.clip(lo, hi).mean()) / s.clip(lo, hi).std(ddof=1))

def factor_table(value_col):
    p = d.pivot_table(index=["line", "group_id", "acc_yr"],
                      columns="dev_lag", values=value_col)
    rows = []
    for (line, gid, ay), r in p.iterrows():
        for j in range(1, 10):
            c0, c1 = r.get(j, np.nan), r.get(j + 1, np.nan)
            if np.isfinite(c0) and np.isfinite(c1) and c0 > 0 and c1 > 0:
                rows.append(dict(line=line, group_id=gid, acc_yr=ay, lag=j,
                                 cal_yr=ay + j, f=c1 / c0, w=c0))
    fac = pd.DataFrame(rows)
    fac["lnf"] = np.log(fac["f"])
    return fac.merge(prem[["line", "group_id", "acc_yr", "z_raw", "z_cyc", "g_rel"]],
                     on=["line", "group_id", "acc_yr"], how="left"), p

fac_paid, p_paid = factor_table("cum_paid_loss")
fac_inc, p_inc = factor_table("cum_caseinc_loss")

def run_specs(fac, label):
    est = fac[(fac["cal_yr"] <= VAL_YEAR) & fac["z_cyc"].notna()
              & fac["g_rel"].notna() & (fac["lag"] <= MAX_LAG_REG)].copy()
    rows = []
    for line in ALL_LINES:
        s = est[est["line"] == line]
        if s.empty:
            continue
        for name, xcols in [("raw", ["z_raw"]), ("cyc+rel", ["z_cyc", "g_rel"])]:
            Xd = pd.get_dummies(s["lag"], prefix="lag").astype(float)
            X = pd.concat([Xd, s[xcols]], axis=1)
            m = sm.WLS(s["lnf"], X, weights=s["w"]).fit(
                cov_type="cluster", cov_kwds={"groups": s["acc_yr"]})
            for c in xcols:
                rows.append(dict(basis=label, line=line, spec=name, var=c,
                                 beta=m.params[c], se=m.bse[c], p=m.pvalues[c],
                                 n=int(m.nobs)))
    # pooled main lines
    s = est[est["line"].isin(MAIN_LINES)]
    Xd = pd.get_dummies(s["lag"], prefix="lag").astype(float)
    Xl = pd.get_dummies(s["line"], prefix="ln", drop_first=True).astype(float)
    X = pd.concat([Xd, Xl, s[["z_cyc", "g_rel"]]], axis=1)
    m = sm.WLS(s["lnf"], X, weights=s["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": s["line"] + s["acc_yr"].astype(str)})
    for c in ["z_cyc", "g_rel"]:
        rows.append(dict(basis=label, line="POOLED_MAIN", spec="cyc+rel", var=c,
                         beta=m.params[c], se=m.bse[c], p=m.pvalues[c],
                         n=int(m.nobs)))
    return pd.DataFrame(rows), est

res_paid, est_paid = run_specs(fac_paid, "paid")
res_inc, est_inc = run_specs(fac_inc, "incurred")
res = pd.concat([res_paid, res_inc], ignore_index=True)
res.to_csv(f"{OUT}/beta_estimates_era_a.csv", index=False)

# ---------------- per-lag z_cyc profile on incurred, main lines pooled
s = est_inc[est_inc["line"].isin(MAIN_LINES)].copy()
prof_rows = []
for j in range(1, MAX_LAG_REG + 1):
    sj = s[s["lag"] == j]
    Xl = pd.get_dummies(sj["line"], prefix="ln", drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([Xl, sj[["z_cyc", "g_rel"]]], axis=1))
    m = sm.WLS(sj["lnf"], X, weights=sj["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": sj["line"] + sj["acc_yr"].astype(str)})
    prof_rows.append(dict(lag=j, beta=m.params["z_cyc"], se=m.bse["z_cyc"],
                          p=m.pvalues["z_cyc"], n=int(m.nobs)))
prof = pd.DataFrame(prof_rows)
prof.to_csv(f"{OUT}/incurred_lag_profile.csv", index=False)

# ---------------- fixed one-year-ahead OOS (both bases)
def oos_test(fac, label):
    rows = []
    for line in ALL_LINES:
        sub = fac[(fac["line"] == line) & fac["z_cyc"].notna()
                  & fac["g_rel"].notna() & (fac["lag"] <= MAX_LAG_REG)]
        for T in range(1994, 1998):
            tr, te = sub[sub["cal_yr"] <= T - 1], sub[sub["cal_yr"] == T]
            if len(tr) < 60 or len(te) < 10:
                continue
            agg = tr.assign(num=tr["f"] * tr["w"]).groupby("lag").agg(
                num=("num", "sum"), den=("w", "sum"))
            ldf_log = np.log(agg["num"] / agg["den"])
            Xd = pd.get_dummies(tr["lag"], prefix="lag").astype(float)
            X = pd.concat([Xd, tr[["z_cyc", "g_rel"]]], axis=1)
            m = sm.WLS(tr["lnf"], X, weights=tr["w"]).fit()
            Xt = pd.get_dummies(te["lag"], prefix="lag").astype(float)
            Xt = Xt.reindex(columns=Xd.columns, fill_value=0.0)
            Xt = pd.concat([Xt, te[["z_cyc", "g_rel"]]], axis=1)
            pred_cy = (Xt @ m.params).astype(float)
            pred_cl = te["lag"].map(ldf_log.to_dict()).astype(float)
            ok = pred_cl.notna()
            w = te.loc[ok, "w"]
            mse_cl = np.average((te.loc[ok, "lnf"] - pred_cl[ok]) ** 2, weights=w)
            mse_cy = np.average((te.loc[ok, "lnf"] - pred_cy[ok]) ** 2, weights=w)
            rows.append(dict(basis=label, line=line, T=T, n=int(ok.sum()),
                             mse_cl=mse_cl, mse_cy=mse_cy))
    o = pd.DataFrame(rows)
    osum = o.groupby(["basis", "line"]).apply(
        lambda g: pd.Series(dict(mse_cl=np.average(g["mse_cl"], weights=g["n"]),
                                 mse_cy=np.average(g["mse_cy"], weights=g["n"]),
                                 n=g["n"].sum())), include_groups=False)
    osum["gain_pct"] = 100 * (1 - osum["mse_cy"] / osum["mse_cl"])
    return o, osum

oos_p, osum_p = oos_test(fac_paid, "paid")
oos_i, osum_i = oos_test(fac_inc, "incurred")
pd.concat([oos_p, oos_i]).to_csv(f"{OUT}/oos_by_year_era_a.csv", index=False)
osum = pd.concat([osum_p, osum_i])
osum.to_csv(f"{OUT}/oos_summary_era_a.csv")

# ---------------- booked-ultimate errors vs covariates
bk = d[d["dev_yr"] == VAL_YEAR][["line", "group_id", "acc_yr", "booked_ult_loss"]]
act = d[d["dev_lag"] == 10][["line", "group_id", "acc_yr", "cum_paid_loss"]]\
        .rename(columns={"cum_paid_loss": "actual_ult"})
mot = bk.merge(act, on=["line", "group_id", "acc_yr"])
mot = mot.groupby(["line", "acc_yr"])[["booked_ult_loss", "actual_ult"]].sum().reset_index()
mot["err"] = mot["booked_ult_loss"] / mot["actual_ult"] - 1
mot = mot.merge(ind[["line", "acc_yr", "z_raw", "z_cyc"]], on=["line", "acc_yr"])
mot.to_csv(f"{OUT}/booked_ultimate_errors_era_a.csv", index=False)
mm = mot[mot["line"].isin(MAIN_LINES)].dropna(subset=["z_cyc"])
for var in ["z_raw", "z_cyc"]:
    Xl = pd.get_dummies(mm["line"], prefix="ln").astype(float)
    X = pd.concat([Xl, mm[[var]]], axis=1)
    m = sm.OLS(mm["err"], X).fit(cov_type="cluster", cov_kwds={"groups": mm["acc_yr"]})
    print(f"booked error on {var}: beta={m.params[var]:.4f} "
          f"se={m.bse[var]:.4f} p={m.pvalues[var]:.4f} n={int(m.nobs)}")

print("\nmain specs (both bases):\n",
      res[res["spec"].isin(["raw", "cyc+rel"])].round(4).to_string())
print("\nincurred per-lag z_cyc profile (pooled main lines):\n", prof.round(4).to_string())
print("\nOOS summary:\n", osum.round(5).to_string())
