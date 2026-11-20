"""Era B: calendar-year fixed effects, booked-error test at early valuations, Gibbs re-estimation
(fixed seed) and consistency checks."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(20260716)
OUT = "results"
MAIN = ["comauto", "wkcomp", "othliab", "prodliab"]
LINES = MAIN + ["ppauto", "medmal"]
MAXLAG, MIN_NEP = 6, 2000

import importlib.util
spec = importlib.util.spec_from_file_location("ext", "code/07_extension.py")

# --- rebuild era data with the same functions (import-free copy) ---
exec(open("code/06_era_b_and_pooled.py").read().split("# ------------------------------------------------ provenance diff")[0])
dk_store, prem_store, ind_store, fac_store = {}, {}, {}, {}
for d, era, val in [(dA, "A", 1997), (dB, "B", 2007)]:
    def screen(g):
        full = len(g) == 100 and g["cum_paid"].notna().all()
        nep = g.groupby("acc_yr")["net_ep"].first()
        ok = (full and (nep > 0).all() and nep.median() >= MIN_NEP
              and (nep.std() / nep.mean() < 1.0 if nep.mean() > 0 else False)
              and (g.loc[g["dev_lag"] >= 1, "cum_paid"] > 0).all()
              and g.loc[g["dev_lag"] == 10, "cum_paid"].sum()
                  > g.loc[g["dev_lag"] == 1, "cum_paid"].sum())
        return dict(keep=ok)
    fl = (d.groupby(["line", "group_id", "company"])
          .apply(screen, include_groups=False).apply(pd.Series).reset_index())
    kept = fl.loc[fl["keep"], ["line", "group_id"]]
    dk = d.merge(kept, on=["line", "group_id"])
    prem = (dk[dk["dev_lag"] == 1]
            .groupby(["line", "group_id", "acc_yr"])["net_ep"].first().reset_index())
    ind = prem.groupby(["line", "acc_yr"])["net_ep"].sum().rename("ind_nep").reset_index()
    ind["dlog"] = ind.groupby("line")["ind_nep"].transform(lambda s: np.log(s).diff())
    ind["z_cyc"] = np.nan
    for line_, g in ind.groupby("line"):
        sg = g.dropna(subset=["dlog"])
        X = sm.add_constant(sg["acc_yr"].astype(float))
        r = sm.OLS(sg["dlog"].to_numpy(), X).fit().resid
        ind.loc[sg.index, "z_cyc"] = (r - r.mean()) / r.std(ddof=1)
    prem = prem.merge(ind[["line", "acc_yr", "dlog", "z_cyc"]],
                      on=["line", "acc_yr"], how="left")
    prem["g_co"] = prem.groupby(["line", "group_id"])["net_ep"].transform(
        lambda s: np.log(s).diff())
    prem["g_rel_raw"] = prem["g_co"] - prem["dlog"]
    lo, hi = prem["g_rel_raw"].quantile([0.01, 0.99])
    prem["g_rel"] = prem.groupby("line")["g_rel_raw"].transform(
        lambda s: (s.clip(lo, hi) - s.clip(lo, hi).mean()) / s.clip(lo, hi).std(ddof=1))
    p = dk.pivot_table(index=["line", "group_id", "acc_yr"], columns="dev_lag",
                       values="cum_paid")
    rows = []
    for (line, gid, ay), r in p.iterrows():
        for j in range(1, 10):
            c0, c1 = r.get(j, np.nan), r.get(j + 1, np.nan)
            if np.isfinite(c0) and np.isfinite(c1) and c0 > 0 and c1 > 0:
                rows.append(dict(line=line, group_id=gid, acc_yr=ay, lag=j,
                                 cal_yr=ay + j, f=c1 / c0, w=c0))
    fac = pd.DataFrame(rows)
    fac["lnf"] = np.log(fac["f"])
    fac = fac.merge(prem[["line", "group_id", "acc_yr", "z_cyc", "g_rel"]],
                    on=["line", "group_id", "acc_yr"], how="left")
    dk_store[era], prem_store[era], ind_store[era], fac_store[era] = dk, prem, ind, fac

# --- 1. calendar-year FE robustness, era B, paid ---
val = 2007
s = fac_store["B"]
s = s[(s["cal_yr"] <= val) & (s["lag"] <= MAXLAG) & s["line"].isin(MAIN)].dropna(
    subset=["z_cyc", "g_rel"])
Xd = pd.get_dummies(s["lag"], prefix="lag").astype(float)
Xl = pd.get_dummies(s["line"], prefix="ln", drop_first=True).astype(float)
Xc = pd.get_dummies(s["cal_yr"], prefix="cy", drop_first=True).astype(float)
for label, Xx in [("era B pooled paid, baseline", pd.concat([Xd, Xl], axis=1)),
                  ("era B pooled paid, +CY FE", pd.concat([Xd, Xl, Xc], axis=1))]:
    X = pd.concat([Xx, s[["z_cyc", "g_rel"]]], axis=1)
    m = sm.WLS(s["lnf"], X, weights=s["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": s["line"] + s["acc_yr"].astype(str)})
    print(f"{label}: z_cyc={m.params['z_cyc']:.4f} ({m.bse['z_cyc']:.4f}) "
          f"p={m.pvalues['z_cyc']:.4f}; g_rel={m.params['g_rel']:.4f} "
          f"({m.bse['g_rel']:.4f}) p={m.pvalues['g_rel']:.4f}")

# --- 2. early-valuation booked-ultimate errors (lags 2 and 3), both eras ---
print("\nEarly booked-ultimate errors vs cycle deviation (main lines, line FE):")
er_rows = []
for era, valy in [("A", 1997), ("B", 2007)]:
    dk, ind = dk_store[era], ind_store[era]
    act = dk[dk["dev_lag"] == 10][["line", "group_id", "acc_yr", "cum_paid"]]\
        .rename(columns={"cum_paid": "actual_ult"})
    for early in [2, 3]:
        bk = dk[dk["dev_lag"] == early][["line", "group_id", "acc_yr", "incurred"]]
        mot = bk.merge(act, on=["line", "group_id", "acc_yr"])
        mot = mot[mot["acc_yr"] + early - 1 <= valy]   # booked at that lag observed
        mot = mot.groupby(["line", "acc_yr"])[["incurred", "actual_ult"]].sum().reset_index()
        mot = mot[mot["actual_ult"] > 0]
        mot["err"] = mot["incurred"] / mot["actual_ult"] - 1
        mot = mot.merge(ind[["line", "acc_yr", "z_cyc"]], on=["line", "acc_yr"])
        mm = mot[mot["line"].isin(MAIN)].dropna(subset=["z_cyc"])
        Xl = pd.get_dummies(mm["line"], prefix="ln").astype(float)
        X = pd.concat([Xl, mm[["z_cyc"]]], axis=1)
        m = sm.OLS(mm["err"], X).fit(cov_type="cluster",
                                     cov_kwds={"groups": mm["acc_yr"]})
        print(f"  era {era}, booked at lag {early}: beta={m.params['z_cyc']:.4f} "
              f"se={m.bse['z_cyc']:.4f} p={m.pvalues['z_cyc']:.4f} n={int(m.nobs)}")
        er_rows.append(dict(era=era, lag=early, beta=m.params["z_cyc"],
                            se=m.bse["z_cyc"], p=m.pvalues["z_cyc"], n=int(m.nobs)))
pd.DataFrame(er_rows).to_csv(f"{OUT}/early_booked_errors.csv", index=False)

# --- 3. Gibbs on era B pooled paid ---
s = s.reset_index(drop=True)
lags = np.sort(s["lag"].unique())
Xa = pd.get_dummies(s["lag"], prefix="a").astype(float)
Xl = pd.get_dummies(s["line"], prefix="ln", drop_first=True).astype(float)
Xfix = pd.concat([Xa, Xl, s[["g_rel"]]], axis=1).to_numpy()
Zb = (pd.get_dummies(s["lag"], prefix="b").astype(float).to_numpy()
      * s["z_cyc"].to_numpy()[:, None])
comp_codes, comp_idx = np.unique(s["line"] + "_" + s["group_id"].astype(str),
                                 return_inverse=True)
lag_idx = s["lag"].to_numpy() - 1
y, w = s["lnf"].to_numpy(), s["w"].to_numpy()
n, kf, kb, nc = len(y), Xfix.shape[1], Zb.shape[1], len(comp_codes)
theta, beta, u = np.zeros(kf), np.zeros(kb), np.zeros(nc)
mu_b, om2, tau2 = 0.0, 0.01, 0.01
sig2 = np.full(len(lags), 0.05)
A0, B0, PRI = 2.0, 1e-4, 100.0
keep = {"mu": [], "beta": []}
for it in range(4000):
    sw = w / sig2[lag_idx]
    r0 = y - Zb @ beta - u[comp_idx]
    XtW = Xfix.T * sw
    V = np.linalg.inv(XtW @ Xfix + np.eye(kf) / PRI)
    theta = rng.multivariate_normal(V @ (XtW @ r0), V)
    r1 = y - Xfix @ theta - u[comp_idx]
    ZtW = Zb.T * sw
    V = np.linalg.inv(ZtW @ Zb + np.eye(kb) / om2)
    beta = rng.multivariate_normal(V @ (ZtW @ r1 + mu_b / om2), V)
    vb = 1.0 / (kb / om2 + 1.0)
    mu_b = rng.normal(vb * beta.sum() / om2, np.sqrt(vb))
    om2 = 1.0 / rng.gamma(A0 + kb / 2, 1.0 / (B0 + ((beta - mu_b) ** 2).sum() / 2))
    r2 = y - Xfix @ theta - Zb @ beta
    for c in range(nc):
        mm_ = comp_idx == c
        prec = sw[mm_].sum() + 1.0 / tau2
        u[c] = rng.normal((sw[mm_] * r2[mm_]).sum() / prec, np.sqrt(1.0 / prec))
    tau2 = 1.0 / rng.gamma(A0 + nc / 2, 1.0 / (B0 + (u ** 2).sum() / 2))
    res = r2 - u[comp_idx]
    for jj in range(len(lags)):
        mm_ = lag_idx == jj
        sig2[jj] = 1.0 / rng.gamma(A0 + mm_.sum() / 2,
                                   1.0 / (B0 + (w[mm_] * res[mm_] ** 2).sum() / 2))
    if it >= 1500:
        keep["mu"].append(mu_b); keep["beta"].append(beta.copy())
MU = np.array(keep["mu"])
print(f"\nGibbs era B pooled paid: mu_beta post mean {MU.mean():.4f} "
      f"sd {MU.std():.4f} Pr(<0)={(MU<0).mean():.2f}")
with open(f"{OUT}/bayes_eraB.txt", "w") as fh:
    fh.write(f"mu_beta,{MU.mean()},{MU.std()},{(MU<0).mean()}\n")

# --- 4. consistency: era A pooled from harmonized casact files ---
res = pd.read_csv(f"{OUT}/beta_estimates_twoera.csv")
ra = res[(res["era"] == "A") & (res["line"] == "POOLED_MAIN") & (res["basis"] == "paid")]
print("\nera A pooled (from first-hand casact files):")
print(ra.round(4).to_string(index=False))
print("\npooled two-era file:", open(f"{OUT}/pooled_twoera.txt").read())
