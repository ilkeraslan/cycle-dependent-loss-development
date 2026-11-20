"""Era A: hierarchical Bayesian estimation, predictive calibration, reserve MSEP and robustness.

A. Hierarchical Bayesian model, estimated with a conjugate Gibbs sampler (exact updates):
   ln F = alpha_j + beta_j * z + gamma * g + line FE + u_company + eps,
   eps ~ N(0, sigma_j^2 / w), u_c ~ N(0, tau^2), beta_j ~ N(mu_beta, omega^2).
   Posterior means and standard deviations, compared with the WLS point estimates.
B. PIT calibration of one-year-ahead predictive distributions (paid basis).
C. Conditional MSEP of the aggregate 1997 reserve under the covariate model and under the
   cycle-blind log-normal chain ladder (process and parameter error, delta method).
D. Alternative de-trending (first difference; quadratic trend).
E. Leave-one-company-out for the other-liability result on g_rel.
F. Sensitivity to the screening threshold (median premium of at least USD 1 million or 5 million).
G. Calendar-year fixed effects for z_cyc.

The Gibbs sampler uses a fixed seed.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

rng = np.random.default_rng(20260716)

DATA = "data/cas_full_square.csv"
OUT = "results"
MAIN = ["comauto", "wkcomp", "othliab", "prodliab"]
ALL = MAIN + ["ppauto", "medmal"]
MAXLAG = 6
VAL = 1997

df_raw = pd.read_csv(DATA)

def build_panel(min_nep):
    def screen(g):
        full = len(g) == 100 and g["cum_paid_loss"].notna().all()
        nep = g.groupby("acc_yr")["net_ep"].first()
        ok = (full and (nep > 0).all() and nep.median() >= min_nep
              and (nep.std() / nep.mean() < 1.0 if nep.mean() > 0 else False)
              and (g.loc[g["dev_lag"] >= 1, "cum_paid_loss"] > 0).all()
              and g.loc[g["dev_lag"] == 10, "cum_paid_loss"].sum()
                  > g.loc[g["dev_lag"] == 1, "cum_paid_loss"].sum())
        return dict(keep=ok)
    flags = (df_raw.groupby(["line", "group_id", "company"])
             .apply(screen, include_groups=False).apply(pd.Series).reset_index())
    kept = flags.loc[flags["keep"], ["line", "group_id"]]
    d = df_raw.merge(kept, on=["line", "group_id"])
    prem = (d[d["dev_lag"] == 1]
            .groupby(["line", "group_id", "acc_yr"])["net_ep"].first().reset_index())
    ind = prem.groupby(["line", "acc_yr"])["net_ep"].sum().rename("ind_nep").reset_index()
    ind["dlog"] = ind.groupby("line")["ind_nep"].transform(lambda s: np.log(s).diff())
    for name, powers in [("z_cyc", [1]), ("z_quad", [1, 2])]:
        ind[name] = np.nan
        for line_, g in ind.groupby("line"):
            sg = g.dropna(subset=["dlog"])
            X = sm.add_constant(np.column_stack(
                [sg["acc_yr"].astype(float) ** p for p in powers]))
            r = sm.OLS(sg["dlog"].to_numpy(), X).fit().resid
            ind.loc[sg.index, name] = (r - r.mean()) / r.std(ddof=1)
    ind["z_fd"] = ind.groupby("line")["dlog"].transform(
        lambda s: (s.diff() - s.diff().mean()) / s.diff().std(ddof=1))
    prem = prem.merge(ind.drop(columns=["ind_nep"]), on=["line", "acc_yr"], how="left")
    prem["g_co"] = prem.groupby(["line", "group_id"])["net_ep"].transform(lambda s: np.log(s).diff())
    prem["g_rel_raw"] = prem["g_co"] - prem["dlog"]
    lo, hi = prem["g_rel_raw"].quantile([0.01, 0.99])
    prem["g_rel"] = prem.groupby("line")["g_rel_raw"].transform(
        lambda s: (s.clip(lo, hi) - s.clip(lo, hi).mean()) / s.clip(lo, hi).std(ddof=1))
    p = d.pivot_table(index=["line", "group_id", "acc_yr"], columns="dev_lag",
                      values="cum_paid_loss")
    rows = []
    for (line, gid, ay), r in p.iterrows():
        for j in range(1, 10):
            c0, c1 = r.get(j, np.nan), r.get(j + 1, np.nan)
            if np.isfinite(c0) and np.isfinite(c1) and c0 > 0 and c1 > 0:
                rows.append(dict(line=line, group_id=gid, acc_yr=ay, lag=j,
                                 cal_yr=ay + j, f=c1 / c0, w=c0))
    fac = pd.DataFrame(rows)
    fac["lnf"] = np.log(fac["f"])
    fac = fac.merge(prem[["line", "group_id", "acc_yr", "z_cyc", "z_quad",
                          "z_fd", "g_rel"]], on=["line", "group_id", "acc_yr"],
                    how="left")
    return d, fac, p, ind

d, fac, psq, ind = build_panel(2000)
est = fac[(fac["cal_yr"] <= VAL) & fac["z_cyc"].notna() & fac["g_rel"].notna()
          & (fac["lag"] <= MAXLAG) & fac["line"].isin(MAIN)].copy()

# ---------------------------------------- A. Gibbs sampler
s = est.reset_index(drop=True)
lags = np.sort(s["lag"].unique())
Xa = pd.get_dummies(s["lag"], prefix="a").astype(float)                # alpha_j
Xl = pd.get_dummies(s["line"], prefix="ln", drop_first=True).astype(float)
Xg = s[["g_rel"]].astype(float)
Xfix = pd.concat([Xa, Xl, Xg], axis=1).to_numpy()
Zb = (pd.get_dummies(s["lag"], prefix="b").astype(float).to_numpy()
      * s["z_cyc"].to_numpy()[:, None])                                # beta_j * z
comp_codes, comp_idx = np.unique(s["line"] + "_" + s["group_id"].astype(str),
                                 return_inverse=True)
lag_idx = s["lag"].to_numpy() - 1
y = s["lnf"].to_numpy()
w = s["w"].to_numpy()
n, kf, kb, nc = len(y), Xfix.shape[1], Zb.shape[1], len(comp_codes)

theta = np.zeros(kf); beta = np.zeros(kb); u = np.zeros(nc)
mu_b, om2, tau2 = 0.0, 0.01, 0.01
sig2 = np.full(len(lags), 0.05)
PRI_TH = 100.0; A0, B0 = 2.0, 1e-4

draws = {"beta": [], "mu_b": [], "gamma": [], "tau": [], "om": []}
NIT, BURN = 4000, 1500
for it in range(NIT):
    sw = w / sig2[lag_idx]                       # precision weights
    # fixed block
    r0 = y - Zb @ beta - u[comp_idx]
    XtW = Xfix.T * sw
    Vinv = XtW @ Xfix + np.eye(kf) / PRI_TH
    V = np.linalg.inv(Vinv)
    m = V @ (XtW @ r0)
    theta = rng.multivariate_normal(m, V)
    # beta block (hierarchical prior)
    r1 = y - Xfix @ theta - u[comp_idx]
    ZtW = Zb.T * sw
    Vinv = ZtW @ Zb + np.eye(kb) / om2
    V = np.linalg.inv(Vinv)
    m = V @ (ZtW @ r1 + mu_b / om2)
    beta = rng.multivariate_normal(m, V)
    # hyperpriors
    vb = 1.0 / (kb / om2 + 1.0)
    mu_b = rng.normal(vb * beta.sum() / om2, np.sqrt(vb))
    om2 = 1.0 / rng.gamma(A0 + kb / 2, 1.0 / (B0 + ((beta - mu_b) ** 2).sum() / 2))
    # company effects
    r2 = y - Xfix @ theta - Zb @ beta
    for c in range(nc):
        m_ = comp_idx == c
        prec = sw[m_].sum() + 1.0 / tau2
        mean = (sw[m_] * r2[m_]).sum() / prec
        u[c] = rng.normal(mean, np.sqrt(1.0 / prec))
    tau2 = 1.0 / rng.gamma(A0 + nc / 2, 1.0 / (B0 + (u ** 2).sum() / 2))
    # lag variances
    res = r2 - u[comp_idx]
    for jj, lg in enumerate(lags):
        m_ = lag_idx == jj
        sig2[jj] = 1.0 / rng.gamma(A0 + m_.sum() / 2,
                                   1.0 / (B0 + (w[m_] * res[m_] ** 2).sum() / 2))
    if it >= BURN:
        draws["beta"].append(beta.copy()); draws["mu_b"].append(mu_b)
        draws["gamma"].append(theta[-1]); draws["tau"].append(np.sqrt(tau2))
        draws["om"].append(np.sqrt(om2))

B = np.array(draws["beta"]); MU = np.array(draws["mu_b"]); G = np.array(draws["gamma"])
bayes = pd.DataFrame({
    "param": [f"beta_lag{j}" for j in lags] + ["mu_beta", "gamma", "tau", "omega"],
    "post_mean": list(B.mean(0)) + [MU.mean(), G.mean(),
                                    np.mean(draws["tau"]), np.mean(draws["om"])],
    "post_sd": list(B.std(0)) + [MU.std(), G.std(),
                                 np.std(draws["tau"]), np.std(draws["om"])],
    "p_neg": list((B < 0).mean(0)) + [(MU < 0).mean(), (G < 0).mean(),
                                      np.nan, np.nan]})
bayes.to_csv(f"{OUT}/bayes_posterior.csv", index=False)

# ---------------------------------------- B. PIT calibration
def pit_test(with_covs):
    pits = []
    for line in ALL:
        sub = fac[(fac["line"] == line) & fac["z_cyc"].notna() & fac["g_rel"].notna()
                  & (fac["lag"] <= MAXLAG)]
        for T in range(1994, 1998):
            tr, te = sub[sub["cal_yr"] <= T - 1], sub[sub["cal_yr"] == T]
            te = te[te["lag"].isin(tr["lag"].unique())]
            if len(tr) < 60 or len(te) < 10:
                continue
            Xd = pd.get_dummies(tr["lag"], prefix="lag").astype(float)
            X = pd.concat([Xd, tr[["z_cyc", "g_rel"]]], axis=1) if with_covs else Xd
            m = sm.WLS(tr["lnf"], X, weights=tr["w"]).fit()
            # lag-specific residual variance (per unit weight)
            rs = tr["lnf"] - m.fittedvalues
            s2 = {}
            for lg in Xd.columns:
                j = int(lg.split("_")[1])
                mm = tr["lag"] == j
                dof = max(mm.sum() - X.shape[1] / len(Xd.columns), 2)
                s2[j] = float((tr.loc[mm, "w"] * rs[mm] ** 2).sum() / dof)
            Xt = pd.get_dummies(te["lag"], prefix="lag").astype(float)
            Xt = Xt.reindex(columns=Xd.columns, fill_value=0.0)
            if with_covs:
                Xt = pd.concat([Xt, te[["z_cyc", "g_rel"]]], axis=1)
            mu = (Xt.to_numpy() @ m.params.to_numpy())
            var_par = np.einsum("ij,jk,ik->i", Xt.to_numpy(),
                                m.cov_params().to_numpy(), Xt.to_numpy())
            var = np.array([s2[j] for j in te["lag"]]) / te["w"].to_numpy() + var_par
            pits.append((te["lnf"].to_numpy() - mu) / np.sqrt(var))
    z = np.concatenate(pits)
    pit = stats.norm.cdf(z)
    ks = stats.kstest(pit, "uniform")
    dec = np.histogram(pit, bins=10, range=(0, 1))[0]
    return pit, ks, dec

pit_cy, ks_cy, dec_cy = pit_test(True)
pit_cl, ks_cl, dec_cl = pit_test(False)
pd.DataFrame({"decile": range(1, 11), "cl": dec_cl, "cov": dec_cy}).to_csv(
    f"{OUT}/pit_deciles.csv", index=False)

# ---------------------------------------- C. MSEP
def msep_line(line, with_covs):
    sub = fac[(fac["line"] == line) & (fac["cal_yr"] <= VAL)
              & fac["z_cyc"].notna() & fac["g_rel"].notna() & (fac["lag"] <= 9)]
    Xd = pd.get_dummies(sub["lag"], prefix="lag").astype(float)
    X = pd.concat([Xd, sub[["z_cyc", "g_rel"]]], axis=1) if with_covs else Xd
    m = sm.WLS(sub["lnf"], X, weights=sub["w"]).fit()
    rs = sub["lnf"] - m.fittedvalues
    s2_raw, n_j = {}, {}
    for col in Xd.columns:
        j = int(col.split("_")[1])
        mm = sub["lag"] == j
        s2_raw[j] = float((sub.loc[mm, "w"] * rs[mm] ** 2).sum() / max(mm.sum() - 1, 2))
        n_j[j] = int(mm.sum())
    # variance smoothing for sparse late lags: log-linear decay fitted on
    # well-populated lags (n_j >= 30), extrapolated where data are thin --
    # the standard remedy for tail-variance estimation (cf. Mack 1993, sec. 3)
    good = [j for j in s2_raw if n_j[j] >= 30 and s2_raw[j] > 0]
    if len(good) >= 3:
        bfit = np.polyfit(good, np.log([s2_raw[j] for j in good]), 1)
        s2 = {j: (s2_raw[j] if j in good else float(np.exp(np.polyval(bfit, j))))
              for j in range(1, 10)}
    else:
        s2 = s2_raw
    # industry-aggregate triangle for the line (sum of screened companies);
    # process variance propagates along the chain at the running expected
    # cumulative volume; parameter variance via delta method across the whole
    # reserve (cross-cohort parameter covariance included)
    zmap = ind[ind["line"] == line].set_index("acc_yr")["z_cyc"].to_dict()
    agg = psq.loc[line].groupby("acc_yr").sum(min_count=1)
    cov = m.cov_params().to_numpy()
    cols = list(X.columns)
    total_res, var_proc_tot, grad = 0.0, 0.0, np.zeros(len(cols))
    for ay, r in agg.iterrows():
        latest = VAL - ay + 1
        if latest >= 10 or latest < 1:
            continue
        c = r.get(latest, np.nan)
        if not (np.isfinite(c) and c > 0):
            continue
        zv = zmap.get(ay, 0.0)
        zv = 0.0 if (zv is None or np.isnan(zv)) else float(zv)
        c_run, mu_sum, vproc, xsum = c, 0.0, 0.0, np.zeros(len(cols))
        for j in range(latest, 10):
            xrow = np.zeros(len(cols))
            if f"lag_{j}" in cols:
                xrow[cols.index(f"lag_{j}")] = 1.0
            if with_covs:
                xrow[cols.index("z_cyc")] = zv
                # g_rel = 0 at industry level by construction
            mu_j = float(xrow @ m.params.to_numpy())
            vproc += s2.get(j, 0.0) / c_run
            c_run *= np.exp(mu_j + 0.5 * s2.get(j, 0.0) / c_run)
            mu_sum += mu_j
            xsum += xrow
        ult = c * np.exp(mu_sum + 0.5 * vproc)
        total_res += ult - c
        var_proc_tot += (ult ** 2) * (np.exp(vproc) - 1.0)
        grad += ult * xsum
    var_par_tot = float(grad @ cov @ grad)
    return total_res, np.sqrt(var_proc_tot + var_par_tot)

msep_rows = []
for line in MAIN:
    res_cl, se_cl = msep_line(line, False)
    res_cy, se_cy = msep_line(line, True)
    msep_rows.append(dict(line=line, reserve_cl=res_cl, se_cl=se_cl,
                          cv_cl=se_cl / res_cl, reserve_cy=res_cy, se_cy=se_cy,
                          cv_cy=se_cy / res_cy))
pd.DataFrame(msep_rows).to_csv(f"{OUT}/msep_summary.csv", index=False)

# ---------------------------------------- D to G. Robustness
def pooled_beta(data, zcol):
    ss = data.dropna(subset=[zcol, "g_rel"])
    Xd = pd.get_dummies(ss["lag"], prefix="lag").astype(float)
    Xl = pd.get_dummies(ss["line"], prefix="ln", drop_first=True).astype(float)
    X = pd.concat([Xd, Xl, ss[[zcol, "g_rel"]]], axis=1)
    m = sm.WLS(ss["lnf"], X, weights=ss["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": ss["line"] + ss["acc_yr"].astype(str)})
    return m.params[zcol], m.bse[zcol], m.pvalues[zcol], int(m.nobs)

rob = []
for zc, label in [("z_cyc", "linear detrend (baseline)"),
                  ("z_quad", "quadratic detrend"), ("z_fd", "first difference")]:
    b, se, p, n_ = pooled_beta(est, zc)
    rob.append(dict(check=label, beta=b, se=se, p=p, n=n_))

# calendar-year FE robustness
ss = est.dropna(subset=["z_cyc", "g_rel"])
Xd = pd.get_dummies(ss["lag"], prefix="lag").astype(float)
Xl = pd.get_dummies(ss["line"], prefix="ln", drop_first=True).astype(float)
Xc = pd.get_dummies(ss["cal_yr"], prefix="cy", drop_first=True).astype(float)
X = pd.concat([Xd, Xl, Xc, ss[["z_cyc", "g_rel"]]], axis=1)
m = sm.WLS(ss["lnf"], X, weights=ss["w"]).fit(
    cov_type="cluster", cov_kwds={"groups": ss["line"] + ss["acc_yr"].astype(str)})
rob.append(dict(check="calendar-year FE added", beta=m.params["z_cyc"],
                se=m.bse["z_cyc"], p=m.pvalues["z_cyc"], n=int(m.nobs)))

# screening sensitivity
for thr, label in [(1000, "median NEP >= $1m"), (5000, "median NEP >= $5m")]:
    _, fac_t, _, _ = build_panel(thr)
    est_t = fac_t[(fac_t["cal_yr"] <= VAL) & (fac_t["lag"] <= MAXLAG)
                  & fac_t["line"].isin(MAIN)]
    b, se, p, n_ = pooled_beta(est_t, "z_cyc")
    rob.append(dict(check=f"screen: {label}", beta=b, se=se, p=p, n=n_))
pd.DataFrame(rob).to_csv(f"{OUT}/robustness_zcyc.csv", index=False)

# LOCO for othliab g_rel (paid)
ol = est[est["line"] == "othliab"]
loco = []
for gid in ol["group_id"].unique():
    ss = ol[ol["group_id"] != gid]
    Xd = pd.get_dummies(ss["lag"], prefix="lag").astype(float)
    X = pd.concat([Xd, ss[["z_cyc", "g_rel"]]], axis=1)
    m = sm.WLS(ss["lnf"], X, weights=ss["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": ss["acc_yr"]})
    loco.append(dict(dropped=gid, beta=m.params["g_rel"], p=m.pvalues["g_rel"]))
loco = pd.DataFrame(loco)
loco.to_csv(f"{OUT}/loco_othliab.csv", index=False)

print("=== Bayes posterior (pooled main lines) ===")
print(bayes.round(4).to_string())
print("\n=== PIT (paid, one-year-ahead) ===")
print(f"CL: KS={ks_cl.statistic:.4f} p={ks_cl.pvalue:.4f} deciles={dec_cl.tolist()}")
print(f"COV: KS={ks_cy.statistic:.4f} p={ks_cy.pvalue:.4f} deciles={dec_cy.tolist()}")
print(f"n cells = {len(pit_cl)}")
print("\n=== MSEP (aggregate 1997 reserve, $000s) ===")
print(pd.DataFrame(msep_rows).round(3).to_string())
print("\n=== Robustness of pooled z_cyc ===")
print(pd.DataFrame(rob).round(4).to_string())
print("\n=== LOCO othliab g_rel ===")
print(f"beta range [{loco['beta'].min():.4f}, {loco['beta'].max():.4f}], "
      f"max p = {loco['p'].max():.4f}, all beta>0: {(loco['beta']>0).all()}")
