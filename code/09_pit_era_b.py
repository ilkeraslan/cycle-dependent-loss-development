"""Era B: PIT calibration of one-year-ahead predictive distributions."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

MAIN = ["comauto", "wkcomp", "othliab", "prodliab"]
LINES = MAIN + ["ppauto", "medmal"]
MAXLAG, MIN_NEP = 6, 2000

# rebuild the era-B factor table with the loader of script 06
exec(open("code/06_era_b_and_pooled.py").read().split(
    "# ------------------------------------------------ provenance diff")[0])

def build(d):
    def screen(g):
        full = len(g) == 100 and g["cum_paid"].notna().all()
        nep = g.groupby("acc_yr")["net_ep"].first()
        return dict(keep=(full and (nep > 0).all() and nep.median() >= MIN_NEP
                    and (nep.std() / nep.mean() < 1.0 if nep.mean() > 0 else False)
                    and (g.loc[g["dev_lag"] >= 1, "cum_paid"] > 0).all()
                    and g.loc[g["dev_lag"] == 10, "cum_paid"].sum()
                        > g.loc[g["dev_lag"] == 1, "cum_paid"].sum()))
    fl = (d.groupby(["line", "group_id", "company"])
          .apply(screen, include_groups=False).apply(pd.Series).reset_index())
    dk = d.merge(fl.loc[fl["keep"], ["line", "group_id"]], on=["line", "group_id"])
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
    prem = prem.merge(ind[["line", "acc_yr", "dlog", "z_cyc"]], on=["line", "acc_yr"], how="left")
    prem["g_co"] = prem.groupby(["line", "group_id"])["net_ep"].transform(lambda s: np.log(s).diff())
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
    return fac.merge(prem[["line", "group_id", "acc_yr", "z_cyc", "g_rel"]],
                     on=["line", "group_id", "acc_yr"], how="left")

facB = build(dB)

def pit_test(fac, tyears, with_covs):
    pits = []
    for line in LINES:
        sub = fac[(fac["line"] == line) & fac["z_cyc"].notna()
                  & fac["g_rel"].notna() & (fac["lag"] <= MAXLAG)]
        for T in tyears:
            tr, te = sub[sub["cal_yr"] <= T - 1], sub[sub["cal_yr"] == T]
            te = te[te["lag"].isin(tr["lag"].unique())]
            if len(tr) < 60 or len(te) < 10:
                continue
            Xd = pd.get_dummies(tr["lag"], prefix="lag").astype(float)
            X = pd.concat([Xd, tr[["z_cyc", "g_rel"]]], axis=1) if with_covs else Xd
            m = sm.WLS(tr["lnf"], X, weights=tr["w"]).fit()
            rs = tr["lnf"] - m.fittedvalues
            s2 = {}
            for lg in Xd.columns:
                j = int(lg.split("_")[1])
                mm = tr["lag"] == j
                s2[j] = float((tr.loc[mm, "w"] * rs[mm] ** 2).sum() / max(mm.sum() - 2, 2))
            Xt = pd.get_dummies(te["lag"], prefix="lag").astype(float)
            Xt = Xt.reindex(columns=Xd.columns, fill_value=0.0)
            if with_covs:
                Xt = pd.concat([Xt, te[["z_cyc", "g_rel"]]], axis=1)
            mu = Xt.to_numpy() @ m.params.to_numpy()
            vpar = np.einsum("ij,jk,ik->i", Xt.to_numpy(),
                             m.cov_params().to_numpy(), Xt.to_numpy())
            var = np.array([s2[j] for j in te["lag"]]) / te["w"].to_numpy() + vpar
            pits.append((te["lnf"].to_numpy() - mu) / np.sqrt(var))
    z = np.concatenate(pits)
    pit = stats.norm.cdf(z)
    ks = stats.kstest(pit, "uniform")
    dec = np.histogram(pit, bins=10, range=(0, 1))[0]
    return ks, dec, len(pit)

ks_cl, dec_cl, n_cl = pit_test(facB, range(2004, 2008), False)
ks_cy, dec_cy, n_cy = pit_test(facB, range(2004, 2008), True)
mid_cl = dec_cl[2:6].sum() / dec_cl.sum()
mid_cy = dec_cy[2:6].sum() / dec_cy.sum()
print(f"era-B PIT CL : KS={ks_cl.statistic:.4f} p={ks_cl.pvalue:.2e} "
      f"central4={mid_cl:.3f} n={n_cl}")
print(f"era-B PIT COV: KS={ks_cy.statistic:.4f} p={ks_cy.pvalue:.2e} "
      f"central4={mid_cy:.3f} n={n_cy}")
pd.DataFrame(dict(decile=range(1, 11), cl=dec_cl, cov=dec_cy)).to_csv(
    "results/pit_deciles_eraB.csv", index=False)
