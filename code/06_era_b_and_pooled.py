"""Era B (CAS accident years 1998-2007, valuation 2007) and the pooled two-era analysis.
Both eras have realised run-off to lag 10.

Consistency check: the era-A panel loaded here from data/cas_earlier/ is compared with
data/cas_full_square.csv, the flat file used by scripts 01 to 04 (results/provenance_diff.csv).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(20260716)
OUT = "results"
LINES = ["comauto", "wkcomp", "othliab", "prodliab", "ppauto", "medmal"]
MAIN = ["comauto", "wkcomp", "othliab", "prodliab"]
MAXLAG = 6
MIN_NEP = 2000

# ------------------------------------------------ load and harmonize
def load_era(folder, era, val_year):
    frames = []
    for line in LINES:
        f = pd.read_csv(f"data/{folder}/{line}.csv")
        f.columns = [c.strip() for c in f.columns]
        ren = {}
        for c in f.columns:
            cl = c.lower()
            if cl.startswith("incur"):
                ren[c] = "incurred"
            elif cl.startswith("cumpaid"):
                ren[c] = "cum_paid"
            elif cl.startswith("bulk"):
                ren[c] = "bulk"
            elif cl.startswith("earnedpremdir"):
                ren[c] = "ep_dir"
            elif cl.startswith("earnedpremceded"):
                ren[c] = "ep_ceded"
            elif cl.startswith("earnedpremnet"):
                ren[c] = "net_ep"
            elif cl.startswith("postedreserve"):
                ren[c] = "posted"
        f = f.rename(columns=ren)
        f["line"], f["era"] = line, era
        frames.append(f[["line", "era", "GRCODE", "GRNAME", "AccidentYear",
                         "DevelopmentYear", "DevelopmentLag", "incurred",
                         "cum_paid", "bulk", "net_ep"]])
    d = pd.concat(frames, ignore_index=True).rename(columns={
        "GRCODE": "group_id", "GRNAME": "company", "AccidentYear": "acc_yr",
        "DevelopmentYear": "dev_yr", "DevelopmentLag": "dev_lag"})
    d["case_inc"] = d["incurred"] - d["bulk"]
    d["val_year"] = val_year
    return d

dA = load_era("cas_earlier", "A", 1997)
dB = load_era("cas_current", "B", 2007)

# ------------------------------------------------ provenance diff (era A)
mirror = pd.read_csv("data/cas_full_square.csv")
prov = []
for line in LINES:
    a = dA[dA["line"] == line]
    m = mirror[mirror["line"] == line]
    prov.append(dict(
        line=line,
        n_records_casact=len(a), n_records_mirror=len(m),
        paid_sum_casact=a["cum_paid"].sum(), paid_sum_mirror=m["cum_paid_loss"].sum(),
        caseinc_sum_casact=a["case_inc"].sum(),
        caseinc_sum_mirror=m["cum_caseinc_loss"].sum(),
        nep_sum_casact=a["net_ep"].sum(), nep_sum_mirror=m["net_ep"].sum()))
prov = pd.DataFrame(prov)
prov["paid_match"] = prov["paid_sum_casact"] == prov["paid_sum_mirror"]
prov["caseinc_match"] = prov["caseinc_sum_casact"] == prov["caseinc_sum_mirror"]
prov["nep_match"] = prov["nep_sum_casact"] == prov["nep_sum_mirror"]
prov.to_csv(f"{OUT}/provenance_diff.csv", index=False)

# ------------------------------------------------ pipeline per era
def screen_panel(d):
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
    return d.merge(kept, on=["line", "group_id"])

def covariates(d):
    prem = (d[d["dev_lag"] == 1]
            .groupby(["line", "group_id", "acc_yr"])["net_ep"].first().reset_index())
    ind = prem.groupby(["line", "acc_yr"])["net_ep"].sum().rename("ind_nep").reset_index()
    ind["dlog"] = ind.groupby("line")["ind_nep"].transform(lambda s: np.log(s).diff())
    ind["z_raw"] = ind.groupby("line")["dlog"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=1))
    ind["z_cyc"] = np.nan
    for line_, g in ind.groupby("line"):
        sg = g.dropna(subset=["dlog"])
        X = sm.add_constant(sg["acc_yr"].astype(float))
        r = sm.OLS(sg["dlog"].to_numpy(), X).fit().resid
        ind.loc[sg.index, "z_cyc"] = (r - r.mean()) / r.std(ddof=1)
    prem = prem.merge(ind[["line", "acc_yr", "dlog", "z_raw", "z_cyc"]],
                      on=["line", "acc_yr"], how="left")
    prem["g_co"] = prem.groupby(["line", "group_id"])["net_ep"].transform(
        lambda s: np.log(s).diff())
    prem["g_rel_raw"] = prem["g_co"] - prem["dlog"]
    lo, hi = prem["g_rel_raw"].quantile([0.01, 0.99])
    prem["g_rel"] = prem.groupby("line")["g_rel_raw"].transform(
        lambda s: (s.clip(lo, hi) - s.clip(lo, hi).mean()) / s.clip(lo, hi).std(ddof=1))
    return prem, ind

def factors(d, prem, value_col):
    p = d.pivot_table(index=["line", "group_id", "acc_yr"], columns="dev_lag",
                      values=value_col)
    rows = []
    for (line, gid, ay), r in p.iterrows():
        for j in range(1, 10):
            c0, c1 = r.get(j, np.nan), r.get(j + 1, np.nan)
            if np.isfinite(c0) and np.isfinite(c1) and c0 > 0 and c1 > 0:
                rows.append(dict(line=line, group_id=gid, acc_yr=ay, lag=j,
                                 cal_yr=ay + j, f=c1 / c0, w=c0))
    fac = pd.DataFrame(rows)
    fac["lnf"] = np.log(fac["f"])
    return fac.merge(prem[["line", "group_id", "acc_yr", "z_raw", "z_cyc",
                           "g_rel"]], on=["line", "group_id", "acc_yr"],
                     how="left"), p

def run_line_specs(est, era_label, basis):
    rows = []
    for line in LINES:
        s = est[(est["line"] == line)].dropna(subset=["z_cyc", "g_rel"])
        if len(s) < 100:
            continue
        for name, xcols in [("raw", ["z_raw"]), ("cyc+rel", ["z_cyc", "g_rel"])]:
            Xd = pd.get_dummies(s["lag"], prefix="lag").astype(float)
            X = pd.concat([Xd, s[xcols]], axis=1)
            m = sm.WLS(s["lnf"], X, weights=s["w"]).fit(
                cov_type="cluster", cov_kwds={"groups": s["acc_yr"]})
            for c in xcols:
                rows.append(dict(era=era_label, basis=basis, line=line,
                                 spec=name, var=c, beta=m.params[c],
                                 se=m.bse[c], p=m.pvalues[c], n=int(m.nobs)))
    s = est[est["line"].isin(MAIN)].dropna(subset=["z_cyc", "g_rel"])
    Xd = pd.get_dummies(s["lag"], prefix="lag").astype(float)
    Xl = pd.get_dummies(s["line"], prefix="ln", drop_first=True).astype(float)
    X = pd.concat([Xd, Xl, s[["z_cyc", "g_rel"]]], axis=1)
    m = sm.WLS(s["lnf"], X, weights=s["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": s["line"] + s["acc_yr"].astype(str)})
    for c in ["z_cyc", "g_rel"]:
        rows.append(dict(era=era_label, basis=basis, line="POOLED_MAIN",
                         spec="cyc+rel", var=c, beta=m.params[c], se=m.bse[c],
                         p=m.pvalues[c], n=int(m.nobs)))
    return rows

results, oos_rows, booked_rows, lagprof_rows = [], [], [], []
era_store = {}
for d, era, val in [(dA, "A", 1997), (dB, "B", 2007)]:
    dk = screen_panel(d)
    prem, ind = covariates(dk)
    for value_col, basis in [("cum_paid", "paid"), ("case_inc", "incurred")]:
        fac, psq = factors(dk, prem, value_col)
        est = fac[(fac["cal_yr"] <= val) & (fac["lag"] <= MAXLAG)]
        results += run_line_specs(est, era, basis)
        if basis == "paid":
            era_store[era] = dict(dk=dk, prem=prem, ind=ind, fac=fac, psq=psq)
        # per-lag profile, pooled main lines
        s = est[est["line"].isin(MAIN)].dropna(subset=["z_cyc", "g_rel"])
        for j in range(1, MAXLAG + 1):
            sj = s[s["lag"] == j]
            if len(sj) < 100:
                continue
            Xl = pd.get_dummies(sj["line"], prefix="ln", drop_first=True).astype(float)
            X = sm.add_constant(pd.concat([Xl, sj[["z_cyc", "g_rel"]]], axis=1))
            m = sm.WLS(sj["lnf"], X, weights=sj["w"]).fit(
                cov_type="cluster",
                cov_kwds={"groups": sj["line"] + sj["acc_yr"].astype(str)})
            lagprof_rows.append(dict(era=era, basis=basis, lag=j,
                                     beta=m.params["z_cyc"], se=m.bse["z_cyc"],
                                     p=m.pvalues["z_cyc"], n=int(m.nobs)))
        # OOS one-year-ahead (last four diagonals of the era)
        for line in LINES:
            sub = fac[(fac["line"] == line) & (fac["lag"] <= MAXLAG)].dropna(
                subset=["z_cyc", "g_rel"])
            for T in range(val - 3, val + 1):
                tr, te = sub[sub["cal_yr"] <= T - 1], sub[sub["cal_yr"] == T]
                te = te[te["lag"].isin(tr["lag"].unique())]
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
                wte = te.loc[ok, "w"]
                oos_rows.append(dict(
                    era=era, basis=basis, line=line, T=T, n=int(ok.sum()),
                    mse_cl=np.average((te.loc[ok, "lnf"] - pred_cl[ok]) ** 2, weights=wte),
                    mse_cy=np.average((te.loc[ok, "lnf"] - pred_cy[ok]) ** 2, weights=wte)))
    # booked-ultimate errors (booked = Part 2 incurred at era valuation)
    bk = dk[dk["dev_yr"] == val][["line", "group_id", "acc_yr", "incurred"]]
    act = dk[dk["dev_lag"] == 10][["line", "group_id", "acc_yr", "cum_paid"]]\
        .rename(columns={"cum_paid": "actual_ult"})
    mot = bk.merge(act, on=["line", "group_id", "acc_yr"])
    mot = mot.groupby(["line", "acc_yr"])[["incurred", "actual_ult"]].sum().reset_index()
    mot["err"] = mot["incurred"] / mot["actual_ult"] - 1
    mot = mot.merge(ind[["line", "acc_yr", "z_raw", "z_cyc"]], on=["line", "acc_yr"])
    mot["era"] = era
    booked_rows.append(mot)

res = pd.DataFrame(results)
res.to_csv(f"{OUT}/beta_estimates_twoera.csv", index=False)
oos = pd.DataFrame(oos_rows)
oos.to_csv(f"{OUT}/oos_twoera.csv", index=False)
lagprof = pd.DataFrame(lagprof_rows)
lagprof.to_csv(f"{OUT}/lagprofile_twoera.csv", index=False)
booked = pd.concat(booked_rows, ignore_index=True)
booked.to_csv(f"{OUT}/booked_twoera.csv", index=False)

# booked-error regressions per era (main lines, line FE, cluster by AY)
for era in ["A", "B"]:
    mm = booked[(booked["era"] == era) & booked["line"].isin(MAIN)].dropna(subset=["z_cyc"])
    Xl = pd.get_dummies(mm["line"], prefix="ln").astype(float)
    X = pd.concat([Xl, mm[["z_cyc"]]], axis=1)
    m = sm.OLS(mm["err"], X).fit(cov_type="cluster", cov_kwds={"groups": mm["acc_yr"]})
    print(f"[era {era}] booked-ult error on z_cyc: beta={m.params['z_cyc']:.4f} "
          f"se={m.bse['z_cyc']:.4f} p={m.pvalues['z_cyc']:.4f} n={int(m.nobs)}")

# pooled two-era regression (paid): z standardized within line-era already
facA = era_store["A"]["fac"].assign(era="A", val=1997)
facB = era_store["B"]["fac"].assign(era="B", val=2007)
pool = pd.concat([facA[facA["cal_yr"] <= 1997], facB[facB["cal_yr"] <= 2007]])
pool = pool[(pool["lag"] <= MAXLAG) & pool["line"].isin(MAIN)].dropna(
    subset=["z_cyc", "g_rel"])
Xd = pd.get_dummies(pool["lag"], prefix="lag").astype(float)
Xle = pd.get_dummies(pool["line"] + "_" + pool["era"], prefix="le",
                     drop_first=True).astype(float)
X = pd.concat([Xd, Xle, pool[["z_cyc", "g_rel"]]], axis=1)
m = sm.WLS(pool["lnf"], X, weights=pool["w"]).fit(
    cov_type="cluster", cov_kwds={"groups": pool["line"] + pool["acc_yr"].astype(str)})
print(f"\n[POOLED 1988-2007, paid, main lines] z_cyc: {m.params['z_cyc']:.4f} "
      f"({m.bse['z_cyc']:.4f}) p={m.pvalues['z_cyc']:.4f}; "
      f"g_rel: {m.params['g_rel']:.4f} ({m.bse['g_rel']:.4f}) "
      f"p={m.pvalues['g_rel']:.4f}; n={int(m.nobs)}")
with open(f"{OUT}/pooled_twoera.txt", "w") as fh:
    fh.write(f"z_cyc,{m.params['z_cyc']},{m.bse['z_cyc']},{m.pvalues['z_cyc']}\n")
    fh.write(f"g_rel,{m.params['g_rel']},{m.bse['g_rel']},{m.pvalues['g_rel']}\n")
    fh.write(f"n,{int(m.nobs)}\n")

print("\n=== provenance (era A vs mirror) ===")
print(prov[["line", "n_records_casact", "n_records_mirror", "paid_match",
            "caseinc_match", "nep_match"]].to_string(index=False))
print("\n=== era-B covariate (z_cyc) by line/AY ===")
indB = era_store["B"]["ind"]
print(indB.pivot_table(index="acc_yr", columns="line", values="z_cyc").round(2))
print("\n=== main specs, era B ===")
rb = res[(res["era"] == "B") & res["spec"].isin(["raw", "cyc+rel"])]
print(rb.round(4).to_string(index=False))
print("\n=== OOS summary era B ===")
ob = oos[oos["era"] == "B"]
osum = ob.groupby(["basis", "line"]).apply(
    lambda g: pd.Series(dict(mse_cl=np.average(g["mse_cl"], weights=g["n"]),
                             mse_cy=np.average(g["mse_cy"], weights=g["n"]))),
    include_groups=False)
osum["gain_pct"] = 100 * (1 - osum["mse_cy"] / osum["mse_cl"])
print(osum.round(5).to_string())
print("\nkept companies era B:",
      era_store["B"]["dk"].groupby("line")["group_id"].nunique().to_dict())
