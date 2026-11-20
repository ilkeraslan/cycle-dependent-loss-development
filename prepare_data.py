"""Build data/cas_full_square.csv, the flat era-A panel read by scripts 01 to 04,
from the CAS files in data/cas_earlier/ (accident years 1988-1997).

Columns: acc_yr, dev_yr, dev_lag; cumulative and incremental case-incurred losses
(incurred less bulk reserves) and paid losses; booked ultimate (incurred), bulk IBNR;
direct, ceded and net earned premium; single-entity flag; posted reserve 1997;
line, group code and group name. Rows keep the order of the CAS files, lines in
alphabetical order. Amounts in USD thousands.
Run: python prepare_data.py
"""
import csv
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LINES = ["comauto", "medmal", "othliab", "ppauto", "prodliab", "wkcomp"]

frames = []
for line in LINES:
    e = pd.read_csv(os.path.join(HERE, "data", "cas_earlier", f"{line}.csv"))
    e.columns = [c.strip() for c in e.columns]
    # CAS column names carry a line-specific suffix (IncurLoss_C, IncurLoss_D, ...)
    pick = lambda prefix: e[next(c for c in e.columns if c.lower().startswith(prefix.lower()))]
    f = pd.DataFrame({
        "acc_yr": e["AccidentYear"], "dev_yr": e["DevelopmentYear"], "dev_lag": e["DevelopmentLag"],
        "cum_caseinc_loss": pick("IncurLoss") - pick("BulkLoss"),
        "cum_paid_loss": pick("CumPaidLoss"),
        "booked_ult_loss": pick("IncurLoss"), "bulk_ibnr": pick("BulkLoss"),
        "direct_ep": pick("EarnedPremDIR"), "ceded_ep": pick("EarnedPremCeded"),
        "net_ep": pick("EarnedPremNet"), "single_entity": e["Single"],
        "posted_reserve_1997": pick("PostedReserve97"),
        "line": line, "group_id": e["GRCODE"], "company": e["GRNAME"]})
    g = f.groupby(["group_id", "acc_yr"])
    f["increm_caseinc_loss"] = g["cum_caseinc_loss"].diff().fillna(f["cum_caseinc_loss"]).astype("int64")
    f["increm_paid_loss"] = g["cum_paid_loss"].diff().fillna(f["cum_paid_loss"]).astype("int64")
    frames.append(f)
cols = ["acc_yr", "dev_yr", "dev_lag", "cum_caseinc_loss", "increm_caseinc_loss", "cum_paid_loss",
        "increm_paid_loss", "booked_ult_loss", "bulk_ibnr", "direct_ep", "ceded_ep", "net_ep",
        "single_entity", "posted_reserve_1997", "line", "group_id", "company"]
out = pd.concat(frames, ignore_index=True)[cols]
out.to_csv(os.path.join(HERE, "data", "cas_full_square.csv"), index=False, quoting=csv.QUOTE_NONNUMERIC)
for sub in ("tables", "figures"):
    os.makedirs(os.path.join(HERE, "results", sub), exist_ok=True)
print(f"wrote data/cas_full_square.csv: {len(out):,} rows, {out['line'].nunique()} lines")
