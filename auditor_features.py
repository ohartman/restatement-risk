#!/usr/bin/env python3
"""The auditor's own record: PCAOB inspections and Form AP, per 10-K.

Who is looking, and how good are they at it. The PCAOB inspects every registered
audit firm -- annually for the large ones, every three years for the rest --
and publishes, per report, how many of the audits it reviewed had a Part I.A
deficiency (an audit that failed to obtain sufficient evidence for its opinion).
Form AP names the firm and the engagement partner on every issuer audit since
2017. Both are free downloads (data/raw/pcaob/).

Per filing, using only reports published before the filing date:
  pcaob_rate            the auditor's latest Part I.A deficiency rate (0-1)
  pcaob_rate_prev       the report before that (trend = rate - prev)
  pcaob_rate_trend
  pcaob_audits_reviewed audits reviewed in that inspection
  pcaob_clients         the firm's issuer audit clients, per the report
  pcaob_annual          1 if annually inspected (the large firms), 0 if triennial
  pcaob_years_since     years since the latest report
  pcaob_never           1 if no report had been published before the filing
And from Form AP (fiscal periods from 2017):
  ap_auditor_changed    firm ID differs from the issuer's prior fiscal year
  ap_partner_changed    engagement partner ID differs from the prior year
  ap_participants       other firms that took part in the audit (count)
  ap_participant_pct    the largest other-firm share of audit hours
  ap_report_lag         audit report date minus fiscal period end, days
  ap_firm_issuers       issuer audits the firm signed in the prior calendar year (size)

Auditor identity comes from Form AP where it exists (2017+). The information --
which firm signed, when, who led -- is printed in the 10-K's own audit report,
so it is knowable on filing day; Form AP is the structured copy.

  python auditor_features.py   -> data/out/features_auditor.npz
"""

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from relabel import submissions

PCAOB = Path("data/raw/pcaob")
NAMES = ["pcaob_rate", "pcaob_rate_prev", "pcaob_rate_trend", "pcaob_audits_reviewed", "pcaob_clients", "pcaob_annual",
         "pcaob_years_since", "pcaob_never", "ap_auditor_changed", "ap_partner_changed", "ap_participants",
         "ap_participant_pct", "ap_report_lag", "ap_firm_issuers"]


def main():
    ir = pd.read_csv(PCAOB / "inspection_reports.csv", encoding="utf-16-le", dtype=str)
    ir.columns = [c.strip().lstrip("﻿") for c in ir.columns]
    ir["date"] = pd.to_datetime(ir["Inspection Report Date"], format="%d-%b-%Y", errors="coerce")
    ir["rate"] = pd.to_numeric(ir["Part I.A Deficiency Rate"].str.rstrip("%"), errors="coerce") / 100.0
    ir["reviewed"] = pd.to_numeric(ir["Total Audits Reviewed"], errors="coerce")
    ir["clients"] = pd.to_numeric(ir["Total Issuer Audit Clients"], errors="coerce")
    ir["annual"] = (ir["Inspection Type"].str.strip() == "Annually Inspected").astype(float)
    ir = ir.dropna(subset=["date", "rate"]).sort_values("date")
    reports = {fid: g[["date", "rate", "reviewed", "clients", "annual"]].values.tolist()
               for fid, g in ir.groupby(ir["Registration ID"].str.strip())}
    print(f"{len(ir):,} inspection reports with a deficiency rate, {len(reports):,} firms")

    ap = pd.read_csv(PCAOB / "FirmFilings.csv", dtype=str, low_memory=False)
    ap = ap[(ap["Audit Report Type"].str.startswith("Issuer", na=False)) & (ap["Latest Form AP Filing"] == "1")].copy()
    ap["cik"] = ap["Issuer CIK"].str.lstrip("0")
    ap["fpe"] = pd.to_datetime(ap["Fiscal Period End Date"], errors="coerce")
    ap["ard"] = pd.to_datetime(ap["Audit Report Date"], errors="coerce")
    ap["pct"] = pd.to_numeric(ap["Participant Percentage"], errors="coerce")
    ap["nparts"] = pd.to_numeric(ap["Number of Participants"], errors="coerce")
    ap = ap.dropna(subset=["cik", "fpe"])
    by_issuer = {c: g.sort_values("fpe") for c, g in ap.groupby("cik")}
    firm_year_counts = ap.groupby([ap["Firm ID"], ap["fpe"].dt.year]).size().to_dict()
    print(f"{len(ap):,} issuer audits on Form AP, {len(by_issuer):,} issuers")

    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y = z["adsh"], z["y"]
    subs = submissions()
    X = np.full((len(adsh), len(NAMES)), np.nan)
    matched = 0
    for i, a in enumerate(adsh.tolist()):
        cik, filed, pyear = subs.get(a, (None, None, None))
        if cik is None:
            continue
        g = by_issuer.get(cik)
        row = [np.nan] * len(NAMES)
        firm = None
        if g is not None:
            # the Form AP record for this 10-K's fiscal period end
            try:
                pend = datetime.strptime(str(int(pyear)), "%Y")  # placeholder, replaced below
            except Exception:
                pend = None
            per = None
            r = None
            # match on the period end date from sub.txt via the features cache's filed date window
            cand = g[(g["fpe"] <= pd.Timestamp(filed)) & (g["fpe"] >= pd.Timestamp(filed) - pd.Timedelta(days=400))]
            if len(cand):
                r = cand.iloc[-1]
                firm = str(r["Firm ID"]).strip()
                prev = g[g["fpe"] < r["fpe"] - pd.Timedelta(days=300)]
                prev = prev.iloc[-1] if len(prev) else None
                row[8] = float(prev is not None and str(prev["Firm ID"]).strip() != firm) if prev is not None else np.nan
                row[9] = float(prev is not None and str(prev["Engagement Partner ID"]) != str(r["Engagement Partner ID"])) if prev is not None else np.nan
                row[10] = float(r["nparts"]) if pd.notna(r["nparts"]) else 0.0
                row[11] = float(r["pct"]) if pd.notna(r["pct"]) else 0.0
                row[12] = float((r["ard"] - r["fpe"]).days) if pd.notna(r["ard"]) else np.nan
                row[13] = float(firm_year_counts.get((firm, r["fpe"].year - 1), 0))
                matched += 1
        if firm is not None:
            reps = [x for x in reports.get(firm, []) if x[0] < pd.Timestamp(filed)]
            if reps:
                d, rate, reviewed, clients, annual = reps[-1]
                row[0] = rate; row[3] = reviewed; row[4] = clients; row[5] = annual
                row[6] = (pd.Timestamp(filed) - d).days / 365.25; row[7] = 0.0
                if len(reps) > 1:
                    row[1] = reps[-2][1]; row[2] = rate - reps[-2][1]
            else:
                row[7] = 1.0
        X[i] = row
    Path("data/out").mkdir(exist_ok=True)
    np.savez("data/out/features_auditor.npz", X=X, adsh=adsh, names=np.array(NAMES))
    from sklearn.metrics import roc_auc_score
    print(f"\nForm AP matched for {matched:,} of {len(adsh):,} filings")
    for k, n in enumerate(NAMES):
        v = X[:, k]; ok = np.isfinite(v)
        if ok.sum() and 0 < y[ok].sum() < ok.sum() and len(np.unique(v[ok])) > 1:
            line = f"  {n:<22} coverage {100*ok.mean():3.0f}%  AUC alone {roc_auc_score(y[ok], v[ok]):.3f}"
            if set(np.unique(v[ok])) <= {0.0, 1.0}:
                line += f"   rate when 1: {y[ok][v[ok]==1].mean():6.2%} (n={int((v[ok]==1).sum()):>6,})   when 0: {y[ok][v[ok]==0].mean():.2%}"
            print(line)
    ok = np.isfinite(X[:, 0])
    if ok.sum():
        q = np.nanquantile(X[ok, 0], [0.25, 0.5, 0.75])
        for lo, hi, lab in ((0, q[0], "lowest quartile"), (q[2], 1.01, "highest quartile")):
            m = ok & (X[:, 0] >= lo) & (X[:, 0] < hi)
            print(f"  auditor deficiency rate {lab} (<{hi:.0%}): restatement rate {y[m].mean():.2%} (n={int(m.sum()):,})")


if __name__ == "__main__":
    main()
