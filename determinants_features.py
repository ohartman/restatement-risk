#!/usr/bin/env python3
"""Determinants of weak internal control (Doyle, Ge & McVay 2007; Ashbaugh-Skaife, Collins &
Kinney 2007), each one lookup in data already on disk:

  firm_age_years        years since the company's first EDGAR filing of any kind (form indices)
  n_segments            distinct business-segment members tagged for the current period
  n_geographies         distinct geographic members tagged for the current period
  restructuring_to_assets   RestructuringCharges / Assets (0 when not tagged but assets are)
  dte_to_assets         DeferredIncomeTaxExpenseBenefit / Assets (Ettredge et al. 2008)
  foreign_pretax_share  foreign pre-tax income / total pre-tax income (|.| capped at 2)
  btd_to_assets         book-tax difference: (pretax - current tax / 0.21) / Assets
  va_change_to_assets   change in the deferred-tax valuation allowance / Assets
  has_foreign, has_restructuring, has_va

  python determinants_features.py   -> data/out/features_determinants.npz
"""
import csv, glob, io, math, zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from relabel import submissions

RAW = Path("data/raw")
NAMES = ["firm_age_years", "n_segments", "n_geographies", "restructuring_to_assets", "dte_to_assets", "foreign_pretax_share",
         "btd_to_assets", "va_change_to_assets", "has_foreign", "has_restructuring", "has_va"]
TAGS = {
    "assets": ["Assets"],
    "restructuring": ["RestructuringCharges", "RestructuringCosts", "RestructuringAndRelatedCostIncurredCost"],
    "dte": ["DeferredIncomeTaxExpenseBenefit", "DeferredIncomeTaxesAndTaxCredits"],
    "pretax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "foreign": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign"],
    "curtax": ["CurrentIncomeTaxExpenseBenefit"],
    "va": ["DeferredTaxAssetsValuationAllowance"],
}
TAG2 = {t: c for c, ts in TAGS.items() for t in ts}
SEG_AXES = ("BusinessSegments=", "Segments=")            # the data sets write axes as "BusinessSegments=Member;"
GEO_AXES = ("Geographical=",)


def year_before(ddate):
    return f"{int(ddate[:4]) - 1}{ddate[4:]}"


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh = z["adsh"].tolist(); filed = z["filed"].astype("datetime64[D]")
    pos = {a: i for i, a in enumerate(adsh)}
    subs = submissions()
    # firm age: first EDGAR filing of any form type, from the indices (1993 on)
    first = {}
    for f in sorted(glob.glob("data/raw/edgar_index/form_*.idx")):
        for line in open(f, encoding="latin-1"):
            parts = line.rstrip("\n").split()
            if len(parts) < 4 or not parts[-2][:1].isdigit():
                continue
            cik, date = parts[-3], parts[-2]
            if cik.isdigit() and (cik not in first or date < first[cik]):
                first[cik] = date
    print(f"first-filing dates for {len(first):,} CIKs", flush=True)
    vals = defaultdict(dict)          # adsh -> {(concept, which): value}
    segs = defaultdict(set); geos = defaultdict(set)
    for zp in sorted(RAW.glob("*q?.zip")):
        zf = zipfile.ZipFile(zp)
        with zf.open("sub.txt") as fh:
            sub = {r["adsh"]: r["period"] for r in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t")
                   if r["adsh"] in pos and r["period"]}
        with zf.open("num.txt") as fh:
            for r in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t"):
                a = r["adsh"]; period = sub.get(a)
                if period is None:
                    continue
                seg = r.get("segments") or ""
                if seg and r["ddate"] == period:
                    for part in seg.split(";"):
                        if part.startswith(SEG_AXES): segs[a].add(part)
                        elif part.startswith(GEO_AXES): geos[a].add(part)
                    continue
                if seg or r.get("coreg"):
                    continue
                c = TAG2.get(r["tag"])
                if c is None:
                    continue
                try:
                    v = float(r["value"])
                except (TypeError, ValueError):
                    continue
                which = "cur" if r["ddate"] == period else ("pri" if r["ddate"] == year_before(period) else None)
                if which is None:
                    continue
                if c in ("assets", "va") and r["qtrs"] != "0":
                    continue
                if c not in ("assets", "va") and r["qtrs"] != "4":
                    continue
                vals[a].setdefault((c, which), v)
        print(f"  {zp.stem}", flush=True)
    X = np.full((len(adsh), len(NAMES)), np.nan)
    for a, i in pos.items():
        cik, fd, _ = subs.get(a, (None, None, None))
        if cik and cik in first:
            X[i, 0] = (fd - datetime.strptime(first[cik], "%Y-%m-%d").date()).days / 365.25
        v = vals.get(a, {})
        assets = v.get(("assets", "cur"))
        X[i, 1] = len(segs.get(a, ())); X[i, 2] = len(geos.get(a, ()))
        if assets and assets > 0:
            rs = v.get(("restructuring", "cur")); X[i, 3] = (rs or 0.0) / assets; X[i, 9] = float(rs is not None and rs != 0)
            dte = v.get(("dte", "cur")); X[i, 4] = dte / assets if dte is not None else np.nan
            pre = v.get(("pretax", "cur")); fo = v.get(("foreign", "cur"))
            if pre is not None and fo is not None and pre != 0:
                X[i, 5] = max(-2.0, min(2.0, fo / abs(pre)))
            X[i, 8] = float(fo is not None and fo != 0)
            ct = v.get(("curtax", "cur"))
            if pre is not None and ct is not None:
                X[i, 6] = (pre - ct / 0.21) / assets
            va, vap = v.get(("va", "cur")), v.get(("va", "pri"))
            if va is not None and vap is not None:
                X[i, 7] = (va - vap) / assets
            X[i, 10] = float(va is not None and va != 0)
    np.savez("data/out/features_determinants.npz", X=X, adsh=np.array(adsh), names=np.array(NAMES))
    y = z["y"]
    from sklearn.metrics import roc_auc_score
    print(f"\nsaved features_determinants.npz for {len(adsh):,} filings")
    for k, nm in enumerate(NAMES):
        col = X[:, k]; m = np.isfinite(col)
        if m.sum() > 100 and 0 < y[m].sum() < m.sum():
            line = f"  {nm:<26} coverage {m.mean():>4.0%}  AUC alone {roc_auc_score(y[m], col[m]):.3f}"
            if set(np.unique(col[m])) <= {0.0, 1.0}:
                b = col[m] > 0; line += f"   rate when 1: {y[m][b].mean():.1%} (n={int(b.sum()):,})  when 0: {y[m][~b].mean():.1%}"
            print(line)


if __name__ == "__main__":
    main()
