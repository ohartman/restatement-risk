#!/usr/bin/env python3
"""The year's own arithmetic: what the three 10-Qs say about the 10-K.

Every 10-K sits on top of three quarterly reports the company already filed.
Together they let the annual figures be checked against their own parts, on the
day the 10-K lands:

  q4_share_ni, q4_share_rev   the implied fourth quarter (annual minus Q1-Q3) as a share of the year --
                              a fourth quarter carrying most of the profit is the classic plug
  q4_ni_vs_q123_mean          implied Q4 net income relative to the mean of the first three quarters
  q_rev_cv, q_ni_cv           coefficient of variation of quarterly revenue and net income
  q_accrual_swing             range of quarterly (net income - operating cash flow)/assets across the year
  ytd_mismatch_rev            |Q3 year-to-date revenue - (Q1+Q2+Q3)| / annual revenue: the quarterly filings
                              disagreeing with themselves
  n_10q                       10-Qs found for the fiscal year (0-3); missing quarters are a signal in themselves
  q_assets_swing              max quarter-end assets / min quarter-end assets within the year
  q_ni_sign_changes           how often quarterly net income changed sign

Consolidated figures only; quarterly flows are qtrs=1 (a single quarter), year-to-date qtrs=2/3.
Written as NaN where a firm's 10-Qs are missing.

  python quarterly_features.py   -> data/out/features_quarterly.npz (X, adsh, names)
"""

import csv
import io
import zipfile
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

from fscore import TAGS, is_financial

RAW = Path("data/raw")
NAMES = ["q4_share_ni", "q4_share_rev", "q4_ni_vs_q123_mean", "q_rev_cv", "q_ni_cv", "q_accrual_swing",
         "ytd_mismatch_rev", "n_10q", "q_assets_swing", "q_ni_sign_changes"]
CFO_TAGS = ["NetCashProvidedByUsedInOperatingActivities"]
WANT = {t: c for c, ts in TAGS.items() for t in ts if c in ("revenue", "net_income", "assets")}
WANT.update({t: "cfo" for t in CFO_TAGS})
RANK = {t: i for c, ts in TAGS.items() for i, t in enumerate(ts)}
RANK.update({t: i for i, t in enumerate(CFO_TAGS)})


def read_quarterlies():
    """(cik, ddate) -> {concept: value} for single-quarter flows / quarter-end stocks, from 10-Q filings.
    Also (cik, ddate) -> year-to-date revenue for the Q3 10-Q (qtrs=3)."""
    q, ytd = defaultdict(dict), {}
    best = defaultdict(dict)
    for zp in sorted(RAW.glob("*q?.zip")):
        z = zipfile.ZipFile(zp)
        with z.open("sub.txt") as f:
            subs = {r["adsh"]: (str(int(r["cik"])), r["form"]) for r in
                    csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"), delimiter="\t")
                    if r["form"] in ("10-Q", "10-Q/A") and not is_financial(r["sic"])}
        with z.open("num.txt") as f:
            for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"), delimiter="\t"):
                s = subs.get(r["adsh"])
                if s is None or r["tag"] not in WANT or r["segments"] or r["coreg"] or r["uom"] != "USD":
                    continue
                try:
                    v = float(r["value"])
                except (TypeError, ValueError):
                    continue
                cik, concept = s[0], WANT[r["tag"]]
                key = (cik, r["ddate"])
                if concept == "assets" and r["qtrs"] == "0" or concept != "assets" and r["qtrs"] == "1":
                    prev = best[key].get(concept)
                    if prev is None or RANK[r["tag"]] < prev:
                        best[key][concept] = RANK[r["tag"]]; q[key][concept] = v
                elif concept == "revenue" and r["qtrs"] == "3":
                    ytd[key] = v
        print(f"  {zp.stem}: {len(q):,} firm-quarters so far", flush=True)
    return q, ytd


def main():
    from fscore import read_quarter
    q, ytd = read_quarterlies()
    by_cik = defaultdict(list)
    for (cik, dd), vals in q.items():
        by_cik[cik].append((dd, vals))
    for v in by_cik.values():
        v.sort()

    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y = z["adsh"], z["y"]
    # each 10-K's own annual figures and period, from the feature build's source
    annual = {}
    for zp in sorted(RAW.glob("*q?.zip")):
        subs, facts = read_quarter(zp)
        for a in adsh.tolist():
            s = subs.get(a)
            if s and a in facts and s["period"] in facts[a]:
                annual[a] = (str(int(s["cik"])), s["period"], facts[a][s["period"]])
    print(f"annual figures for {len(annual):,} of {len(adsh):,} filings")

    X = np.full((len(adsh), len(NAMES)), np.nan)
    for i, a in enumerate(adsh.tolist()):
        m = annual.get(a)
        if m is None:
            continue
        cik, P, ann = m
        pend = datetime.strptime(P, "%Y%m%d").date()
        start = date(pend.year - 1, pend.month, min(pend.day, 28))
        # the three quarter-ends inside the fiscal year, strictly before the year-end
        qs = [(dd, v) for dd, v in by_cik.get(cik, []) if start < datetime.strptime(dd, "%Y%m%d").date() < pend]
        qs = qs[-3:]
        n = len(qs)
        rev_q = [v.get("revenue") for _, v in qs]; ni_q = [v.get("net_income") for _, v in qs]
        as_q = [v.get("assets") for _, v in qs]
        rev_a, ni_a, as_a = ann.get("revenue"), ann.get("net_income"), ann.get("assets")
        row = [np.nan] * len(NAMES)
        row[7] = float(n)
        if n == 3 and None not in rev_q and rev_a not in (None, 0):
            q4_rev = rev_a - sum(rev_q)
            row[1] = q4_rev / abs(rev_a)
            row[3] = float(np.std(rev_q + [q4_rev]) / max(1e-9, abs(np.mean(rev_q + [q4_rev]))))
            key3 = (cik, qs[-1][0])
            if key3 in ytd:
                row[6] = abs(ytd[key3] - sum(rev_q)) / abs(rev_a)
        if n == 3 and None not in ni_q and ni_a is not None:
            q4_ni = ni_a - sum(ni_q)
            denom = sum(abs(x) for x in ni_q) + abs(q4_ni)
            row[0] = q4_ni / denom if denom else np.nan
            mean123 = np.mean(ni_q)
            row[2] = q4_ni / abs(mean123) if abs(mean123) > 1e3 else np.nan
            row[4] = float(np.std(ni_q + [q4_ni]) / max(1e-9, abs(np.mean(ni_q + [q4_ni]))))
            signs = np.sign(ni_q + [q4_ni])
            row[9] = float(np.sum(signs[1:] != signs[:-1]))
        if n >= 2 and all(x is not None for x in as_q) and as_a:
            allv = [x for x in as_q if x and x > 0] + [as_a]
            row[8] = max(allv) / min(allv) if min(allv) > 0 else np.nan
        cfo_q = [v.get("cfo") for _, v in qs]
        if n == 3 and None not in cfo_q and None not in ni_q and as_a:
            acc = [(ni - cf) / as_a for ni, cf in zip(ni_q, cfo_q)]
            row[5] = max(acc) - min(acc)
        X[i] = row
    Path("data/out").mkdir(exist_ok=True)
    np.savez("data/out/features_quarterly.npz", X=X, adsh=adsh, names=np.array(NAMES))
    from sklearn.metrics import roc_auc_score
    print(f"\nfilings with all three 10-Qs: {int((X[:, 7] == 3).sum()):,} of {len(adsh):,}; "
          f"restatement rate with 3: {y[X[:, 7] == 3].mean():.2%}   with fewer: {y[(X[:, 7] < 3)].mean():.2%}")
    for k, nm in enumerate(NAMES):
        v = X[:, k]; ok = np.isfinite(v)
        if ok.sum() and 0 < y[ok].sum() < ok.sum() and len(np.unique(v[ok])) > 1:
            print(f"  {nm:<20} coverage {100*ok.mean():3.0f}%  AUC alone {roc_auc_score(y[ok], v[ok]):.3f}")


if __name__ == "__main__":
    main()
