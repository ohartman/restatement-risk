#!/usr/bin/env python3
"""Forecasting, not detection: will the company's NEXT 10-K (or the one after) be restated?

Parker, Jiang, Cho & Vasarhelyi (2025, The Accounting Review) forecast material
misstatements one and two years ahead from the current year's data. Our headline
label is detection: whether THIS 10-K is later restated. Here each filing is
relabelled with the loose label of the same company's next annual filing (9-15
months later) and of the one after that (21-27 months later), and the clean
full stack is trained and tested on those labels with the usual split.

  BRF_STRATEGY=0.1 SCRUTINY_NPZ=... python forecast_test.py
"""
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from edge import impute
from events_test import brf, joined
from final import TEST_START, ci
from relabel import submissions


def fit_score(X, y, trn, tst, seeds=5):
    Xtr, Xte = impute(X[trn], X[tst])
    return np.mean([brf(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    R, _ = joined(adsh, "data/out/features_raw.npz"); R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    E, _ = joined(adsh, "data/out/features_events.npz"); I, _ = joined(adsh, "data/out/features_insider.npz")
    M, mn = joined(adsh, "data/out/features_market.npz"); M = M[:, [i for i, n in enumerate(mn) if n != "price_cov"]]
    L, _ = joined(adsh, "data/out/features_letters.npz"); S, _ = joined(adsh, os.environ.get("SCRUTINY_NPZ", "data/out/features_scrutiny.npz"))
    full = np.hstack([fin, E, I, M, L, S])
    subs = submissions()
    by_cik = defaultdict(list)
    for i, a in enumerate(adsh.tolist()):
        cik = subs.get(a, (None,))[0]
        if cik:
            by_cik[cik].append((filed[i], i))
    nxt = {}   # i -> (index of next filing, index of the one after)
    for cik, lst in by_cik.items():
        lst.sort()
        for k, (d, i) in enumerate(lst):
            n1 = n2_ = None
            for d2, j in lst[k + 1:]:
                gap = (d2 - d).astype(int)
                if n1 is None and 270 <= gap <= 460: n1 = j
                elif n1 is not None and 630 <= gap <= 820: n2_ = j; break
            nxt[i] = (n1, n2_)
    y1 = np.full(len(adsh), np.nan); y2 = np.full(len(adsh), np.nan)
    for i, (a, b) in nxt.items():
        if a is not None: y1[i] = y[a]
        if b is not None: y2[i] = y[b]
    print(f"filings with a next 10-K in the panel: {int(np.isfinite(y1).sum()):,}; with the one after: {int(np.isfinite(y2).sum()):,}")
    print(f"{'target':<52}{'train (pos)':>14}{'test (pos)':>13}{'test AUC':>10}{'95% CI':>17}   per year")
    for name, yy, X in (("detection: this 10-K restated (the headline label)", y.astype(float), full),
                        ("forecast: the next 10-K restated (one year ahead)", y1, full),
                        ("forecast: the 10-K after next (two years ahead)", y2, full),
                        ("forecast one year ahead, statements only (242)", y1, fin),
                        ("forecast one year ahead, next 10-K not yet restated-labelled: exclude filings already positive", np.where(y == 1, np.nan, y1), full)):
        ok = np.isfinite(yy)
        trn, tst = ok & (filed < TEST_START), ok & (filed >= TEST_START)
        # the forecast test set must end early enough for the next filing to exist and its label to mature: keep it as is, note the truncation
        st = fit_score(X, yy.astype(int), trn, tst)
        yt = yy[tst].astype(int); ty = years[tst]
        lo, hi = ci(yt, st)
        per = "  ".join(f"{yr}: {roc_auc_score(yt[ty == yr], st[ty == yr]):.3f}" for yr in sorted(set(ty)) if 0 < yt[ty == yr].sum() < (ty == yr).sum())
        print(f"{name:<52}{int(trn.sum()):>8,} ({int(yy[trn].sum()):>4}){int(tst.sum()):>8,} ({int(yt.sum()):>4}){roc_auc_score(yt, st):>10.3f}   [{lo:.3f}-{hi:.3f}]   {per}", flush=True)
    print("FORECAST DONE")


if __name__ == "__main__":
    main()
