#!/usr/bin/env python3
"""Four ideas aimed at what the as-of table exposed, tested on the tuned forest.

  I1. censoring-aware negatives (as-of protocol). A filing labelled clean on the cutoff date
      is only as clean as its label is mature: 2016 filings had 1,000 days for an announcement
      to arrive, 2018 filings a few months. Weight each as-of negative by the empirical share
      of eventual positives announced within its elapsed time (a maturity curve estimated on
      the oldest training years), or drop the least mature. Honest by construction: uses only
      what was known on the cutoff date. Compared against plain as-of training (0.725).
  I2. multi-horizon ensemble (headline protocol). Separate forests for restated within 1, 2
      and 3 years, rank-averaged, scored on the 3-year label. Shorter horizons have fewer
      immature labels.
  I3. firm lag block (headline protocol). Each of the 242 statement features minus the same
      firm's previous 10-K value: 242 more columns.
  I4. era-wise ranks (headline protocol). Every continuous column replaced by its percentile
      within its filing year, so drift in scale (size, inflation, tag usage) cannot be learned.

  BRF_STRATEGY=0.1 BRF_MAX_FEATURES=15 SCRUTINY_NPZ=... FLAGS2_NPZ=... python innovate.py
"""
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score

from asof_test import asof_labels
from edge import impute
from events_test import brf, joined
from final import TEST_START, ci, paired
from relabel import submissions
from train import load_labels

VAL_YEARS = (2017, 2018)


def rank(v):
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


def fit(X, y, trn, tst, seeds=5, w=None):
    Xtr, Xte = impute(X[trn], X[tst])
    out = []
    for s in range(seeds):
        m = brf(s)
        m.fit(Xtr, y[trn], sample_weight=None if w is None else w[trn])
        out.append(m.predict_proba(Xte)[:, 1])
    return np.mean(out, axis=0)


def report(name, yt, st, base, ty=None):
    lo, hi = ci(yt, st)
    line = f"{name:<62}{roc_auc_score(yt, st):>8.3f}  [{lo:.3f}-{hi:.3f}]"
    if base is not None:
        m, (dlo, dhi), p = paired(yt, base, st)
        line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]  P {p:.3f}"
    if ty is not None:
        line += "   " + " ".join(f"{yr}:{roc_auc_score(yt[ty == yr], st[ty == yr]):.3f}" for yr in sorted(set(ty)))
    print(line, flush=True)


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    R, _ = joined(adsh, "data/out/features_raw.npz")
    R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    E, _ = joined(adsh, "data/out/features_events.npz")
    I, _ = joined(adsh, "data/out/features_insider.npz")
    M, mn = joined(adsh, "data/out/features_market.npz")
    M = M[:, [i for i, n in enumerate(mn) if n != "price_cov"]]
    L, _ = joined(adsh, "data/out/features_letters.npz")
    S, _ = joined(adsh, os.environ["SCRUTINY_NPZ"])
    F, _ = joined(adsh, os.environ["FLAGS2_NPZ"])
    X = np.hstack([fin, E, I, M, L, S, F])
    trn, tst = filed < TEST_START, filed >= TEST_START
    yt, ty = y[tst], years[tst]
    print(f"stack {X.shape[1]} columns; train {int(trn.sum()):,} ({int(y[trn].sum())})  test {int(tst.sum()):,} ({int(yt.sum())})\n", flush=True)
    subs = submissions()
    labels = load_labels()
    fd = [datetime.strptime(str(d), "%Y-%m-%d").date() for d in filed.astype(str)]
    ciks = [subs.get(a, (None,))[0] for a in adsh.tolist()]
    cutoff = datetime.strptime(str(TEST_START), "%Y-%m-%d").date()

    # ---------- I1. censoring-aware negatives under the as-of protocol ----------
    print("I1. as-of protocol: training labels as known on 2019-01-01")
    y_asof = asof_labels(adsh, filed)(TEST_START)
    lags = []
    for i in range(len(adsh)):
        if y[i] == 1 and years[i] <= 2015 and ciks[i]:
            d = [(x - fd[i]).days for x in labels.get(ciks[i], []) if 0 <= (x - fd[i]).days <= 1095]
            if d:
                lags.append(min(d))
    lags = np.array(sorted(lags))

    def maturity(days):
        return np.searchsorted(lags, days, side="right") / len(lags)

    elapsed = np.array([(cutoff - d).days for d in fd])
    print(f"  maturity curve from {len(lags):,} positives filed 2014-15: announced within 180 d {maturity(180):.0%}, "
          f"365 d {maturity(365):.0%}, 730 d {maturity(730):.0%}, 1,095 d {maturity(1095):.0%}")
    for yr in (2016, 2017, 2018):
        m = years == yr
        med = int(np.median(elapsed[m]))
        print(f"  filings of {yr}: elapsed {med} d at the cutoff, label maturity {maturity(med):.0%}")
    base_asof = fit(X, y_asof, trn, tst)
    report("plain as-of labels (the deployable number)", yt, base_asof, None, ty)
    mat = np.clip(maturity(np.clip(elapsed, 0, 1095)), 0.05, 1.0)
    w = np.where(y_asof == 1, 1.0, mat)
    report("as-of, negatives weighted by label maturity", yt, fit(X, y_asof, trn, tst, w=w), base_asof, ty)
    w2 = np.where(y_asof == 1, 1.0, mat ** 2)
    report("as-of, negatives weighted by maturity squared", yt, fit(X, y_asof, trn, tst, w=w2), base_asof, ty)
    keep = trn & ~((y_asof == 0) & (elapsed < 365))
    report("as-of, negatives under one year old dropped", yt, fit(X, y_asof, keep, tst), base_asof, ty)
    keep2 = trn & ~((y_asof == 0) & (elapsed < 730))
    report("as-of, negatives under two years old dropped", yt, fit(X, y_asof, keep2, tst), base_asof, ty)
    report("headline protocol (hindsight labels), for reference", yt, fit(X, y, trn, tst), base_asof, ty)

    # ---------- headline-protocol ideas ----------
    print("\nheadline protocol: hindsight labels, one fixed model")
    base = fit(X, y, trn, tst)
    report("tuned forest, 330 stack (the headline)", yt, base, None, ty)

    def horizon(h):
        out = np.zeros(len(adsh), dtype=int)
        for i in range(len(adsh)):
            if ciks[i] and any(0 <= (x - fd[i]).days <= h for x in labels.get(ciks[i], [])):
                out[i] = 1
        return out

    y1, y2 = horizon(365), horizon(730)
    print(f"  positives within 1 / 2 / 3 years in training: {int(y1[trn].sum())} / {int(y2[trn].sum())} / {int(y[trn].sum())}")
    s1, s2 = fit(X, y1, trn, tst), fit(X, y2, trn, tst)
    report("I2. 1-year-horizon forest alone, scored on the 3-year label", yt, s1, base, ty)
    report("I2. rank-average of 1-, 2-, 3-year-horizon forests", yt, (rank(s1) + rank(s2) + rank(base)) / 3, base, ty)

    by_cik = defaultdict(list)
    for i, c in enumerate(ciks):
        if c:
            by_cik[c].append((filed[i], i))
    prev = np.full(len(adsh), -1)
    for c, lst in by_cik.items():
        lst.sort()
        for k in range(1, len(lst)):
            gap = (lst[k][0] - lst[k - 1][0]).astype(int)
            if 270 <= gap <= 460:
                prev[lst[k][1]] = lst[k - 1][1]
    D = np.full(fin.shape, np.nan)
    has = prev >= 0
    D[has] = fin[has] - fin[prev[has]]
    print(f"  filings with a prior-year 10-K in the panel: {int(has.sum()):,}")
    report("I3. + 242 firm lag differences (572 columns)", yt, fit(np.hstack([X, D]), y, trn, tst), base, ty)

    XR = X.copy()
    for yr in sorted(set(years)):
        m = years == yr
        for j in range(X.shape[1]):
            col = X[m, j]
            ok = np.isfinite(col)
            if ok.sum() > 20 and len(np.unique(col[ok])) > 2:
                r = np.full(col.shape, np.nan)
                r[ok] = rank(col[ok])
                XR[m, j] = r
    report("I4. every continuous column ranked within its filing year", yt, fit(XR, y, trn, tst), base, ty)
    print("INNOVATE DONE")


if __name__ == "__main__":
    main()
