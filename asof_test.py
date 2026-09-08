#!/usr/bin/env python3
"""Does the headline survive knowing only what a forecaster would have known?

The headline trains once on filings through 2018 and tests on 2019-2023. Its
training labels use every Item 4.02 through the end of the panel, so a 2018
10-K is labelled positive by a 2020 announcement that nobody standing on
1 January 2019 had seen. Two stricter protocols:

  A. fixed split, labels as of the cutoff: train on filings before 2019 with
     labels truncated at 2019-01-01, test 2019-2023 with the full labels.
  B. one year ahead, refit yearly: for each test year t, train on filings
     before t with labels truncated at t-01-01, score year t.
  C. one year ahead with hindsight labels: the refit alone, without the
     truncation, to separate the two effects.

  python asof_test.py
"""

import os
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score
from imblearn.ensemble import BalancedRandomForestClassifier

from edge import impute
from events_test import joined
from final import TEST_START, ci, paired
from relabel import submissions
from train import load_labels

WINDOW = 1095
JOBS = int(os.environ.get("ASOF_JOBS", "6"))


def brf(seed):
    from events_test import strategy, max_features
    return BalancedRandomForestClassifier(n_estimators=600, min_samples_leaf=5, sampling_strategy=strategy(), max_features=max_features(),
                                          replacement=True, bootstrap=False, random_state=seed, n_jobs=JOBS)


PLAIN = [0]     # width of the audited market block, imputed without indicators, kept last


def fit_score(X, y, trn, tst, seeds=3):
    Xtr, Xte = impute(X[trn], X[tst], plain_last=PLAIN[0] if X.shape[1] > 242 else 0)
    return np.mean([brf(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)


def asof_labels(adsh, filed):
    """Returns asof(cutoff): the loose label using only Item 4.02s announced before the cutoff."""
    subs = submissions(); labels = load_labels()
    fdates = [datetime.strptime(str(d), "%Y-%m-%d").date() for d in filed.astype(str)]
    ciks = [subs.get(a, (None,))[0] for a in adsh.tolist()]

    def asof(cutoff):
        c = datetime.strptime(str(cutoff), "%Y-%m-%d").date()
        out = np.zeros(len(adsh), dtype=int)
        for i, (cik, fd) in enumerate(zip(ciks, fdates)):
            if cik and any(0 <= (x - fd).days <= WINDOW and x < c for x in labels.get(cik, [])):
                out[i] = 1
        return out
    return asof


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    R, _ = joined(adsh, "data/out/features_raw.npz"); R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    E, _ = joined(adsh, "data/out/features_events.npz"); I, _ = joined(adsh, "data/out/features_insider.npz")
    from events_test import market_block, scrutiny_block
    M = market_block(adsh); PLAIN[0] = M.shape[1]                       # audited block, imputed without indicators, kept last
    L, _ = joined(adsh, "data/out/features_letters.npz"); S = scrutiny_block(adsh, os.environ.get("SCRUTINY_NPZ", "data/out/features_scrutiny.npz"))
    X = np.hstack([fin, E, I, L, S, M])
    print(f"stack: {X.shape[1]} columns")

    asof = asof_labels(adsh, filed)

    trn, tst = filed < TEST_START, filed >= TEST_START
    yt, ty = y[tst], years[tst]
    y19 = asof(TEST_START)
    print(f"training positives with full labels: {int(y[trn].sum()):,}; known on {TEST_START}: {int(y19[trn].sum()):,} "
          f"({int(y[trn].sum() - y19[trn].sum()):,} of them are announced only after the cutoff)")
    for yr in (2016, 2017, 2018):
        m = years == yr
        print(f"  filed {yr}: {int(y[m].sum()):>4} positives, {int(y19[m].sum()):>4} known by 2019-01-01")

    print("\n=== A. fixed split, train < 2019 ===", flush=True)
    base = fit_score(X, y, trn, tst)
    strict = fit_score(X, y19, trn, tst)
    for name, s in (("hindsight labels (the headline)", base), ("labels as of 2019-01-01", strict)):
        lo, hi = ci(yt, s)
        print(f"  {name:<34} AUC {roc_auc_score(yt, s):.3f} [{lo:.3f}-{hi:.3f}]  " +
              "  ".join(f"{yr}: {roc_auc_score(yt[ty == yr], s[ty == yr]):.3f}" for yr in sorted(set(ty))))
    m, (lo, hi), p = paired(yt, base, strict)
    print(f"  as-of minus hindsight: {m:+.3f} [{lo:+.3f},{hi:+.3f}]  P(<=0) {p:.3f}")

    print("\n=== B/C. one year ahead, refit each year ===", flush=True)
    roll_strict = np.full(tst.sum(), np.nan); roll_hind = np.full(tst.sum(), np.nan)
    pos = {i: k for k, i in enumerate(np.flatnonzero(tst))}
    for t in sorted(set(ty)):
        cut = np.datetime64(f"{t}-01-01")
        tr_m, te_m = filed < cut, years == t
        ya = asof(cut)
        sb = fit_score(X, ya, tr_m, te_m); sc = fit_score(X, y, tr_m, te_m)
        idx = np.array([pos[i] for i in np.flatnonzero(te_m)])
        roll_strict[idx] = sb; roll_hind[idx] = sc
        yy = y[te_m]
        print(f"  {t}: train {int(tr_m.sum()):,} filings ({int(ya[tr_m].sum()):,} positives known, {int(y[tr_m].sum()):,} with hindsight)"
              f"  fixed/hindsight {roc_auc_score(yy, base[idx]):.3f}  refit/as-of {roc_auc_score(yy, sb):.3f}  refit/hindsight {roc_auc_score(yy, sc):.3f}", flush=True)
    print("\n  pooled over 2019-2023 (each year scored by that year's model; ranks only comparable within a year for the refits):")
    for name, s in (("fixed, hindsight (headline)", base), ("refit yearly, labels as of each 1 Jan", roll_strict), ("refit yearly, hindsight labels", roll_hind)):
        lo, hi = ci(yt, s)
        per = [roc_auc_score(yt[ty == yr], s[ty == yr]) for yr in sorted(set(ty))]
        print(f"  {name:<42} pooled {roc_auc_score(yt, s):.3f} [{lo:.3f}-{hi:.3f}]   mean of yearly {np.mean(per):.3f}")
    for name, s in (("refit/as-of", roll_strict), ("refit/hindsight", roll_hind)):
        m, (lo, hi), p = paired(yt, base, s)
        print(f"  {name} minus headline (pooled, paired): {m:+.3f} [{lo:+.3f},{hi:+.3f}]  P(<=0) {p:.3f}")
    np.savez("data/out/scores_asof.npz", adsh=adsh[tst], y=yt, base=base, strict=strict, roll_strict=roll_strict, roll_hind=roll_hind)


if __name__ == "__main__":
    main()
