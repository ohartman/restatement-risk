#!/usr/bin/env python3
"""Does the year's own arithmetic add to everything else?

The quarterly block on top of the full stack (statements, events, insider,
market, letters, and the 10-K text flags where they exist), both validation
years, test once, paired. Evaluated on all test rows (the quarterly features
are NaN only where a firm's 10-Qs are missing, which is itself a feature) and
on the rows with text flags.

  python quarterly_test.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from edge import impute
from events_test import brf, joined
from final import TEST_START, ci, paired

VAL_YEARS = (2017, 2018)


def fit_score(X, y, trn, tst, seeds=5):
    Xtr, Xte = impute(X[trn], X[tst])
    return np.mean([brf(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    R, _ = joined(adsh, "data/out/features_raw.npz")
    R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    E, _ = joined(adsh, "data/out/features_events.npz"); I, _ = joined(adsh, "data/out/features_insider.npz")
    M, mn = joined(adsh, "data/out/features_market.npz"); M = M[:, [i for i, n in enumerate(mn) if n != "price_cov"]]
    L, _ = joined(adsh, "data/out/features_letters.npz")
    S, _ = joined(adsh, "data/out/features_scrutiny.npz")
    Q, qn = joined(adsh, "data/out/features_quarterly.npz")
    stack = np.hstack([fin, E, I, M, L, S])
    trn, tst = filed < TEST_START, filed >= TEST_START
    yt = y[tst]; ty = years[tst]
    for label, rows in (("all test rows", np.ones(len(y), bool)), ("rows with text flags", np.isfinite(S[:, 0]))):
        tr, te = trn & rows, tst & rows
        print(f"\n=== {label}: train {tr.sum():,} ({int(y[tr].sum())})  test {te.sum():,} ({int(y[te].sum())}) ===")
        print(f"{'stage':<34}{'n':>5}{'val 2017':>10}{'val 2018':>10}{'test AUC':>10}{'95% CI':>17}{'  vs previous (paired)':>26}{'P(<=0)':>8}")
        prev = None
        for name, X in (("full stack", stack), ("+ quarterly arithmetic", np.hstack([stack, Q]))):
            vals = [roc_auc_score(y[rows & (years == v)], fit_score(X, y, rows & (years < v), rows & (years == v), seeds=3)) for v in VAL_YEARS]
            st = fit_score(X, y, tr, te)
            lo, hi = ci(y[te], st)
            line = f"{name:<34}{X.shape[1]:>5}{vals[0]:>10.3f}{vals[1]:>10.3f}{roc_auc_score(y[te], st):>10.3f}   [{lo:.3f}-{hi:.3f}]"
            if prev is not None:
                m, (dlo, dhi), p = paired(y[te], prev, st)
                line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>12.3f}"
                print(line)
                tyy = years[te]
                print("per year: " + "  ".join(f"{yr}: {roc_auc_score(y[te][tyy==yr], prev[tyy==yr]):.3f}->{roc_auc_score(y[te][tyy==yr], st[tyy==yr]):.3f}" for yr in sorted(set(tyy))))
            else:
                print(line, flush=True)
            prev = st
    Xtr, _ = impute(np.hstack([stack, Q])[trn], np.hstack([stack, Q])[tst])
    imp = brf(0).fit(Xtr, y[trn]).feature_importances_[stack.shape[1]:stack.shape[1] + Q.shape[1]]
    print("\nquarterly features by importance: " + ", ".join(f"{qn[i]} ({imp[i]:.4f})" for i in np.argsort(-imp)))


if __name__ == "__main__":
    main()
