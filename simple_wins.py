#!/usr/bin/env python3
"""Cheap, published modelling tricks, tested the usual way on the full free stack.

  1. undersampling ratio per tree: Perols et al. (2017) found ~20% positives per
     subset better than 50%; our balanced forest uses 1:1.
  2. hierarchical shrinkage (Agarwal et al. 2022): post-hoc regularisation of
     leaf values, one parameter.
  3. a measurement, not a model: AUC on first-time filers only (no prior Item
     4.02), Walker's (2022) serial-case point.

Selection on the 2017/2018 validation years, then the test set, paired against
the current forest.

  python simple_wins.py
"""

import os
import numpy as np
from sklearn.metrics import roc_auc_score
from imblearn.ensemble import BalancedRandomForestClassifier

from edge import impute
from events_test import joined
from final import TEST_START, ci, paired

VAL_YEARS = (2017, 2018)
JOBS = int(os.environ.get("ASOF_JOBS", "-1"))


def forest(seed, strategy="all", n=600, leaf=5):
    return BalancedRandomForestClassifier(n_estimators=n, min_samples_leaf=leaf, sampling_strategy=strategy,
                                          replacement=True, bootstrap=False, random_state=seed, n_jobs=JOBS)


def fit_score(make, X, y, trn, tst, seeds):
    Xtr, Xte = impute(X[trn], X[tst])
    return np.mean([make(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)


def hs_maker(lam, strategy="all"):
    from imodels import HSTreeClassifier
    def make(seed):
        return HSTreeClassifier(estimator_=forest(seed, strategy), reg_param=lam)
    return make


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    R, _ = joined(adsh, "data/out/features_raw.npz"); R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    E, _ = joined(adsh, "data/out/features_events.npz"); I, _ = joined(adsh, "data/out/features_insider.npz")
    M, mn = joined(adsh, "data/out/features_market.npz"); M = M[:, [i for i, n in enumerate(mn) if n != "price_cov"]]
    L, _ = joined(adsh, "data/out/features_letters.npz")
    S, sn = joined(adsh, os.environ.get("SCRUTINY_NPZ", "data/out/features_scrutiny.npz"))
    X = np.hstack([fin, E, I, M, L, S])
    trn, tst = filed < TEST_START, filed >= TEST_START
    yt, ty = y[tst], years[tst]
    print(f"stack {X.shape[1]} columns; train {int(trn.sum()):,} ({int(y[trn].sum())})  test {int(tst.sum()):,} ({int(yt.sum())})\n")

    variants = [("balanced forest, 1:1 per tree (current)", lambda s: forest(s)),
                ("1:2 per tree", lambda s: forest(s, 0.5)),
                ("1:4 per tree", lambda s: forest(s, 0.25)),
                ("1:10 per tree", lambda s: forest(s, 0.1)),
                ("1:4 per tree, 1,200 trees", lambda s: forest(s, 0.25, n=1200)),
                ("1:4 per tree, leaf 10", lambda s: forest(s, 0.25, leaf=10)),
                ("hierarchical shrinkage, lambda 2", hs_maker(2)),
                ("hierarchical shrinkage, lambda 10", hs_maker(10)),
                ("hierarchical shrinkage, lambda 50", hs_maker(50))]
    print(f"{'variant':<44}{'val 2017':>10}{'val 2018':>10}{'test AUC':>10}{'95% CI':>17}{'p@100':>7}{'  vs current (paired)':>24}{'P(<=0)':>8}")
    base = None
    for name, make in variants:
        try:
            vals = [roc_auc_score(y[years == v], fit_score(make, X, y, years < v, years == v, 3)) for v in VAL_YEARS]
            st = fit_score(make, X, y, trn, tst, 5)
        except Exception as exc:
            print(f"{name:<44} failed: {type(exc).__name__}: {str(exc)[:80]}", flush=True); continue
        lo, hi = ci(yt, st); order = np.argsort(-st)
        line = f"{name:<44}{vals[0]:>10.3f}{vals[1]:>10.3f}{roc_auc_score(yt, st):>10.3f}   [{lo:.3f}-{hi:.3f}]{yt[order[:100]].mean():>7.0%}"
        if base is None:
            base = st
        else:
            m, (dlo, dhi), p = paired(yt, base, st)
            line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>10.3f}"
        print(line, flush=True)

    # first-time filers only: Walker's serial-case objection to Bao et al.
    j = list(sn).index("prior_402")
    first = np.isfinite(S[tst][:, j]) & (S[tst][:, j] == 0)
    print(f"\ncurrent forest on test filings with no prior Item 4.02 in the panel: {int(first.sum()):,} filings, {int(yt[first].sum())} restated: "
          f"AUC {roc_auc_score(yt[first], base[first]):.3f} [{ci(yt[first], base[first])[0]:.3f}-{ci(yt[first], base[first])[1]:.3f}]")
    rep = ~first & np.isfinite(S[tst][:, j])
    print(f"filings with a prior Item 4.02: {int(rep.sum()):,}, {int(yt[rep].sum())} restated: AUC {roc_auc_score(yt[rep], base[rep]):.3f}")
    print("SIMPLE WINS DONE")


if __name__ == "__main__":
    main()
