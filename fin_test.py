#!/usr/bin/env python3
"""Financial statements only: how much more is there?

Adds the extended block from raw_items2.py to the headline model in stages, so
the credit lands where it belongs:

  79            28 ratios + 17 raw items x3            (current headline, 0.717)
  + items       + 44 more raw items x3 and 15 ratios   (more of what already worked)
  + benford     + leading-digit and shape statistics   (the numbers on themselves)
  + filing      + lag, lateness, amendments, filer class (how the filing was made)

Each stage is scored on validation (2017-18, for an honest look at whether it
helps before the test set sees it) and on test, with the paired bootstrap of
(stage - 79) as the verdict. Everything here comes from the 10-K and its
metadata; no market, governance or event data is involved.

  python fin_test.py
"""

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from edge import impute
from events_test import joined
from final import TEST_START, ci, paired

VAL_START = np.datetime64("2017-01-01")


def brf(seed):
    return BalancedRandomForestClassifier(
        n_estimators=600, min_samples_leaf=5, sampling_strategy="all", replacement=True,
        bootstrap=False, random_state=seed, n_jobs=-1)


def fit_score(X, y, trn, tst, seeds=5):
    Xtr, Xte = impute(X[trn], X[tst])
    return np.mean([brf(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    X28, y, filed, adsh = z["X"], z["y"], z["filed"].astype("datetime64[D]"), z["adsh"]
    R, _ = joined(adsh, "data/out/features_raw.npz")
    R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    # sub.txt's prevrpt means "this submission was LATER amended" -- information
    # from after the filing date, and a near-proxy for the label. Never a feature.
    keep = [i for i, n in enumerate(n2) if n != "prevrpt"]
    R2, n2 = R2[:, keep], [n2[i] for i in keep]
    X79 = np.hstack([X28, R])
    is_items = np.array([not (n.startswith("benford") or n in
                         ("round_share", "n_facts", "n_custom_tags", "custom_share",
                          "filing_lag", "late_days", "afs", "wksi", "prevrpt", "nciks",
                          "foreign", "fye_changed", "n_amend_3y", "n_10k_3y")) for n in n2])
    is_benford = np.array([n.startswith("benford") or n in
                           ("round_share", "n_facts", "n_custom_tags", "custom_share") for n in n2])
    is_filing = ~is_items & ~is_benford
    stages = [
        ("79 (headline)", X79),
        ("+ items", np.hstack([X79, R2[:, is_items]])),
        ("+ items + benford", np.hstack([X79, R2[:, is_items | is_benford]])),
        ("+ items + benford + filing", np.hstack([X79, R2])),
        ("79 + benford only", np.hstack([X79, R2[:, is_benford]])),
        ("79 + filing only", np.hstack([X79, R2[:, is_filing]])),
    ]
    fm, vm = filed < VAL_START, (filed >= VAL_START) & (filed < TEST_START)
    trn, tst = filed < TEST_START, filed >= TEST_START
    yt = y[tst]
    years = filed[tst].astype("datetime64[Y]").astype(int) + 1970
    print(f"train {trn.sum():,} ({y[trn].sum()})   test {tst.sum():,} ({yt.sum()})\n")

    # Univariate look at the self-verification features before any model.
    for name in ("benford_mad", "round_share", "custom_share", "late_days", "n_amend_3y", "sloan_accrual"):
        j = list(n2).index(name)
        v = R2[:, j]
        ok = np.isfinite(v)
        print(f"  {name:<14} coverage {100*ok.mean():3.0f}%   AUC alone {roc_auc_score(y[ok], v[ok]):.3f}")
    print()

    base = None
    print(f"{'features':<30}{'n':>5}{'val AUC':>9}{'test AUC':>10}{'95% CI':>17}{'p@100':>8}"
          f"{'  vs 79 (paired)':>22}{'P(<=0)':>8}")
    results = {}
    for name, X in stages:
        sv = fit_score(X, y, fm, vm, seeds=3)
        st = fit_score(X, y, trn, tst)
        results[name] = st
        lo, hi = ci(yt, st)
        p100 = yt[np.argsort(-st)[:100]].mean()
        line = (f"{name:<30}{X.shape[1]:>5}{roc_auc_score(y[vm], sv):>9.3f}"
                f"{roc_auc_score(yt, st):>10.3f}   [{lo:.3f}-{hi:.3f}]{p100:>8.1%}")
        if base is None:
            base = st
        else:
            m, (dlo, dhi), p = paired(yt, base, st)
            line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>8.3f}"
        print(line, flush=True)

    best = max(results, key=lambda k: roc_auc_score(yt, results[k]))
    new = results[best]
    print(f"\nper year, 79 vs '{best}':")
    print(f"{'year':<10}{'n':>7}{'pos':>6}{'79':>9}{'new':>9}{'gain':>7}")
    for yr in sorted(set(years)):
        m = years == yr
        a, b = roc_auc_score(yt[m], base[m]), roc_auc_score(yt[m], new[m])
        print(f"{yr:<10}{m.sum():>7,}{int(yt[m].sum()):>6}{a:>9.3f}{b:>9.3f}{b-a:>+7.3f}")

    # Which of the new columns did the forest actually use?
    Xfull = np.hstack([X79, R2])
    Xtr, _ = impute(Xfull[trn], Xfull[tst])
    imp = brf(0).fit(Xtr, y[trn]).feature_importances_[:Xfull.shape[1]]
    names = [f"f{i}" for i in range(X28.shape[1])] + [f"raw:{i}" for i in range(R.shape[1])] + list(n2)
    order = np.argsort(-imp)
    print("\ntop 20 features by importance (full model):")
    for i in order[:20]:
        print(f"  {imp[i]:.4f}  {names[i]}")
    np.savez("data/out/scores_fin_test.npz", adsh=adsh[tst], y=yt,
             **{k.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", ""): v
                for k, v in results.items()})


if __name__ == "__main__":
    main()
