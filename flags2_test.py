#!/usr/bin/env python3
"""Do the thirty new disclosure flags add to the ten that already worked?

Rows with flags only (both blocks share coverage), both validation years, test
once, paired against the stack that already includes the first flags.

  python flags2_test.py
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
    F, fn = joined(adsh, "data/out/features_flags2.npz")
    has = np.isfinite(S[:, 0]) & np.isfinite(F[:, 0])
    base = np.hstack([fin, E, I, M, L, S])
    trn, tst = has & (filed < TEST_START), has & (filed >= TEST_START)
    yt = y[tst]; ty = years[tst]
    print(f"rows with both flag blocks: train {trn.sum():,} ({int(y[trn].sum())})  test {tst.sum():,} ({int(yt.sum())})  "
          f"test years {sorted(set(ty))}\n")
    print(f"{'stage':<40}{'n':>5}{'val 2017':>10}{'val 2018':>10}{'test AUC':>10}{'95% CI':>17}{'  vs previous (paired)':>26}{'P(<=0)':>8}")
    prev = None
    for name, X in (("stack + first flags (295)", base), ("+ thirty disclosure flags", np.hstack([base, F]))):
        vals = [roc_auc_score(y[has & (years == v)], fit_score(X, y, has & (years < v), has & (years == v), seeds=3)) for v in VAL_YEARS]
        st = fit_score(X, y, trn, tst)
        lo, hi = ci(yt, st)
        line = f"{name:<40}{X.shape[1]:>5}{vals[0]:>10.3f}{vals[1]:>10.3f}{roc_auc_score(yt, st):>10.3f}   [{lo:.3f}-{hi:.3f}]"
        if prev is not None:
            m, (dlo, dhi), p = paired(yt, prev, st)
            line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>12.3f}"
            print(line); print("per year: " + "  ".join(f"{yr}: {roc_auc_score(yt[ty==yr], prev[ty==yr]):.3f}->{roc_auc_score(yt[ty==yr], st[ty==yr]):.3f}" for yr in sorted(set(ty))))
        else:
            print(line, flush=True)
        prev = st
    # which of the new flags did the forest use?
    Xtr, _ = impute(np.hstack([base, F])[trn], np.hstack([base, F])[tst])
    imp = brf(0).fit(Xtr, y[trn]).feature_importances_[:base.shape[1] + F.shape[1]]
    new = imp[base.shape[1]:]
    order = np.argsort(-new)[:10]
    print("\nmost used new flags: " + ", ".join(f"{fn[i]} ({new[i]:.4f})" for i in order))


if __name__ == "__main__":
    main()
