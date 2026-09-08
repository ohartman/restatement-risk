#!/usr/bin/env python3
"""Do SEC comment letters and late-filing notices add to the financial statements?

Same harness as every other block: the 242-feature forest with and without the
letters block, both validation years shown before the test set, 2019-2023 once,
paired bootstrap. Then the same on top of the full free stack (events, insider,
market) to see what survives when everything else is present.

  python letters_test.py
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
    L, lnames = joined(adsh, "data/out/features_letters.npz")
    E, _ = joined(adsh, "data/out/features_events.npz")
    I, _ = joined(adsh, "data/out/features_insider.npz")
    M, mn = joined(adsh, "data/out/features_market.npz")
    M = M[:, [i for i, n in enumerate(mn) if n != "price_cov"]]
    stack = np.hstack([fin, E, I, M])

    trn, tst = filed < TEST_START, filed >= TEST_START
    yt = y[tst]
    stages = [("financial (242)", fin), ("financial + letters", np.hstack([fin, L])),
              ("full free stack (270)", stack), ("full free stack + letters", np.hstack([stack, L]))]
    print(f"{'stage':<30}{'n':>5}{'val 2017':>10}{'val 2018':>10}{'test AUC':>10}{'95% CI':>17}{'p@100':>8}"
          f"{'  vs previous (paired)':>26}{'P(<=0)':>8}")
    prev = None
    out = {}
    for name, X in stages:
        vals = [roc_auc_score(y[years == v], fit_score(X, y, years < v, years == v, seeds=3)) for v in VAL_YEARS]
        st = fit_score(X, y, trn, tst)
        out[name] = st
        lo, hi = ci(yt, st)
        line = (f"{name:<30}{X.shape[1]:>5}{vals[0]:>10.3f}{vals[1]:>10.3f}{roc_auc_score(yt, st):>10.3f}"
                f"   [{lo:.3f}-{hi:.3f}]{yt[np.argsort(-st)[:100]].mean():>8.1%}")
        if prev is not None and name.endswith("letters"):
            m, (dlo, dhi), p = paired(yt, prev, st)
            line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>12.3f}"
        print(line, flush=True)
        prev = st if not name.endswith("letters") else prev
    a, b = out["financial (242)"], out["financial + letters"]
    ty = years[tst]
    print("\nper year, financial vs financial + letters: " + "  ".join(
        f"{yr}: {roc_auc_score(yt[ty==yr], a[ty==yr]):.3f}->{roc_auc_score(yt[ty==yr], b[ty==yr]):.3f}" for yr in sorted(set(ty))))
    np.savez("data/out/scores_letters.npz", adsh=adsh[tst], y=yt,
             **{k.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", ""): v for k, v in out.items()})


if __name__ == "__main__":
    main()
