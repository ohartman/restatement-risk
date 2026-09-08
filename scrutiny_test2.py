#!/usr/bin/env python3
"""The text flags on every test year, evaluated only where they exist.

After scrutiny_extend.py, the Item 9A / 8 / 3 flags exist for all 2014-2020
filings (EDGAR-CORPUS) and for the fetch sample in 2021-2023. Rows without flags
are dropped from BOTH training and test for every model here, so no model can
learn from a flag being missing. The 2021-23 rows are a label-stratified sample
(base rate ~25%), which leaves AUC comparable and p@100 not; p@100 is omitted.

  python scrutiny_test2.py
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
    E, _ = joined(adsh, "data/out/features_events.npz")
    I, _ = joined(adsh, "data/out/features_insider.npz")
    M, mn = joined(adsh, "data/out/features_market.npz"); M = M[:, [i for i, n in enumerate(mn) if n != "price_cov"]]
    L, _ = joined(adsh, "data/out/features_letters.npz")
    S, sn = joined(adsh, "data/out/features_scrutiny.npz")
    text_cols = list(range(10))
    has = np.isfinite(S[:, 0])
    best = np.hstack([fin, E, I, M, L])
    with_flags = np.hstack([best, S])

    trn, tst = has & (filed < TEST_START), has & (filed >= TEST_START)
    yt = y[tst]; ty = years[tst]
    print(f"rows with text flags: train {trn.sum():,} ({int(y[trn].sum())})  test {tst.sum():,} ({int(yt.sum())})")
    print("test by year: " + "  ".join(f"{yr}: {int((ty==yr).sum()):,} ({int(yt[ty==yr].sum())})" for yr in sorted(set(ty))))

    print(f"\n{'stage':<40}{'n':>5}{'val 2017':>10}{'val 2018':>10}{'test AUC':>10}{'95% CI':>17}{'  vs without (paired)':>24}{'P(<=0)':>8}")
    out = {}
    for name, X in (("financial (242)", fin), ("financial + text flags", np.hstack([fin, S])),
                    ("current best (280)", best), ("current best + scrutiny (295)", with_flags)):
        vals = [roc_auc_score(y[has & (years == v)], fit_score(X, y, has & (years < v), has & (years == v), seeds=3)) for v in VAL_YEARS]
        st = fit_score(X, y, trn, tst)
        out[name] = st
        lo, hi = ci(yt, st)
        line = f"{name:<40}{X.shape[1]:>5}{vals[0]:>10.3f}{vals[1]:>10.3f}{roc_auc_score(yt, st):>10.3f}   [{lo:.3f}-{hi:.3f}]"
        ref = {"financial + text flags": "financial (242)", "current best + scrutiny (295)": "current best (280)"}.get(name)
        if ref:
            m, (dlo, dhi), p = paired(yt, out[ref], st)
            line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>10.3f}"
        print(line, flush=True)
    a, b = out["current best (280)"], out["current best + scrutiny (295)"]
    print("\nper year: " + "  ".join(f"{yr}: {roc_auc_score(yt[ty==yr], a[ty==yr]):.3f}->{roc_auc_score(yt[ty==yr], b[ty==yr]):.3f}" for yr in sorted(set(ty))))
    np.savez("data/out/scores_scrutiny2.npz", adsh=adsh[tst], y=yt, **{f"s{k}": v for k, v in enumerate(out.values())})


if __name__ == "__main__":
    main()
