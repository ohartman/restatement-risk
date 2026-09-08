#!/usr/bin/env python3
"""Does "who is already worried" add to everything else?

Two blocks from scrutiny_features.py: history (prior Item 4.02s, last year's silent
revision, the industry's restatement wave -- available for every filing) and the
10-K's own text flags (controls not effective, material weakness, going concern,
Big 4, class action, SEC investigation -- available where EDGAR-CORPUS has the
filing, i.e. 2014-2020). Added in stages on top of the current best stack (280),
both validation years shown, test once, paired.

The text flags cover the training years fully but only 2019-2020 of the test
years, so the text stage is also reported on 2019-2020 alone, where every row
has it.

  python scrutiny_test.py
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
    hist_cols = [i for i, n in enumerate(sn) if n in ("prior_402", "days_since_402", "revised_last_year", "industry_wave_1y", "industry_wave_rate")]
    text_cols = [i for i in range(len(sn)) if i not in hist_cols]
    best = np.hstack([fin, E, I, M, L])
    stages = [("current best (280)", best),
              ("+ history: prior 4.02, last-year revision, industry wave", np.hstack([best, S[:, hist_cols]])),
              ("+ text flags: controls, auditor, legal", np.hstack([best, S[:, hist_cols], S])),
              ("financial (242) + history only", np.hstack([fin, S[:, hist_cols]]))]

    trn, tst = filed < TEST_START, filed >= TEST_START
    yt = y[tst]
    print(f"{'stage':<58}{'n':>5}{'val 2017':>10}{'val 2018':>10}{'test AUC':>10}{'95% CI':>17}{'p@100':>8}"
          f"{'  vs previous (paired)':>26}{'P(<=0)':>8}")
    prev, out = None, {}
    for k, (name, X) in enumerate(stages):
        vals = [roc_auc_score(y[years == v], fit_score(X, y, years < v, years == v, seeds=3)) for v in VAL_YEARS]
        st = fit_score(X, y, trn, tst)
        out[name] = st
        lo, hi = ci(yt, st)
        line = (f"{name:<58}{X.shape[1]:>5}{vals[0]:>10.3f}{vals[1]:>10.3f}{roc_auc_score(yt, st):>10.3f}"
                f"   [{lo:.3f}-{hi:.3f}]{yt[np.argsort(-st)[:100]].mean():>8.1%}")
        ref = out["current best (280)"] if k in (1, 2) else (out["financial (242) + history only"] if False else None)
        if k == 1:
            ref = out["current best (280)"]
        elif k == 2:
            ref = out["+ history: prior 4.02, last-year revision, industry wave"]
        if ref is not None:
            m, (dlo, dhi), p = paired(yt, ref, st)
            line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>12.3f}"
        print(line, flush=True)

    # the text stage where every test row has the flags
    cov = tst & np.isfinite(S[:, text_cols[0]])
    ty = years[tst]
    print(f"\ntext flags cover {cov.sum():,} test filings ({sorted(set(years[cov]))}); on those rows:")
    a = out["+ history: prior 4.02, last-year revision, industry wave"][np.isfinite(S[tst][:, text_cols[0]])]
    b = out["+ text flags: controls, auditor, legal"][np.isfinite(S[tst][:, text_cols[0]])]
    yc = yt[np.isfinite(S[tst][:, text_cols[0]])]
    m, (dlo, dhi), p = paired(yc, a, b)
    print(f"  without text flags {roc_auc_score(yc, a):.3f}   with {roc_auc_score(yc, b):.3f}   paired {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]  P(<=0) {p:.3f}")
    a0, b0 = out["current best (280)"], out["+ history: prior 4.02, last-year revision, industry wave"]
    print("\nper year, current best -> + history: " + "  ".join(
        f"{yr}: {roc_auc_score(yt[ty==yr], a0[ty==yr]):.3f}->{roc_auc_score(yt[ty==yr], b0[ty==yr]):.3f}" for yr in sorted(set(ty))))
    np.savez("data/out/scores_scrutiny.npz", adsh=adsh[tst], y=yt,
             **{f"s{k}": v for k, v in enumerate(out.values())})


if __name__ == "__main__":
    main()
