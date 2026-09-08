#!/usr/bin/env python3
"""Late fusion of the forest and the fine-tuned MD&A model, weight chosen on validation.

Stacking BERT's score into the forest would leak: both are trained on the same
years, so BERT's scores on training filings are in-sample. Late fusion avoids it.
The forest trained on 2014-16 and BERT trained on 2014-16 are both out-of-sample
on 2017-18; the blend weight that maximises AUC there is chosen once. Then the
forest is refit on 2014-18, BERT's saved scores are used as they are, and the
blend is scored on 2019-2023 filings that have text -- once, paired against the
forest alone on the same filings.

  python fusion_test.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from edge import impute
from events_test import joined
from final import TEST_START, ci, paired
from stack_test import brf


def rank(v):
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


def fit_score(X, y, trn, tst, seeds=5):
    Xtr, Xte = impute(X[trn], X[tst])
    return np.mean([brf(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    R, _ = joined(adsh, "data/out/features_raw.npz")
    R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    Xfin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/out/bert_score.npy"
    bert = np.load(path)
    print(f"scores: {path}")
    has = np.isfinite(bert)
    print(f"BERT scores for {has.sum():,} filings")

    # validation: both models trained on 2014-16
    vm = has & np.isin(years, (2017, 2018))
    f_val = fit_score(Xfin, y, years < 2017, vm)
    print(f"validation 2017-18 (n={vm.sum():,}, restated {int(y[vm].sum())}):  forest {roc_auc_score(y[vm], f_val):.3f}"
          f"   BERT {roc_auc_score(y[vm], bert[vm]):.3f}")
    best = (roc_auc_score(y[vm], f_val), 0.0)
    for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        a = roc_auc_score(y[vm], (1 - w) * rank(f_val) + w * rank(bert[vm]))
        print(f"  weight on BERT {w:.1f}  val AUC {a:.3f}")
        if a > best[0]:
            best = (a, w)
    w = best[1]
    print(f"chosen weight on BERT: {w:.1f}")

    # test: forest refit on 2014-18; BERT scores as saved; filings with text only
    tm = has & (filed >= TEST_START)
    f_te = fit_score(Xfin, y, filed < TEST_START, tm)
    yt = y[tm]
    fused = (1 - w) * rank(f_te) + w * rank(bert[tm])
    print(f"\ntest 2019-23, filings with MD&A text (n={tm.sum():,}, restated {int(yt.sum())}):")
    for name, s in (("forest alone", f_te), ("BERT alone", bert[tm]), (f"fusion (w={w:.1f})", fused)):
        lo, hi = ci(yt, s)
        print(f"  {name:<18} AUC {roc_auc_score(yt, s):.3f}  [{lo:.3f}-{hi:.3f}]  p@100 {yt[np.argsort(-s)[:100]].mean():.0%}")
    m, (lo, hi), p = paired(yt, f_te, fused)
    print(f"  fusion vs forest: {m:+.3f} [{lo:+.3f},{hi:+.3f}]  P(<=0) {p:.3f}")
    ty = years[tm]
    print("  per year: " + "  ".join(f"{yr}: {roc_auc_score(yt[ty==yr], f_te[ty==yr]):.3f}->{roc_auc_score(yt[ty==yr], fused[ty==yr]):.3f}"
                                    for yr in sorted(set(ty)) if 0 < yt[ty == yr].sum() < (ty == yr).sum()))
    np.savez("data/out/scores_fusion.npz", adsh=adsh[tm], y=yt, forest=f_te, bert=bert[tm], fused=fused, w=w)


if __name__ == "__main__":
    main()
