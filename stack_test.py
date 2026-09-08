#!/usr/bin/env python3
"""Two algorithmic ideas on top of the headline forest, tested the same honest way.

1. Stack little r instead of merging it. Merging the silent-revision label into
   the target hurt (0.730 -> 0.70). Here a forest is trained on the little-r
   label (net income off by >5% in the next year's comparatives), its
   out-of-fold score (five folds by company) becomes ONE column, and the Big-R
   forest gets that column. Same target, one more view.

2. Firm-level persistence. Restating is a firm trait; a firm's earlier 10-Ks
   carry information about this one. Each filing's score is averaged with the
   model's scores for the same firm's previous filings (decayed), all computed
   from features that were public at the time.

3. Interlock features from interlock_features.py, if present.

Each is selected on both validation years (2017, 2018) and reported on
2019-2023 once, paired against the 242-feature financial forest.

  python stack_test.py
"""

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from edge import impute
from events_test import joined
from final import TEST_START, ci, paired
from littler2 import collect, label
from relabel import submissions

VAL_YEARS = (2017, 2018)


def brf(seed, n=600):
    return BalancedRandomForestClassifier(n_estimators=n, min_samples_leaf=5, sampling_strategy="all",
                                          replacement=True, bootstrap=False, random_state=seed, n_jobs=-1)


def fit_score(X, y, trn, tst, seeds=5):
    Xtr, Xte = impute(X[trn], X[tst])
    return np.mean([brf(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh_arr = z["adsh"]; adsh = adsh_arr.tolist()
    y, filed = z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    subs = submissions()
    ciks = np.array([subs.get(a, ("",))[0] for a in adsh])
    R, _ = joined(adsh_arr, "data/out/features_raw.npz")
    R2, n2 = joined(adsh_arr, "data/out/features_raw2.npz")
    Xfin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    trn, tst = filed < TEST_START, filed >= TEST_START
    yt = y[tst]

    # ---- 1. little-r score, out of fold by company on train; fit on all train for test ----
    own, comp, meta = collect()
    y_r = label(own, comp, meta, adsh, ("net_income",), 0.05, 1e5, False)
    has_r = y_r >= 0
    rcol = np.full(len(adsh), np.nan)
    tr_r = np.flatnonzero(trn & has_r)
    Xtr_i, Xte_i = impute(Xfin[tr_r], Xfin[tst])
    oof = np.zeros(len(tr_r))
    for f_tr, f_va in GroupKFold(5).split(Xtr_i, y_r[tr_r], ciks[tr_r]):
        m = brf(0, 300).fit(Xtr_i[f_tr], y_r[tr_r][f_tr])
        oof[f_va] = m.predict_proba(Xtr_i[f_va])[:, 1]
    rcol[tr_r] = oof
    rcol[tst] = np.mean([brf(s, 300).fit(Xtr_i, y_r[tr_r]).predict_proba(Xte_i)[:, 1] for s in range(3)], axis=0)
    print(f"little-r model: out-of-fold AUC on its own label {roc_auc_score(y_r[tr_r], oof):.3f}; "
          f"as a predictor of Big R on train {roc_auc_score(y[tr_r], oof):.3f}")
    Xstack = np.hstack([Xfin, rcol.reshape(-1, 1)])

    # ---- 3. interlocks ----
    stages = [("financial (242)", Xfin), ("+ little-r score", Xstack)]
    try:
        I, inames = joined(adsh_arr, "data/out/features_interlock.npz")
        stages.append(("+ little-r score + interlocks", np.hstack([Xstack, I])))
        stages.append(("financial + interlocks only", np.hstack([Xfin, I])))
    except FileNotFoundError:
        print("(no interlock features yet)")

    print(f"\n{'stage':<34}{'n':>5}{'val 2017':>10}{'val 2018':>10}{'test AUC':>10}{'95% CI':>17}{'p@100':>8}"
          f"{'  vs financial (paired)':>26}{'P(<=0)':>8}")
    base = None
    scores = {}
    for name, X in stages:
        vals = [roc_auc_score(y[years == v], fit_score(X, y, years < v, years == v, seeds=3)) for v in VAL_YEARS]
        st = fit_score(X, y, trn, tst)
        scores[name] = st
        lo, hi = ci(yt, st)
        line = (f"{name:<34}{X.shape[1]:>5}{vals[0]:>10.3f}{vals[1]:>10.3f}{roc_auc_score(yt, st):>10.3f}"
                f"   [{lo:.3f}-{hi:.3f}]{yt[np.argsort(-st)[:100]].mean():>8.1%}")
        if base is None:
            base = st
        else:
            m, (dlo, dhi), p = paired(yt, base, st)
            line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>12.3f}"
        print(line, flush=True)

    # ---- 2. firm persistence: decayed average with the firm's earlier filings' scores ----
    # Score every filing (train out-of-fold by company, test from the full train model).
    print("\nfirm persistence (scores of the firm's earlier 10-Ks folded in):")
    Xtr_i, Xte_i = impute(Xfin[trn], Xfin[tst])
    full = np.full(len(adsh), np.nan)
    tr_idx = np.flatnonzero(trn)
    oof = np.zeros(len(tr_idx))
    for f_tr, f_va in GroupKFold(5).split(Xtr_i, y[trn], ciks[trn]):
        oof[f_va] = brf(0, 300).fit(Xtr_i[f_tr], y[trn][f_tr]).predict_proba(Xtr_i[f_va])[:, 1]
    full[tr_idx] = oof
    full[tst] = base
    order = np.argsort(filed, kind="stable")
    prev_scores = {}
    for w_prev in (0.0, 0.3, 0.5, 0.7):
        sm = full.copy()
        last = {}
        for i in order:
            c = ciks[i]
            if c in last and np.isfinite(full[i]):
                sm[i] = (full[i] + w_prev * last[c]) / (1 + w_prev)
            if np.isfinite(sm[i]):
                last[c] = sm[i]
        vals = [roc_auc_score(y[years == v], sm[years == v]) for v in VAL_YEARS]
        m, (dlo, dhi), p = paired(yt, base, sm[tst])
        print(f"  weight on firm's past {w_prev:.1f}   val 2017 {vals[0]:.3f}  val 2018 {vals[1]:.3f}   "
              f"test {roc_auc_score(yt, sm[tst]):.3f}   vs none {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]  P(<=0) {p:.3f}")
    np.savez("data/out/scores_stack.npz", adsh=adsh_arr[tst], y=yt,
             **{k.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("-", "_"): v
                for k, v in scores.items()})


if __name__ == "__main__":
    main()
