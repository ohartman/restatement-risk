#!/usr/bin/env python3
"""Learner experiments on the clean full stack, each selected on the 2017/2018 validation
years and paired against the 1:10 balanced forest on the test set.

  A. forest hyperparameters at the new ratio (features per split, leaf size, trees)
  B. a rank-average of forests at 1:2, 1:4 and 1:10 (diversity across the ratio)
  C. LightGBM with the same 1:10 negative subsample per boosting round, alone and blended
  D. label-quality weights: positives whose own fiscal year was condemned (the tight
     label) weigh 1, the rest of the loose positives weigh less
  E. train on the tight label, test on the loose one
  F. fifteen seeds instead of five (is the average at its limit?)

  BRF_STRATEGY=0.1 SCRUTINY_NPZ=... python algo_wins.py
"""
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from imblearn.ensemble import BalancedRandomForestClassifier

from edge import impute
from events_test import joined
from final import TEST_START, ci, paired

VAL_YEARS = (2017, 2018)


def forest(seed, strategy=0.1, n=600, leaf=5, max_features="sqrt"):
    return BalancedRandomForestClassifier(n_estimators=n, min_samples_leaf=leaf, sampling_strategy=strategy, max_features=max_features,
                                          replacement=True, bootstrap=False, random_state=seed, n_jobs=-1)


def rank(v):
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


def fit_forest(make, X, y, trn, tst, seeds, w=None):
    Xtr, Xte = impute(X[trn], X[tst])
    out = []
    for s in range(seeds):
        m = make(s)
        m.fit(Xtr, y[trn], sample_weight=None if w is None else w[trn])
        out.append(m.predict_proba(Xte)[:, 1])
    return np.mean(out, axis=0)


def fit_lgb(X, y, trn, tst, seeds, neg_frac=0.1):
    import lightgbm as lgb
    out = []
    for s in range(seeds):
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=20, feature_fraction=0.5,
                               pos_bagging_fraction=1.0, neg_bagging_fraction=neg_frac, bagging_freq=1, reg_lambda=1.0,
                               random_state=s, verbose=-1, n_jobs=-1)
        m.fit(X[trn], y[trn]); out.append(m.predict_proba(X[tst])[:, 1])
    return np.mean(out, axis=0)


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    y_tight = np.load("data/out/features_tight.npz", allow_pickle=True)["y"]
    R, _ = joined(adsh, "data/out/features_raw.npz"); R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    E, _ = joined(adsh, "data/out/features_events.npz"); I, _ = joined(adsh, "data/out/features_insider.npz")
    M, mn = joined(adsh, "data/out/features_market.npz"); M = M[:, [i for i, n in enumerate(mn) if n != "price_cov"]]
    L, _ = joined(adsh, "data/out/features_letters.npz"); S, _ = joined(adsh, os.environ.get("SCRUTINY_NPZ", "data/out/features_scrutiny.npz"))
    F, _ = joined(adsh, os.environ.get("FLAGS2_NPZ", "data/out/features_flags2_ec.npz"))
    X = np.hstack([fin, E, I, M, L, S, F])
    trn, tst = filed < TEST_START, filed >= TEST_START
    yt = y[tst]
    print(f"stack {X.shape[1]} columns (the 330 headline stack); train {int(trn.sum()):,} ({int(y[trn].sum())})  test {int(tst.sum()):,} ({int(yt.sum())})\n", flush=True)

    def evaluate(name, run):
        """run(trn_mask, tst_mask, seeds) -> scores on tst_mask rows."""
        try:
            vals = [roc_auc_score(y[years == v], run(years < v, years == v, 3)) for v in VAL_YEARS]
            st = run(trn, tst, 5)
        except Exception as exc:
            print(f"{name:<58} failed: {type(exc).__name__}: {str(exc)[:70]}", flush=True); return None
        lo, hi = ci(yt, st); order = np.argsort(-st)
        line = f"{name:<58}{vals[0]:>9.3f}{vals[1]:>9.3f}{roc_auc_score(yt, st):>9.3f}  [{lo:.3f}-{hi:.3f}]{yt[order[:100]].mean():>6.0%}"
        if evaluate.base is not None:
            m, (dlo, dhi), p = paired(yt, evaluate.base, st); line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>8.3f}"
        print(line, flush=True); return st
    evaluate.base = None
    print(f"{'variant':<58}{'val 2017':>9}{'val 2018':>9}{'test':>9}{'95% CI':>16}{'p@100':>6}{'   vs 1:10 forest (paired)':>26}{'P(<=0)':>8}")

    base = evaluate("1:10 forest, 600 trees, leaf 5, sqrt features (current)", lambda a, b, k: fit_forest(forest, X, y, a, b, k))
    evaluate.base = base
    # A. hyperparameters at the ratio
    evaluate("A. features per split 0.2 of columns", lambda a, b, k: fit_forest(lambda s: forest(s, max_features=0.2), X, y, a, b, k))
    evaluate("A. features per split 0.35 of columns", lambda a, b, k: fit_forest(lambda s: forest(s, max_features=0.35), X, y, a, b, k))
    evaluate("A. features per split log2", lambda a, b, k: fit_forest(lambda s: forest(s, max_features="log2"), X, y, a, b, k))
    evaluate("A. leaf 2", lambda a, b, k: fit_forest(lambda s: forest(s, leaf=2), X, y, a, b, k))
    evaluate("A. leaf 12", lambda a, b, k: fit_forest(lambda s: forest(s, leaf=12), X, y, a, b, k))
    evaluate("A. leaf 25", lambda a, b, k: fit_forest(lambda s: forest(s, leaf=25), X, y, a, b, k))
    evaluate("A. 1,500 trees", lambda a, b, k: fit_forest(lambda s: forest(s, n=1500), X, y, a, b, k))
    # B. ensemble across ratios
    def ratio_ens(a, b, k):
        return np.mean([rank(fit_forest(lambda s: forest(s, strategy=r), X, y, a, b, k)) for r in (0.5, 0.25, 0.1)], axis=0)
    evaluate("B. rank-average of 1:2, 1:4, 1:10 forests", ratio_ens)
    # C. boosting with the same negative subsample, and the blend
    lgb_scores = {}
    def lgb_run(a, b, k):
        s = fit_lgb(X, y, a, b, k); lgb_scores[(a.sum(), b.sum())] = s; return s
    evaluate("C. LightGBM, 1:10 negatives per round, 600 rounds", lgb_run)
    def blend(a, b, k):
        f = fit_forest(forest, X, y, a, b, k); g = lgb_scores.get((a.sum(), b.sum()))
        if g is None: g = fit_lgb(X, y, a, b, k)
        return 0.5 * rank(f) + 0.5 * rank(g)
    evaluate("C. rank blend: forest + LightGBM", blend)
    # D. label-quality weights
    for w_loose in (0.5, 0.25):
        w = np.where(y_tight == 1, 1.0, np.where(y == 1, w_loose, 1.0))
        evaluate(f"D. loose-only positives weighted {w_loose}", lambda a, b, k, w=w: fit_forest(forest, X, y, a, b, k, w=w))
    # E. train on the tight label
    evaluate("E. trained on the tight label, tested on the loose", lambda a, b, k: fit_forest(forest, X, y_tight, a, b, k))
    # F. more seeds
    evaluate("F. fifteen seeds", lambda a, b, k: fit_forest(forest, X, y, a, b, 15 if k == 5 else k))
    print("ALGO WINS DONE")


if __name__ == "__main__":
    main()
