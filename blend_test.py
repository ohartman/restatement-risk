#!/usr/bin/env python3
"""Forest + boosting blend, retuned to the tuned forest on the audited stack.

LightGBM variants share the forest's 1:10 negative subsample per round; each is selected on
the validation years, then the best is rank-blended with the forest at 0.3 / 0.5 weight and
paired against the forest alone on the test set.

  BRF_STRATEGY=0.1 BRF_MAX_FEATURES=15 SCRUTINY_NPZ=... FLAGS2_NPZ=... python blend_test.py
"""
import os

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from edge import impute
from events_test import brf, joined, market_block, scrutiny_block
from final import TEST_START, ci, paired

VAL_YEARS = (2017, 2018)


def rank(v):
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


def fit_forest(X, y, trn, tst, plain, seeds=5):
    Xtr, Xte = impute(X[trn], X[tst], plain_last=plain)
    return np.mean([brf(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)


def fit_lgb(X, y, trn, tst, seeds, leaves, lr, ff, neg, rounds):
    out = []
    for s in range(seeds):
        m = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=lr, num_leaves=leaves, min_child_samples=20, feature_fraction=ff,
                               pos_bagging_fraction=1.0, neg_bagging_fraction=neg, bagging_freq=1, reg_lambda=1.0,
                               random_state=s, verbose=-1, n_jobs=-1)
        m.fit(X[trn], y[trn]); out.append(m.predict_proba(X[tst])[:, 1])
    return np.mean(out, axis=0)


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    R, _ = joined(adsh, "data/out/features_raw.npz"); R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    E, _ = joined(adsh, "data/out/features_events.npz"); I, _ = joined(adsh, "data/out/features_insider.npz")
    M = market_block(adsh); L, _ = joined(adsh, "data/out/features_letters.npz")
    S = scrutiny_block(adsh, os.environ["SCRUTINY_NPZ"]); F, _ = joined(adsh, os.environ["FLAGS2_NPZ"])
    X = np.hstack([fin, E, I, L, S, F, M]); plain = M.shape[1]
    trn, tst = filed < TEST_START, filed >= TEST_START
    yt = y[tst]
    print(f"stack {X.shape[1]} columns\n", flush=True)
    forest = fit_forest(X, y, trn, tst, plain)
    lo, hi = ci(yt, forest)
    print(f"{'model':<58}{'val 2017':>9}{'val 2018':>9}{'test':>8}{'95% CI':>16}{'   vs forest (paired)':>24}")
    print(f"{'tuned forest (the headline)':<58}{'':>9}{'':>9}{roc_auc_score(yt, forest):>8.3f}  [{lo:.3f}-{hi:.3f}]", flush=True)
    grid = [(15, 0.03, 0.5, 0.1, 600), (7, 0.03, 0.5, 0.1, 800), (31, 0.02, 0.5, 0.1, 600), (15, 0.03, 0.3, 0.1, 600),
            (15, 0.03, 0.5, 0.25, 600), (7, 0.05, 0.3, 0.1, 400)]
    best = None
    for leaves, lr, ff, neg, rounds in grid:
        vals = [roc_auc_score(y[years == v], fit_lgb(X, y, years < v, years == v, 3, leaves, lr, ff, neg, rounds)) for v in VAL_YEARS]
        name = f"LightGBM leaves {leaves}, lr {lr}, feature_fraction {ff}, neg {neg}, {rounds} rounds"
        mean = sum(vals) / 2
        print(f"{name:<58}{vals[0]:>9.3f}{vals[1]:>9.3f}", flush=True)
        if best is None or mean > best[0]:
            best = (mean, (leaves, lr, ff, neg, rounds), name)
    _, params, name = best
    g = fit_lgb(X, y, trn, tst, 5, *params)
    lo, hi = ci(yt, g); m, (dlo, dhi), p = paired(yt, forest, g)
    print(f"{'best by validation: ' + name:<58}{'':>18}{roc_auc_score(yt, g):>8.3f}  [{lo:.3f}-{hi:.3f}]   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}] P {p:.3f}")
    for w in (0.3, 0.5):
        b = (1 - w) * rank(forest) + w * rank(g)
        lo, hi = ci(yt, b); m, (dlo, dhi), p = paired(yt, forest, b)
        print(f"{'rank blend, boosting weight ' + str(w):<58}{'':>18}{roc_auc_score(yt, b):>8.3f}  [{lo:.3f}-{hi:.3f}]   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}] P {p:.3f}", flush=True)
    print("BLEND DONE")


if __name__ == "__main__":
    main()
