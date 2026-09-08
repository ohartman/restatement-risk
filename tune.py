#!/usr/bin/env python3
"""Tune the booster honestly, then touch the test set once.

The gradient booster's settings were chosen by hand and never moved. This
searches them properly -- but against a validation slice carved out of the
*training* period by time, never against the test period. Every earlier
comparison was reported on the test set, which is fine for reporting and
disqualifying for tuning: a setting picked because it scored well on the test
set has already leaked the answer.

  fit    2014-2016 filings
  select 2017-2018 filings   (validation, inside the training period)
  report 2019-2023 filings   (test, touched once, at the end)

Four things are tried on top of the search:
  - early stopping on the validation slice
  - recency weighting, since the base rate drifts from 2% to 6% over the period
  - a bagged ensemble of seeds
  - a blend with a regularised logistic model, which errs differently

Features are cached to data/out/features.npz so the 40-quarter build runs once.

  python tune.py --trials 40
"""

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import ALL_VARS
from train import build

CACHE = Path("data/out/features.npz")
RNG = np.random.default_rng(11)


def load(window_days):
    """Feature matrix with NaN for missing, cached after the first build."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return z["X"], z["y"], z["filed"].astype("datetime64[D]"), z["f"]
    rows = build(window_days)
    n = len(ALL_VARS)
    # train.build stores [values..., missing-mask...]; turn the mask into NaN so
    # the booster's native missing-value handling sees it.
    vals = np.array([r["x"][:n] for r in rows], dtype=float)
    mask = np.array([r["x"][n:] for r in rows], dtype=float)
    X = np.where(mask > 0.5, np.nan, vals)
    f = np.array([r["f"] for r in rows], dtype=float)
    X = np.hstack([X, np.log1p(np.clip(f, 0, 500)).reshape(-1, 1)])
    y = np.array([r["y"] for r in rows], dtype=int)
    filed = np.array([str(r["filed"]) for r in rows])
    adsh = np.array([r["adsh"] for r in rows])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=X, y=y, filed=filed, f=f, adsh=adsh)
    return X, y, filed.astype("datetime64[D]"), f


def sample_params():
    return {
        "learning_rate": float(10 ** RNG.uniform(-2.0, -0.7)),
        "max_iter": int(RNG.choice([200, 400, 800, 1200])),
        "max_leaf_nodes": int(RNG.choice([7, 15, 31, 63])),
        "min_samples_leaf": int(RNG.choice([20, 40, 80, 160, 320])),
        "l2_regularization": float(10 ** RNG.uniform(-2, 1.5)),
        "max_depth": int(RNG.choice([3, 4, 6, 8, 12])) if RNG.random() < 0.6 else None,
        "max_features": float(RNG.choice([0.4, 0.6, 0.8, 1.0])),
    }


def fit(params, X, y, w=None, Xval=None, yval=None, seed=0):
    kw = dict(params)
    if Xval is not None:
        kw.update(early_stopping=True, validation_fraction=None,
                  n_iter_no_change=40, scoring="roc_auc")
    m = HistGradientBoostingClassifier(random_state=seed, **kw)
    if Xval is not None:
        # sklearn cannot take an external validation set, so emulate early
        # stopping by training on train and picking the iteration count on val.
        kw.pop("early_stopping"); kw.pop("validation_fraction")
        kw.pop("n_iter_no_change"); kw.pop("scoring")
        m = HistGradientBoostingClassifier(random_state=seed, **kw).fit(X, y, sample_weight=w)
        best_it, best_auc = kw["max_iter"], -1
        for it, proba in enumerate(m.staged_predict_proba(Xval), 1):
            a = roc_auc_score(yval, proba[:, 1])
            if a > best_auc:
                best_auc, best_it = a, it
        kw["max_iter"] = max(50, best_it)
        return HistGradientBoostingClassifier(random_state=seed, **kw).fit(
            X, y, sample_weight=w), best_auc
    return m.fit(X, y, sample_weight=w), None


def recency_weights(filed, lo=0.5):
    """Linearly up-weight recent filings; the earliest get `lo`, the latest 1.0."""
    d = filed.astype("datetime64[D]").astype(np.int64)
    t = (d - d.min()) / max(1, d.max() - d.min())
    return lo + (1 - lo) * t


def boot_auc(y, s, n=400):
    vals = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--val-start", default="2017-01-01")
    ap.add_argument("--test-start", default="2019-01-01")
    ap.add_argument("--window-days", type=int, default=1095)
    args = ap.parse_args()

    X, y, filed, f = load(args.window_days)
    vs, ts = np.datetime64(args.val_start), np.datetime64(args.test_start)
    fit_m, val_m, test_m = filed < vs, (filed >= vs) & (filed < ts), filed >= ts
    print(f"fit {fit_m.sum():,} ({y[fit_m].sum()})   val {val_m.sum():,} "
          f"({y[val_m].sum()})   test {test_m.sum():,} ({y[test_m].sum()})\n")

    Xf, yf = X[fit_m], y[fit_m]
    Xv, yv = X[val_m], y[val_m]
    Xt, yt = X[test_m], y[test_m]

    base_params = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                       l2_regularization=1.0, min_samples_leaf=40)
    m0 = HistGradientBoostingClassifier(random_state=0, **base_params).fit(Xf, yf)
    print(f"{'hand-set params':<34} val AUC {roc_auc_score(yv, m0.predict_proba(Xv)[:,1]):.3f}")

    # --- random search, scored on validation only ---
    best = (roc_auc_score(yv, m0.predict_proba(Xv)[:, 1]), base_params, False)
    for t in range(args.trials):
        p = sample_params()
        for use_w in (False, True):
            w = recency_weights(filed[fit_m]) if use_w else None
            m, _ = fit(p, Xf, yf, w)
            a = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
            if a > best[0]:
                best = (a, p, use_w)
        if (t + 1) % 10 == 0:
            print(f"  trial {t+1:>3}/{args.trials}   best val AUC {best[0]:.3f}", flush=True)
    a_val, p_best, w_best = best
    print(f"\n{'best of search':<34} val AUC {a_val:.3f}   recency weights: {w_best}")
    print(f"  {json.dumps(p_best)}")

    # --- early-stopped iteration count on the validation slice ---
    w = recency_weights(filed[fit_m]) if w_best else None
    m_es, a_es = fit(p_best, Xf, yf, w, Xv, yv)
    print(f"{'  + early stopping':<34} val AUC {a_es:.3f}   iters {m_es.max_iter}")

    # --- now refit on fit+val with the chosen settings, and touch test once ---
    trn = fit_m | val_m
    wt = recency_weights(filed[trn]) if w_best else None
    p_final = dict(p_best, max_iter=m_es.max_iter)

    def report(name, s):
        a = roc_auc_score(yt, s)
        lo, hi = boot_auc(yt, s)
        p100 = yt[np.argsort(-s)[:100]].mean()
        print(f"{name:<34} TEST AUC {a:.3f} [{lo:.3f}-{hi:.3f}]   p@100 {p100:.1%}")
        return a

    print("\n--- test set, touched once ---")
    m_hand = HistGradientBoostingClassifier(random_state=0, **base_params).fit(X[trn], y[trn])
    report("hand-set params (prior result)", m_hand.predict_proba(Xt)[:, 1])

    m_tuned = HistGradientBoostingClassifier(random_state=0, **p_final).fit(
        X[trn], y[trn], sample_weight=wt)
    s_tuned = m_tuned.predict_proba(Xt)[:, 1]
    report("tuned", s_tuned)

    seeds = []
    for sd in range(7):
        m = HistGradientBoostingClassifier(random_state=sd, **p_final).fit(
            X[trn], y[trn], sample_weight=wt)
        seeds.append(m.predict_proba(Xt)[:, 1])
    s_bag = np.mean(seeds, axis=0)
    report("tuned, bagged over 7 seeds", s_bag)

    # A regularised linear model errs differently from trees; blend by rank.
    med = np.nanmedian(X[trn], axis=0)
    Xtr_l = np.where(np.isnan(X[trn]), med, X[trn])
    Xt_l = np.where(np.isnan(Xt), med, Xt)
    sc = StandardScaler().fit(Xtr_l)
    lr = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(
        sc.transform(Xtr_l), y[trn])
    s_lr = lr.predict_proba(sc.transform(Xt_l))[:, 1]
    report("logistic alone", s_lr)

    def rk(v):
        return np.argsort(np.argsort(v)) / (len(v) - 1)
    for wgt in (0.7, 0.85):
        report(f"blend {wgt:.2f} bag + {1-wgt:.2f} logistic",
               wgt * rk(s_bag) + (1 - wgt) * rk(s_lr))

    json.dump({"best_params": p_final, "recency_weights": bool(w_best),
               "val_auc": float(a_es)},
              open("data/out/tuned.json", "w"), indent=2)


if __name__ == "__main__":
    main()
