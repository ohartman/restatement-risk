#!/usr/bin/env python3
"""Bake-off of the current best tabular learners against the hand-set booster.

What the 2025-26 literature says about a problem shaped like ours (36k rows,
28 numeric features, 3-5% positives, noisy labels):

  - Gradient boosting is still the reference; CatBoost and LightGBM with sane
    defaults usually match a tuned HistGradientBoosting, sometimes beat it.
  - Imbalance-aware ensembles (RUSBoost, EasyEnsemble, balanced forests) are
    what the accounting-fraud literature converged on (Bao et al. 2020, JAR),
    and TILBench (2026) finds them worth a moderate gain when labels are noisy.
  - Focal loss down-weights the easy negatives; class-balanced losses help
    recall more than AUC, so expect small movement here.
  - Tabular foundation models (TabPFN v2/2.5, TabICL) win ~83% of TabBench V2
    tasks in a forward pass, but the wins thin out above ~44k rows. TabPFN is
    run separately (edge_tabpfn.py) because it needs the GPU and its own
    subsampling.

Selection is on the 2017-18 validation slice inside the training period, as
in tune.py. Test (2019+) is reported for every model so the whole table is
visible, but the honest headline is whichever model validation picked.

  python edge.py [--raw data/out/features_raw.npz] [--labels data/out/features_tight.npz]
"""

import argparse
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(5)


def boot(y, s, n=400):
    v = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def rank(v):
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


def impute(Xtr, *others, plain_last=0):
    """Median fill plus a missing indicator per column, for learners without NaN support.
    plain_last: the last k columns are median-filled WITHOUT an indicator (used for the
    market block, whose missingness means "delisted by the download date")."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    out = []
    for X in (Xtr,) + others:
        filled = np.where(np.isnan(X), med, X)
        miss = np.isnan(X[:, : X.shape[1] - plain_last] if plain_last else X).astype(float)
        out.append(np.hstack([filled, miss]))
    return out


def focal_objective(gamma=2.0, alpha=0.75):
    """Focal loss for LightGBM, differentiated numerically; stable enough for a bake-off."""
    def obj(y, pred):
        eps = 1e-4

        def loss(z):
            q = 1 / (1 + np.exp(-z))
            pt = np.where(y == 1, q, 1 - q)
            at = np.where(y == 1, alpha, 1 - alpha)
            return -at * (1 - pt) ** gamma * np.log(np.clip(pt, 1e-9, 1))
        g = (loss(pred + eps) - loss(pred - eps)) / (2 * eps)
        h = (loss(pred + eps) - 2 * loss(pred) + loss(pred - eps)) / (eps ** 2)
        return g, np.maximum(h, 1e-6)
    return obj


def models(Xf, yf, Xv, yv, Xt):
    """Yield (name, val scores, test scores from fit-only, test scores refit on fit+val)."""
    Xfi, Xvi, Xti = impute(Xf, Xv, Xt)

    def two(fit_fn, needs_impute=False):
        a, b, c = (Xfi, Xvi, Xti) if needs_impute else (Xf, Xv, Xt)
        m = fit_fn(a, yf)
        sv, st = m.predict_proba(b)[:, 1], m.predict_proba(c)[:, 1]
        m2 = fit_fn(np.vstack([a, b]), np.concatenate([yf, yv]))
        return sv, st, m2.predict_proba(c)[:, 1]

    # 1. the incumbent
    hand = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=1.0, min_samples_leaf=40, random_state=0)
    yield "HGB hand-set (incumbent)", *two(
        lambda X, y: HistGradientBoostingClassifier(**hand).fit(X, y))

    # 2. LightGBM, plain, reweighted, and with focal loss
    try:
        import lightgbm as lgb
        base = dict(n_estimators=800, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                    verbose=-1, random_state=0)
        yield "LightGBM", *two(lambda X, y: lgb.LGBMClassifier(**base).fit(X, y))
        yield "LightGBM scale_pos_weight 5", *two(
            lambda X, y: lgb.LGBMClassifier(**base, scale_pos_weight=5.0).fit(X, y))

        class Focal(lgb.LGBMClassifier):
            def predict_proba(self, X):
                z = self.booster_.predict(X, raw_score=True)
                p = 1 / (1 + np.exp(-z))
                return np.c_[1 - p, p]
        yield "LightGBM focal loss", *two(
            lambda X, y: Focal(**base, objective=focal_objective()).fit(X, y))
    except ImportError:
        print("  (lightgbm not installed)")

    # 3. CatBoost, plain and class-balanced
    try:
        from catboost import CatBoostClassifier
        cb = dict(iterations=1200, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                  random_seed=0, verbose=0, allow_writing_files=False, thread_count=-1)
        yield "CatBoost", *two(lambda X, y: CatBoostClassifier(**cb).fit(X, y))
        yield "CatBoost balanced", *two(
            lambda X, y: CatBoostClassifier(**cb, auto_class_weights="Balanced").fit(X, y))
    except ImportError:
        print("  (catboost not installed)")

    # 4. imbalance-aware ensembles from the fraud literature
    try:
        from imblearn.ensemble import (BalancedRandomForestClassifier, EasyEnsembleClassifier,
                                       RUSBoostClassifier)
        yield "RUSBoost (Bao et al. style)", *two(
            lambda X, y: RUSBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=5),
                n_estimators=500, learning_rate=0.1, sampling_strategy=0.5,
                random_state=0).fit(X, y), needs_impute=True)
        yield "EasyEnsemble", *two(
            lambda X, y: EasyEnsembleClassifier(n_estimators=40, random_state=0,
                                                n_jobs=-1).fit(X, y), needs_impute=True)
        yield "Balanced random forest", *two(
            lambda X, y: BalancedRandomForestClassifier(
                n_estimators=600, min_samples_leaf=5, sampling_strategy="all",
                replacement=True, bootstrap=False, random_state=0, n_jobs=-1).fit(X, y),
            needs_impute=True)
    except ImportError:
        print("  (imbalanced-learn not installed)")


def load(args):
    z = np.load(args.features, allow_pickle=True)
    X, y, filed, adsh = z["X"], z["y"], z["filed"].astype("datetime64[D]"), z["adsh"]
    if args.labels:
        zl = np.load(args.labels, allow_pickle=True)
        assert (zl["adsh"] == adsh).all()
        y = zl["y"]
        print(f"labels: {args.labels}")
    if args.raw:
        zr = np.load(args.raw, allow_pickle=True)
        pos = {a: i for i, a in enumerate(zr["adsh"].tolist())}
        XR = zr["X"]
        R = np.full((len(adsh), XR.shape[1]), np.nan)
        hit = 0
        for i, a in enumerate(adsh.tolist()):
            j = pos.get(a)
            if j is not None:
                R[i] = XR[j]
                hit += 1
        X = np.hstack([X, R])
        print(f"raw items: {XR.shape[1]} columns joined on {hit:,}/{len(adsh):,} filings")
    return X, y, filed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/out/features.npz")
    ap.add_argument("--labels", help="npz whose y replaces the loose label (same adsh order)")
    ap.add_argument("--raw", help="npz of raw-item features to append, joined on adsh")
    ap.add_argument("--val-start", default="2017-01-01")
    ap.add_argument("--test-start", default="2019-01-01")
    args = ap.parse_args()

    X, y, filed = load(args)
    vs, ts = np.datetime64(args.val_start), np.datetime64(args.test_start)
    fm, vm, tm = filed < vs, (filed >= vs) & (filed < ts), filed >= ts
    print(f"fit {fm.sum():,} ({y[fm].sum()})  val {vm.sum():,} ({y[vm].sum()})  "
          f"test {tm.sum():,} ({y[tm].sum()})   features {X.shape[1]}\n")

    yt = y[tm]
    rows = []
    print(f"{'model':<30}{'val AUC':>9}{'test AUC':>10}{'refit test':>12}{'95% CI':>17}{'p@100':>8}")
    for name, sv, st, st2 in models(X[fm], y[fm], X[vm], y[vm], X[tm]):
        av, at, at2 = roc_auc_score(y[vm], sv), roc_auc_score(yt, st), roc_auc_score(yt, st2)
        lo, hi = boot(yt, st2)
        p100 = yt[np.argsort(-st2)[:100]].mean()
        print(f"{name:<30}{av:>9.3f}{at:>10.3f}{at2:>12.3f}   [{lo:.3f}-{hi:.3f}]{p100:>8.1%}",
              flush=True)
        rows.append((name, av, sv, st2))

    # Blends: rank-average the top few by validation, and all of them.
    rows.sort(key=lambda r: -r[1])
    print()
    for k in (2, 3, len(rows)):
        sv = np.mean([rank(r[2]) for r in rows[:k]], axis=0)
        st = np.mean([rank(r[3]) for r in rows[:k]], axis=0)
        lo, hi = boot(yt, st)
        p100 = yt[np.argsort(-st)[:100]].mean()
        label = f"rank blend of top {k} by val" if k < len(rows) else "rank blend of all"
        print(f"{label:<30}{roc_auc_score(y[vm], sv):>9.3f}{'':>10}"
              f"{roc_auc_score(yt, st):>12.3f}   [{lo:.3f}-{hi:.3f}]{p100:>8.1%}")
    print(f"\nvalidation picked: {rows[0][0]}")


if __name__ == "__main__":
    main()
