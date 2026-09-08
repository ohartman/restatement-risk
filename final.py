#!/usr/bin/env python3
"""Is the new model actually better than the old one? The honest checks.

edge.py found that raw statement items plus a balanced random forest score
0.715 where the hand-set booster on ratios scored 0.691. Overlapping
confidence intervals on two separate AUCs do not settle whether one model
beats another, because both are scored on the same filings and their errors
are correlated. The right test is a paired bootstrap of the difference: resample
the test filings, score both models on the same resample, and look at the
distribution of (new - old).

Also reported: per-year AUC, so the gain is not one good year; the same numbers
with 2021 dropped, because that year's SPAC wave is unlike the others; and a
seed-bagged version of the forest, since a single random forest is itself a
random variable.

  python final.py
"""

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from edge import impute, rank

RNG = np.random.default_rng(21)
TEST_START = np.datetime64("2019-01-01")


def paired(y, a, b, n=2000):
    d = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    d = np.array(d)
    return d.mean(), np.percentile(d, [2.5, 97.5]), (d <= 0).mean()


def paired_cluster(y, a, b, groups, n=2000):
    """Paired bootstrap resampling firms (clusters), not filings: a firm's five 10-Ks move together."""
    groups = np.asarray(groups); uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    d = []
    for _ in range(n):
        pick = RNG.integers(0, len(uniq), len(uniq))
        i = np.concatenate([members[g] for g in pick])
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    d = np.array(d)
    return d.mean(), np.percentile(d, [2.5, 97.5]), (d <= 0).mean()


def ci_cluster(y, s, groups, n=1000):
    groups = np.asarray(groups); uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    v = []
    for _ in range(n):
        pick = RNG.integers(0, len(uniq), len(uniq))
        i = np.concatenate([members[g] for g in pick])
        if 0 < y[i].sum() < len(i):
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def ci(y, s, n=1000):
    v = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    X28, y, filed, adsh = z["X"], z["y"], z["filed"].astype("datetime64[D]"), z["adsh"]
    zr = np.load("data/out/features_raw.npz", allow_pickle=True)
    pos = {a: i for i, a in enumerate(zr["adsh"].tolist())}
    XR = zr["X"]
    R = np.array([XR[pos[a]] for a in adsh.tolist()])
    X79 = np.hstack([X28, R])
    trn, tst = filed < TEST_START, filed >= TEST_START
    yt = y[tst]
    years = filed[tst].astype("datetime64[Y]").astype(int) + 1970
    print(f"train {trn.sum():,} ({y[trn].sum()})   test {tst.sum():,} ({yt.sum()})\n")

    hand = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=1.0, min_samples_leaf=40, random_state=0)
    scores = {}
    scores["old: HGB on 28 ratios"] = HistGradientBoostingClassifier(**hand).fit(
        X28[trn], y[trn]).predict_proba(X28[tst])[:, 1]
    scores["HGB on 79 (ratios + raw items)"] = HistGradientBoostingClassifier(**hand).fit(
        X79[trn], y[trn]).predict_proba(X79[tst])[:, 1]

    Xtr_i, Xte_i = impute(X79[trn], X79[tst])
    brf = lambda seed: BalancedRandomForestClassifier(
        n_estimators=600, min_samples_leaf=5, sampling_strategy="all", replacement=True,
        bootstrap=False, random_state=seed, n_jobs=-1)
    scores["balanced RF on 79, one seed"] = brf(0).fit(Xtr_i, y[trn]).predict_proba(Xte_i)[:, 1]
    bag = [brf(s).fit(Xtr_i, y[trn]).predict_proba(Xte_i)[:, 1] for s in range(5)]
    scores["balanced RF on 79, 5 seeds"] = np.mean(bag, axis=0)
    ee = EasyEnsembleClassifier(n_estimators=40, random_state=0, n_jobs=-1).fit(
        Xtr_i, y[trn]).predict_proba(Xte_i)[:, 1]
    scores["new: blend RF(5 seeds) + EasyEnsemble"] = 0.5 * rank(scores["balanced RF on 79, 5 seeds"]) + 0.5 * rank(ee)

    old = scores["old: HGB on 28 ratios"]
    print(f"{'model':<42}{'AUC':>7}{'95% CI':>17}{'p@100':>8}{'  vs old (paired)':>26}{'P(<=0)':>8}")
    for name, s in scores.items():
        lo, hi = ci(yt, s)
        p100 = yt[np.argsort(-s)[:100]].mean()
        line = f"{name:<42}{roc_auc_score(yt, s):>7.3f}   [{lo:.3f}-{hi:.3f}]{p100:>8.1%}"
        if s is not old:
            m, (dlo, dhi), p = paired(yt, old, s)
            line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>8.3f}"
        print(line, flush=True)

    new = scores["new: blend RF(5 seeds) + EasyEnsemble"]
    print(f"\n{'per year':<10}{'n':>7}{'pos':>6}{'old AUC':>9}{'new AUC':>9}{'gain':>7}")
    for yr in sorted(set(years)):
        m = years == yr
        if yt[m].sum() < 10:
            continue
        a, b = roc_auc_score(yt[m], old[m]), roc_auc_score(yt[m], new[m])
        print(f"{yr:<10}{m.sum():>7,}{int(yt[m].sum()):>6}{a:>9.3f}{b:>9.3f}{b-a:>+7.3f}")
    m = years != 2021
    print(f"{'drop 2021':<10}{m.sum():>7,}{int(yt[m].sum()):>6}"
          f"{roc_auc_score(yt[m], old[m]):>9.3f}{roc_auc_score(yt[m], new[m]):>9.3f}"
          f"{roc_auc_score(yt[m], new[m]) - roc_auc_score(yt[m], old[m]):>+7.3f}")

    np.savez("data/out/scores_final.npz", adsh=adsh[tst], y=yt, filed=z["filed"][tst],
             **{k.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
                .replace("+", "plus").replace(",", ""): v for k, v in scores.items()})
    print("\nwrote data/out/scores_final.npz")


if __name__ == "__main__":
    main()
