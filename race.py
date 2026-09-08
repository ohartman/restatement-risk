#!/usr/bin/env python3
"""Bao et al. (2020) on their own track: same data, same splits, same metrics.

Bao, Ke, Li, Yu & Zhang published their labelled panel -- Compustat items for
every US firm-year 1990-2014 with the SEC enforcement (AAER) label -- and the
MATLAB that produced their headline. Their protocol, read off run_RUSBoost.m:

  for each test year t in 2003..2008:
      train on fiscal years 1991 .. t-2         (two-year gap: frauds take time to surface)
      test on fiscal year t
      RUSBoost, 300 trees, learn rate 0.1, min leaf 5, undersample to 1:1
      report AUC and NDCG@1% for year t; the paper averages over the six years

Two versions of their serial-fraud step are run. Their code zeroes the training
label of any firm-year whose AAER case also appears among the *test* year's
frauds -- which uses test-set information during training, the point of
Walker's (2021) critique and of the authors' subsequent correction. "bao"
reproduces that; "clean" skips it and touches nothing from the test year.

Our learner is the balanced random forest that won on the restatement problem,
with the same settings. Nothing is tuned here; this is a head-to-head on their
inputs, so any difference is the learner.

  python race.py
"""

import time

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

RAW = ["act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt", "dp", "ib",
       "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk", "re", "rect", "sale", "sstk",
       "txp", "txt", "xint", "prcc_f"]
RATIOS = ["dch_wc", "ch_rsst", "dch_rec", "dch_inv", "soft_assets", "ch_cs", "ch_cm", "ch_roa",
          "issue", "bm", "dpi", "reoa", "EBIT", "ch_fcf"]
TEST_YEARS = range(2003, 2009)


def ndcg_at(y, s, frac=0.01):
    """Their evaluate.m, line for line: binary relevance, top round(n*frac)."""
    k = int(round(len(y) * frac))
    idx = np.argsort(-s, kind="stable")[:k]
    dcg = sum(1 / np.log2(2 + i) for i, j in enumerate(idx) if y[j] == 1)
    ideal = sum(1 / np.log2(2 + i) for i in range(min(k, int(y.sum()))))
    return dcg / ideal if ideal else 0.0


def precision_at(y, s, frac=0.01):
    k = int(round(len(y) * frac))
    return y[np.argsort(-s, kind="stable")[:k]].mean()


def rusboost():
    return RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5),
                              n_estimators=300, learning_rate=0.1, sampling_strategy=1.0,
                              random_state=0)


def brf(seed=0):
    from events_test import strategy           # BRF_STRATEGY: positives:negatives per tree, default 1:1
    return BalancedRandomForestClassifier(n_estimators=600, min_samples_leaf=5,
                                          sampling_strategy=strategy(), replacement=True,
                                          bootstrap=False, random_state=seed, n_jobs=-1)


def hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                                          l2_regularization=1.0, min_samples_leaf=40, random_state=0)


MODELS = {
    "RUSBoost-28 (their model, their inputs)": (rusboost, RAW),
    "balanced RF-28 (our model, their inputs)": (brf, RAW),
    "balanced RF-42 (raw + their 14 ratios)": (brf, RAW + RATIOS),
    "HGB-28": (hgb, RAW),
}


def main():
    df = pd.read_csv("data/raw/bao/data_FraudDetection_JAR2020.csv")
    print(f"{len(df):,} firm-years {df.fyear.min()}-{df.fyear.max()}, "
          f"{int(df.misstate.sum()):,} fraud years ({df.misstate.mean():.2%})\n")

    for handling in ("bao", "clean"):
        print(f"================ serial-fraud handling: {handling} ================")
        per_year = {m: [] for m in MODELS}
        pooled = {m: [] for m in MODELS}
        pooled_y = []
        for t in TEST_YEARS:
            tr = df[(df.fyear >= 1991) & (df.fyear <= t - 2)].copy()
            te = df[df.fyear == t]
            y_tr = tr.misstate.values.astype(int).copy()
            if handling == "bao":
                test_cases = set(te.loc[te.misstate == 1, "p_aaer"].dropna())
                y_tr[tr.p_aaer.isin(test_cases).values] = 0
            y_te = te.misstate.values.astype(int)
            pooled_y.append(y_te)
            line = f"  {t}  train {len(tr):>6,} ({y_tr.sum():>3})  test {len(te):>5,} ({y_te.sum():>2})"
            for name, (make, cols) in MODELS.items():
                t0 = time.time()
                m = make().fit(tr[cols].values, y_tr)
                s = m.predict_proba(te[cols].values)[:, 1]
                a, nd, p = roc_auc_score(y_te, s), ndcg_at(y_te, s), precision_at(y_te, s)
                per_year[name].append((a, nd, p))
                pooled[name].append(s)
                line += f"  | {name.split(' (')[0]:<16} AUC {a:.3f} NDCG@1% {nd:.3f}"
            print(line, flush=True)
        print(f"\n  {'average over 2003-2008':<44}{'AUC':>7}{'NDCG@1%':>9}{'prec@1%':>9}")
        for name, rows in per_year.items():
            r = np.array(rows)
            print(f"  {name:<44}{r[:, 0].mean():>7.3f}{r[:, 1].mean():>9.3f}{r[:, 2].mean():>9.1%}")
        yp = np.concatenate(pooled_y)
        ref = np.concatenate(pooled["RUSBoost-28 (their model, their inputs)"])
        rng = np.random.default_rng(0)
        print(f"\n  pooled 2003-2008 ({len(yp):,} firm-years, {int(yp.sum())} fraud), "
              f"paired vs their RUSBoost:")
        for name, chunks in pooled.items():
            s_all = np.concatenate(chunks)
            d = []
            for _ in range(1000):
                i = rng.integers(0, len(yp), len(yp))
                if 0 < yp[i].sum() < len(i):
                    d.append(roc_auc_score(yp[i], s_all[i]) - roc_auc_score(yp[i], ref[i]))
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  {name:<44} pooled AUC {roc_auc_score(yp, s_all):.3f}   "
                  f"diff {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]   P(<=0) {np.mean(np.array(d) <= 0):.3f}")
        np.savez(f"data/out/race_{handling}.npz", y=yp, **{k.split(' (')[0].replace(' ', '_').replace('-', '_'): np.concatenate(v) for k, v in pooled.items()})
        print()


if __name__ == "__main__":
    main()
