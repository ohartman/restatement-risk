#!/usr/bin/env python3
"""Trying to beat Bao et al. on their own track -- honestly.

race.py showed that on their 28 raw items every competent learner lands at
0.71-0.72. To do better without cheating, the rules are:

  * Their data, their splits, their serial-fraud handling, their metrics.
  * Every choice is made on THEIR validation setup (train 1991-1999, validate
    on fiscal 2001, from tune_RUSBoost.m). The 2003-2008 test years are run
    once, with whatever validation picked.
  * No outside data. The only new inputs are each firm's own prior-year values
    of the same 28 items, and within-year percentile ranks of the same items.

Ideas on trial:
  lag      last year's value and the change, per item          (what won on our data)
  yrank    each item as a percentile within its fiscal year     (undoes 17 years of drift)
  forest   min_samples_leaf and max_features settings
  blend    rank-average of forest, booster and their RUSBoost   (different error patterns)

  python race2.py
"""

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from race import RAW, RATIOS, ndcg_at, precision_at, rusboost

TEST_YEARS = range(2003, 2009)


def rank(v):
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


def add_lags(df):
    """Prior-year value and change for each raw item, from the firm's own history only."""
    df = df.sort_values(["gvkey", "fyear"]).copy()
    prev = df.groupby("gvkey")[RAW].shift(1)
    consecutive = (df.fyear - df.groupby("gvkey")["fyear"].shift(1)) == 1
    prev = prev.where(consecutive)
    for c in RAW:
        df[f"{c}_lag"] = prev[c]
        df[f"{c}_chg"] = df[c] - prev[c]
    df["n_prior_years"] = df.groupby("gvkey").cumcount()
    return df


def add_yrank(df, cols):
    for c in cols:
        df[f"{c}_yr"] = df.groupby("fyear")[c].rank(pct=True)
    return df


LAG = [f"{c}_lag" for c in RAW] + [f"{c}_chg" for c in RAW] + ["n_prior_years"]
YR = [f"{c}_yr" for c in RAW]

FEATURE_SETS = {
    "28": RAW,
    "28+ratios": RAW + RATIOS,
    "28+lag": RAW + LAG,
    "28+lag+ratios": RAW + LAG + RATIOS,
    "yrank": YR,
    "28+yrank": RAW + YR,
    "28+lag+yrank": RAW + LAG + YR,
}


def brf(leaf=5, feats="sqrt", n=600, seed=0):
    return BalancedRandomForestClassifier(n_estimators=n, min_samples_leaf=leaf, max_features=feats,
                                          sampling_strategy="all", replacement=True, bootstrap=False,
                                          random_state=seed, n_jobs=-1)


def hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                                          l2_regularization=1.0, min_samples_leaf=40, random_state=0)


LEARNERS = {
    "RUSBoost": rusboost,
    "BRF leaf5": lambda: brf(5),
    "BRF leaf1": lambda: brf(1),
    "BRF leaf20": lambda: brf(20),
    "BRF leaf5 feat0.3": lambda: brf(5, 0.3),
    "BRF leaf5 feat0.5": lambda: brf(5, 0.5),
    "HGB": hgb,
}


def split(df, train_lo, train_hi, test_year):
    tr = df[(df.fyear >= train_lo) & (df.fyear <= train_hi)]
    te = df[df.fyear == test_year]
    y_tr = tr.misstate.values.astype(int).copy()
    cases = set(te.loc[te.misstate == 1, "p_aaer"].dropna())
    y_tr[tr.p_aaer.isin(cases).values] = 0            # their serial-fraud step, as in their code
    return tr, te, y_tr, te.misstate.values.astype(int)


def fill(tr, te, cols):
    """Median fill for learners without NaN support (the lag columns are NaN in a firm's first year)."""
    med = tr[cols].median()
    return tr[cols].fillna(med).values, te[cols].fillna(med).values


def main():
    df = pd.read_csv("data/raw/bao/data_FraudDetection_JAR2020.csv")
    df = add_lags(df)
    df = add_yrank(df, RAW)

    # ---------------- selection on their validation year ----------------
    tr, te, y_tr, y_va = split(df, 1991, 1999, 2001)
    print(f"validation: train 1991-1999 {len(tr):,} ({y_tr.sum()})  validate 2001 {len(te):,} ({y_va.sum()})\n")
    print(f"{'learner':<20}{'features':<16}{'val AUC':>9}{'NDCG@1%':>9}")
    val_scores = {}
    for fname, cols in FEATURE_SETS.items():
        Xtr, Xva = fill(tr, te, cols)
        for lname, make in LEARNERS.items():
            s = make().fit(Xtr, y_tr).predict_proba(Xva)[:, 1]
            val_scores[(lname, fname)] = s
            print(f"{lname:<20}{fname:<16}{roc_auc_score(y_va, s):>9.3f}{ndcg_at(y_va, s):>9.3f}", flush=True)

    # blends: rank-average the best forest, the booster and RUSBoost on each feature set
    print()
    for fname in FEATURE_SETS:
        parts = [val_scores[(l, fname)] for l in ("BRF leaf5", "HGB", "RUSBoost")]
        s = np.mean([rank(p) for p in parts], axis=0)
        val_scores[("blend BRF+HGB+RUS", fname)] = s
        print(f"{'blend BRF+HGB+RUS':<20}{fname:<16}{roc_auc_score(y_va, s):>9.3f}{ndcg_at(y_va, s):>9.3f}")

    ranked = sorted(val_scores, key=lambda k: -roc_auc_score(y_va, val_scores[k]))
    print("\nvalidation top 5:")
    for k in ranked[:5]:
        print(f"  {k[0]:<20}{k[1]:<16}{roc_auc_score(y_va, val_scores[k]):.3f}")
    chosen = ranked[0]
    print(f"\nchosen on validation: {chosen[0]} on {chosen[1]}")

    # ---------------- the test years, run once ----------------
    def fit_predict(lname, fname, tr, te, y_tr):
        Xtr, Xte = fill(tr, te, FEATURE_SETS[fname])
        if lname.startswith("blend"):
            parts = [LEARNERS[l]().fit(Xtr, y_tr).predict_proba(Xte)[:, 1]
                     for l in ("BRF leaf5", "HGB", "RUSBoost")]
            return np.mean([rank(p) for p in parts], axis=0)
        return LEARNERS[lname]().fit(Xtr, y_tr).predict_proba(Xte)[:, 1]

    print(f"\n{'':<6}{'their RUSBoost-28':>22}{'chosen':>22}")
    ref_all, new_all, y_all, rows = [], [], [], []
    for t in TEST_YEARS:
        tr, te, y_tr, y_te = split(df, 1991, t - 2, t)
        ref = fit_predict("RUSBoost", "28", tr, te, y_tr)
        new = fit_predict(chosen[0], chosen[1], tr, te, y_tr)
        ref_all.append(ref); new_all.append(new); y_all.append(y_te)
        rows.append((roc_auc_score(y_te, ref), ndcg_at(y_te, ref), roc_auc_score(y_te, new), ndcg_at(y_te, new)))
        print(f"  {t}  AUC {rows[-1][0]:.3f} NDCG {rows[-1][1]:.3f}     AUC {rows[-1][2]:.3f} NDCG {rows[-1][3]:.3f}", flush=True)
    r = np.array(rows)
    print(f"  avg   AUC {r[:, 0].mean():.3f} NDCG {r[:, 1].mean():.3f}     AUC {r[:, 2].mean():.3f} NDCG {r[:, 3].mean():.3f}")
    print(f"\n  published: 0.725 / 0.049 (2020)   corrected: 0.723 / 0.024 (2022 erratum)")

    y, ref, new = np.concatenate(y_all), np.concatenate(ref_all), np.concatenate(new_all)
    rng = np.random.default_rng(0)
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], new[i]) - roc_auc_score(y[i], ref[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n  pooled 2003-2008: RUSBoost-28 {roc_auc_score(y, ref):.3f}   chosen {roc_auc_score(y, new):.3f}"
          f"   paired diff {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]   P(<=0) {np.mean(np.array(d) <= 0):.3f}")
    np.savez("data/out/race2.npz", y=y, rusboost=ref, chosen=new, chosen_name=str(chosen))


if __name__ == "__main__":
    main()
