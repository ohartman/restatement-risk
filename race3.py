#!/usr/bin/env python3
"""Second and last attempt on Bao et al.'s track: select on four validation years, not one.

race2.py picked a model on their single validation year (2001), where it beat
RUSBoost by four points, and then lost by one on 2003-2008. One year is one
draw of the period, not just of the noise. Here every candidate is scored on
four validation years -- 1998, 1999, 2000, 2001, each trained on 1991 to two
years before -- and must beat RUSBoost-28 in at least three of the four to be
eligible. The eligible candidate with the best average wins, and 2003-2008 is
run once.

This is the second time the test years are touched. Both attempts are reported.

  python race3.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from race import RAW, ndcg_at
from race2 import FEATURE_SETS, LEARNERS, add_lags, add_yrank, fill, rank, split

VAL_YEARS = (1998, 1999, 2000, 2001)
TEST_YEARS = range(2003, 2009)
BLEND_PARTS = ("BRF leaf5", "HGB", "RUSBoost")


def fit_predict(lname, fname, tr, te, y_tr):
    Xtr, Xte = fill(tr, te, FEATURE_SETS[fname])
    if lname.startswith("blend"):
        parts = [LEARNERS[l]().fit(Xtr, y_tr).predict_proba(Xte)[:, 1] for l in BLEND_PARTS]
        return np.mean([rank(p) for p in parts], axis=0)
    return LEARNERS[lname]().fit(Xtr, y_tr).predict_proba(Xte)[:, 1]


def main():
    df = pd.read_csv("data/raw/bao/data_FraudDetection_JAR2020.csv")
    df = add_yrank(add_lags(df), RAW)

    cands = [(l, f) for f in FEATURE_SETS for l in LEARNERS] + [("blend BRF+HGB+RUS", f) for f in FEATURE_SETS]
    aucs = {c: [] for c in cands}
    for v in VAL_YEARS:
        tr, te, y_tr, y_va = split(df, 1991, v - 2, v)
        print(f"validation {v}: train 1991-{v-2} {len(tr):,} ({y_tr.sum()})  validate {len(te):,} ({y_va.sum()})", flush=True)
        cache = {}
        for lname, fname in cands:
            if lname.startswith("blend"):
                s = np.mean([rank(cache[(l, fname)]) for l in BLEND_PARTS], axis=0)
            else:
                s = fit_predict(lname, fname, tr, te, y_tr)
                cache[(lname, fname)] = s
            aucs[(lname, fname)].append(roc_auc_score(y_va, s))

    ref = np.array(aucs[("RUSBoost", "28")])
    print(f"\n{'learner':<20}{'features':<16}" + "".join(f"{v:>7}" for v in VAL_YEARS) + f"{'avg':>8}{'beats RUS':>11}")
    rows = []
    for c, a in aucs.items():
        a = np.array(a)
        wins = int((a > ref).sum())
        rows.append((c, a.mean(), wins))
    for c, m, wins in sorted(rows, key=lambda r: -r[1])[:15]:
        a = aucs[c]
        print(f"{c[0]:<20}{c[1]:<16}" + "".join(f"{x:>7.3f}" for x in a) + f"{m:>8.3f}{wins:>8}/4")
    print(f"{'RUSBoost':<20}{'28':<16}" + "".join(f"{x:>7.3f}" for x in ref) + f"{ref.mean():>8.3f}{'ref':>11}")

    eligible = [r for r in rows if r[2] >= 3 and r[0] != ("RUSBoost", "28")]
    if not eligible:
        print("\nno candidate beats RUSBoost-28 in three of four validation years; nothing to run on test.")
        return
    chosen = max(eligible, key=lambda r: r[1])[0]
    print(f"\nchosen: {chosen[0]} on {chosen[1]}  (avg validation {dict(((r[0], r[1]) for r in rows))[chosen]:.3f} "
          f"vs RUSBoost-28 {ref.mean():.3f})")

    print(f"\n{'':<6}{'their RUSBoost-28':>22}{'chosen':>22}")
    y_all, ref_all, new_all, per = [], [], [], []
    for t in TEST_YEARS:
        tr, te, y_tr, y_te = split(df, 1991, t - 2, t)
        r = fit_predict("RUSBoost", "28", tr, te, y_tr)
        n = fit_predict(chosen[0], chosen[1], tr, te, y_tr)
        y_all.append(y_te); ref_all.append(r); new_all.append(n)
        per.append((roc_auc_score(y_te, r), ndcg_at(y_te, r), roc_auc_score(y_te, n), ndcg_at(y_te, n)))
        print(f"  {t}  AUC {per[-1][0]:.3f} NDCG {per[-1][1]:.3f}     AUC {per[-1][2]:.3f} NDCG {per[-1][3]:.3f}", flush=True)
    p = np.array(per)
    print(f"  avg   AUC {p[:, 0].mean():.3f} NDCG {p[:, 1].mean():.3f}     AUC {p[:, 2].mean():.3f} NDCG {p[:, 3].mean():.3f}")
    print(f"\n  published: 0.725 / 0.049 (2020)   corrected: 0.723 / 0.024 (2022 erratum)")

    y, r, n = np.concatenate(y_all), np.concatenate(ref_all), np.concatenate(new_all)
    rng = np.random.default_rng(0)
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], n[i]) - roc_auc_score(y[i], r[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n  pooled 2003-2008: RUSBoost-28 {roc_auc_score(y, r):.3f}   chosen {roc_auc_score(y, n):.3f}"
          f"   paired diff {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]   P(<=0) {np.mean(np.array(d) <= 0):.3f}")
    np.savez("data/out/race3.npz", y=y, rusboost=r, chosen=n, chosen_name=str(chosen))


if __name__ == "__main__":
    main()
