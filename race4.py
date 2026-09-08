#!/usr/bin/env python3
"""The persistence attempt on Bao et al.'s track. Final, pre-declared.

Fraud is a firm trait more than a year event: an AAER covers two or three
consecutive fiscal years on average, and on our own data the firm-level label
beat the year-level label by five points. Yet every model so far scores each
firm-year in isolation, from that one year's 28 numbers.

At time t a firm's numbers for t-1 and t-2 are public too. So: fit the model as
before, score the test firm's current year AND its previous one or two years
with the same model, and rank by the decayed average. No labels, no future
information, nothing outside the 28 items -- just less noise per firm.

Candidates (every combination, chosen on four validation years, must beat
RUSBoost-28 in three of them):
  learner       RUSBoost, balanced forest (leaf 5 / 20), HGB
  features      28 items; 28 + prior-year values and changes
  smoothing     none; current + previous year; current + two previous years
  train from    1991 (theirs); 1996 (drop the early-90s regime)

The 2003-2008 test years are run once with the pick. This is the third and last
time they are touched; all three attempts are reported.

  python race4.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from race import RAW, ndcg_at
from race2 import FEATURE_SETS, LEARNERS, add_lags, fill

VAL_YEARS = (1998, 1999, 2000, 2001)
TEST_YEARS = range(2003, 2009)
LEARN = ("RUSBoost", "BRF leaf5", "BRF leaf20", "HGB")
FEATS = ("28", "28+lag")
SMOOTH = {"none": (1.0,), "persist1": (1.0, 0.6), "persist2": (1.0, 0.6, 0.36)}
STARTS = (1991, 1996)
REF = ("RUSBoost", "28", "none", 1991)


def split(df, start, t):
    tr = df[(df.fyear >= start) & (df.fyear <= t - 2)]
    te = df[df.fyear == t]
    y_tr = tr.misstate.values.astype(int).copy()
    cases = set(te.loc[te.misstate == 1, "p_aaer"].dropna())
    y_tr[tr.p_aaer.isin(cases).values] = 0          # their serial-fraud step
    return tr, te, y_tr, te.misstate.values.astype(int)


def scores_by_year(df, lname, fname, start, t):
    """Fit on the training window; return {fyear: Series(score, index=gvkey)} for t, t-1, t-2."""
    # A late start with an early validation year leaves too few years to learn from
    # (1996 start, 1998 validation = one training year); always keep at least four.
    tr, te, y_tr, _ = split(df, min(start, t - 5), t)
    cols = FEATURE_SETS[fname]
    med = tr[cols].median()
    try:
        m = LEARNERS[lname]().fit(tr[cols].fillna(med).values, y_tr)
    except ValueError as exc:        # AdaBoost gives up when a round is worse than random
        print(f"    {lname} on {fname} from {start} for {t}: {exc}", flush=True)
        return None, te
    out = {}
    for yr in (t, t - 1, t - 2):
        rows = df[df.fyear == yr]
        s = m.predict_proba(rows[cols].fillna(med).values)[:, 1]
        out[yr] = pd.Series(s, index=rows.gvkey.values)
    return out, te


def smooth(by_year, te, t, weights):
    """Decayed average of the firm's scores over the years available."""
    num = np.zeros(len(te)); den = np.zeros(len(te))
    for k, w in enumerate(weights):
        s = by_year[t - k].reindex(te.gvkey.values).values
        ok = np.isfinite(s)
        num[ok] += w * s[ok]; den[ok] += w
    return num / np.where(den > 0, den, 1)


def main():
    df = add_lags(pd.read_csv("data/raw/bao/data_FraudDetection_JAR2020.csv"))
    assert not df.duplicated(["gvkey", "fyear"]).any()

    cands = [(l, f, s, st) for st in STARTS for f in FEATS for l in LEARN for s in SMOOTH]
    aucs = {c: [] for c in cands}
    for v in VAL_YEARS:
        for st in STARTS:
            for f in FEATS:
                for l in LEARN:
                    by_year, te = scores_by_year(df, l, f, st, v)
                    y_va = te.misstate.values.astype(int)
                    for s, w in SMOOTH.items():
                        aucs[(l, f, s, st)].append(
                            roc_auc_score(y_va, smooth(by_year, te, v, w)) if by_year else np.nan)
        print(f"validation {v} done", flush=True)

    ref = np.array(aucs[REF])
    print(f"\n{'learner':<12}{'features':<9}{'smooth':<10}{'from':<6}" + "".join(f"{v:>7}" for v in VAL_YEARS)
          + f"{'avg':>8}{'beats REF':>11}")
    rows = [(c, np.nanmean(a), int((np.array(a) > ref).sum())) for c, a in aucs.items()]
    for c, m, wins in sorted(rows, key=lambda r: -r[1])[:20]:
        print(f"{c[0]:<12}{c[1]:<9}{c[2]:<10}{c[3]:<6}" + "".join(f"{x:>7.3f}" for x in aucs[c]) + f"{m:>8.3f}{wins:>8}/4")
    print(f"{'REF: ' + REF[0]:<12}{REF[1]:<9}{REF[2]:<10}{REF[3]:<6}" + "".join(f"{x:>7.3f}" for x in ref) + f"{ref.mean():>8.3f}")

    # How much does smoothing alone add, holding everything else at the reference?
    print("\nsmoothing alone, on RUSBoost-28 from 1991 (validation average):")
    for s in SMOOTH:
        print(f"  {s:<10}{np.mean(aucs[('RUSBoost', '28', s, 1991)]):.3f}")

    eligible = [r for r in rows if r[2] >= 3 and r[0] != REF]
    if not eligible:
        print("\nno candidate beats the reference in three of four validation years; test not run.")
        return
    chosen = max(eligible, key=lambda r: r[1])[0]
    print(f"\nchosen: {chosen}   avg validation {dict((r[0], r[1]) for r in rows)[chosen]:.3f} vs reference {ref.mean():.3f}")

    print(f"\n{'':<6}{'their RUSBoost-28':>22}{'chosen':>22}")
    y_all, r_all, n_all, per = [], [], [], []
    for t in TEST_YEARS:
        by_ref, te = scores_by_year(df, REF[0], REF[1], REF[3], t)
        r = smooth(by_ref, te, t, SMOOTH[REF[2]])
        by_new, _ = scores_by_year(df, chosen[0], chosen[1], chosen[3], t)
        n = smooth(by_new, te, t, SMOOTH[chosen[2]])
        y = te.misstate.values.astype(int)
        y_all.append(y); r_all.append(r); n_all.append(n)
        per.append((roc_auc_score(y, r), ndcg_at(y, r), roc_auc_score(y, n), ndcg_at(y, n)))
        print(f"  {t}  AUC {per[-1][0]:.3f} NDCG {per[-1][1]:.3f}     AUC {per[-1][2]:.3f} NDCG {per[-1][3]:.3f}", flush=True)
    p = np.array(per)
    print(f"  avg   AUC {p[:, 0].mean():.3f} NDCG {p[:, 1].mean():.3f}     AUC {p[:, 2].mean():.3f} NDCG {p[:, 3].mean():.3f}")
    print(f"\n  published: 0.725 / 0.049 (2020)   corrected: 0.723 / 0.024 (2022 erratum)")

    y, r, n = np.concatenate(y_all), np.concatenate(r_all), np.concatenate(n_all)
    rng = np.random.default_rng(0)
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], n[i]) - roc_auc_score(y[i], r[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n  pooled 2003-2008: RUSBoost-28 {roc_auc_score(y, r):.3f}   chosen {roc_auc_score(y, n):.3f}"
          f"   paired diff {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]   P(<=0) {np.mean(np.array(d) <= 0):.3f}")
    np.savez("data/out/race4.npz", y=y, rusboost=r, chosen=n, chosen_name=str(chosen))


if __name__ == "__main__":
    main()
