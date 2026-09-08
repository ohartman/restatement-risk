#!/usr/bin/env python3
"""Fourth and final attempt on Bao et al.'s track: their inputs plus free EDGAR filing behaviour.

The three earlier attempts changed the learner or re-cut the same 28 columns and
all lost. This one adds information their Compustat panel does not have -- how
each firm filed (edgar_behavior.py) -- which was the strongest block on our own
data. The claim being tested is therefore "their label, their years, their
protocol, with free public filing metadata added", not "same 28 inputs".

Because the gvkey-CIK link covers 96.7% of fraud years but 86.6% of all
firm-years, having EDGAR features is itself correlated with the label. So every
model here -- theirs included -- is trained and scored on the linked firm-years
only, and the reference is re-run on exactly those rows.

Selection on 1998-2001, must beat RUSBoost-28 in three of four, then 2003-2008
once. Fourth touch of their test years; all four are reported.

  python race5.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from edgar_behavior import NAMES as EDGAR
from race import RAW, RATIOS, ndcg_at, rusboost
from race2 import LEARNERS, fill, rank

VAL_YEARS = (1998, 1999, 2000, 2001)
TEST_YEARS = range(2003, 2009)
FEATS = {"28": RAW, "28+edgar": RAW + EDGAR, "28+ratios+edgar": RAW + RATIOS + EDGAR}
LEARN = ("RUSBoost", "BRF leaf5", "BRF leaf20", "HGB")
REF = ("RUSBoost", "28")


def split(df, t):
    tr = df[(df.fyear >= 1991) & (df.fyear <= t - 2)]
    te = df[df.fyear == t]
    y_tr = tr.misstate.values.astype(int).copy()
    cases = set(te.loc[te.misstate == 1, "p_aaer"].dropna())
    y_tr[tr.p_aaer.isin(cases).values] = 0
    return tr, te, y_tr, te.misstate.values.astype(int)


def fit_predict(lname, fname, tr, te, y_tr):
    Xtr, Xte = fill(tr, te, FEATS[fname])
    return LEARNERS[lname]().fit(Xtr, y_tr).predict_proba(Xte)[:, 1]


def main():
    import sys
    require_10k = "--require-10k" in sys.argv
    bao = pd.read_csv("data/raw/bao/data_FraudDetection_JAR2020.csv")
    eb = pd.read_csv("data/raw/bao/edgar_behavior.csv")
    df = bao.merge(eb, on=["gvkey", "fyear"], how="inner")
    if require_10k:
        # Whether a fiscal-year 10-K exists on EDGAR tracks the label (AAER firms are larger and
        # better covered), so the honest version keeps only firm-years with a matched 10-K and
        # drops the indicator; every model, the reference included, sees the same rows.
        df = df[df.no_10k_found == 0].copy()
        for k in FEATS:
            FEATS[k] = [c for c in FEATS[k] if c != "no_10k_found"]
        print("restricted to firm-years with a matched 10-K; no_10k_found dropped")
    print(f"linked firm-years {len(df):,} of {len(bao):,}; fraud {int(df.misstate.sum())} of {int(bao.misstate.sum())}")
    for n in EDGAR:
        ok = df[n].notna()
        if ok.sum() and 0 < df.misstate[ok].sum() < ok.sum():
            print(f"  {n:<16} coverage {ok.mean():5.1%}  AUC alone {roc_auc_score(df.misstate[ok], df[n][ok]):.3f}")

    cands = [(l, f) for f in FEATS for l in LEARN]
    aucs = {c: [] for c in cands}
    for v in VAL_YEARS:
        tr, te, y_tr, y_va = split(df, v)
        for l, f in cands:
            try:
                aucs[(l, f)].append(roc_auc_score(y_va, fit_predict(l, f, tr, te, y_tr)))
            except ValueError as exc:
                aucs[(l, f)].append(np.nan); print(f"    {l} {f} {v}: {exc}")
        print(f"validation {v} done", flush=True)
    ref = np.array(aucs[REF])
    print(f"\n{'learner':<12}{'features':<18}" + "".join(f"{v:>7}" for v in VAL_YEARS) + f"{'avg':>8}{'beats REF':>11}")
    rows = [(c, np.nanmean(a), int((np.array(a) > ref).sum())) for c, a in aucs.items()]
    for c, m, w in sorted(rows, key=lambda r: -r[1]):
        print(f"{c[0]:<12}{c[1]:<18}" + "".join(f"{x:>7.3f}" for x in aucs[c]) + f"{m:>8.3f}{w:>8}/4")

    eligible = [r for r in rows if r[2] >= 3 and r[0] != REF]
    if not eligible:
        print("\nno candidate beats the reference in three of four validation years; test not run."); return
    chosen = max(eligible, key=lambda r: r[1])[0]
    also = ("RUSBoost", "28+edgar")          # their learner with our columns, declared in advance
    print(f"\nchosen: {chosen}   also reported: {also} (their learner, our columns)")

    print(f"\n{'':<6}{'RUSBoost-28 (ref)':>20}{'chosen':>20}{'RUSBoost-28+edgar':>22}")
    y_all, r_all, n_all, a_all = [], [], [], []
    for t in TEST_YEARS:
        tr, te, y_tr, y_te = split(df, t)
        r = fit_predict(*REF, tr, te, y_tr); n = fit_predict(*chosen, tr, te, y_tr); a = fit_predict(*also, tr, te, y_tr)
        y_all.append(y_te); r_all.append(r); n_all.append(n); a_all.append(a)
        print(f"  {t}  AUC {roc_auc_score(y_te, r):.3f} NDCG {ndcg_at(y_te, r):.3f}   AUC {roc_auc_score(y_te, n):.3f} NDCG {ndcg_at(y_te, n):.3f}"
              f"   AUC {roc_auc_score(y_te, a):.3f} NDCG {ndcg_at(y_te, a):.3f}", flush=True)
    y, r, n, a = map(np.concatenate, (y_all, r_all, n_all, a_all))
    print(f"  avg   AUC {np.mean([roc_auc_score(yy, rr) for yy, rr in zip(y_all, r_all)]):.3f}"
          f"{'':<14}AUC {np.mean([roc_auc_score(yy, nn) for yy, nn in zip(y_all, n_all)]):.3f}"
          f"{'':<14}AUC {np.mean([roc_auc_score(yy, aa) for yy, aa in zip(y_all, a_all)]):.3f}")
    print(f"\n  published: 0.725 / 0.049 (2020)   corrected: 0.723 / 0.024 (2022 erratum)")
    rng = np.random.default_rng(0)
    for name, s in (("chosen", n), ("RUSBoost-28+edgar", a)):
        d = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(i):
                d.append(roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], r[i]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  pooled 2003-2008: ref {roc_auc_score(y, r):.3f}   {name} {roc_auc_score(y, s):.3f}"
              f"   paired diff {np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]   P(<=0) {np.mean(np.array(d) <= 0):.3f}")
    np.savez("data/out/race5.npz", y=y, ref=r, chosen=n, rus_edgar=a, chosen_name=str(chosen))


if __name__ == "__main__":
    main()
