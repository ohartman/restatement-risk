#!/usr/bin/env python3
"""The record attempt: statements, then public events, insider trading and the market.

Everything learned so far, applied once:

  * the learner is settled -- balanced random forest, five seeds averaged
  * blocks are added in stages so credit lands where it belongs
  * every stage is scored on BOTH validation years (2017 and 2018, trained on
    2014-16) before the test set sees it; one validation year fooled us on
    Bao's track
  * the test years 2019-2023 are run once per stage; the paired bootstrap
    against the financial-only headline is the verdict
  * market data has a survivorship hole (Yahoo drops delisted tickers, and
    restaters delist more), so the market stage is reported two ways: on the
    covered subsample with every other stage re-scored on the same rows, and on
    the full sample with the holes left as NaN -- the second is the optimistic
    one and is labelled as such

  python record_test.py
"""

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score

from edge import impute
from events_test import joined
from final import TEST_START, ci, paired

VAL_YEARS = (2017, 2018)


def brf(seed):
    return BalancedRandomForestClassifier(n_estimators=600, min_samples_leaf=5, sampling_strategy="all",
                                          replacement=True, bootstrap=False, random_state=seed, n_jobs=-1)


def fit_score(X, y, trn, tst, seeds=5):
    Xtr, Xte = impute(X[trn], X[tst])
    return np.mean([brf(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)


def load_block(adsh, path, drop=()):
    try:
        R, names = joined(adsh, path)
    except FileNotFoundError:
        print(f"  ({path} missing; stage skipped)")
        return None, []
    keep = [i for i, n in enumerate(names) if n not in drop]
    return R[:, keep], [names[i] for i in keep]


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    X28, y, filed, adsh = z["X"], z["y"], z["filed"].astype("datetime64[D]"), z["adsh"]
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    R, _ = joined(adsh, "data/out/features_raw.npz")
    R2, _ = load_block(adsh, "data/out/features_raw2.npz", drop=("prevrpt",))
    E, _ = load_block(adsh, "data/out/features_events.npz")
    I, inames = load_block(adsh, "data/out/features_insider.npz")
    M, mnames = load_block(adsh, "data/out/features_market.npz", drop=("price_cov",))

    fin = np.hstack([X28, R, R2])
    stages = [("financial statements only (242)", fin)]
    cur = fin
    if E is not None:
        cur = np.hstack([cur, E]); stages.append(("+ auditor / CFO 8-K events", cur))
    if I is not None:
        cur = np.hstack([cur, I]); stages.append(("+ insider trading (Forms 3/4/5)", cur))
    if M is not None:
        cur = np.hstack([cur, M]); stages.append(("+ market (12 months before filing)", cur))

    trn, tst = filed < TEST_START, filed >= TEST_START
    fit_m = years < VAL_YEARS[0]
    yt = y[tst]
    print(f"train {trn.sum():,} ({y[trn].sum()})   test {tst.sum():,} ({yt.sum()})\n")

    def run(stage_rows, label, base_name):
        print(f"=== {label} ===")
        print(f"{'stage':<40}{'n':>5}{'val 2017':>10}{'val 2018':>10}{'test AUC':>10}{'95% CI':>17}{'p@100':>8}"
              f"{'  vs financial-only (paired)':>30}{'P(<=0)':>8}")
        base = None
        out = {}
        for name, X in stages:
            Xs = X[stage_rows]; ys = y[stage_rows]; fs = filed[stage_rows]; yrs = years[stage_rows]
            trn_s, tst_s = fs < TEST_START, fs >= TEST_START
            vals = []
            for v in VAL_YEARS:
                fm, vm = yrs < v, yrs == v
                vals.append(roc_auc_score(ys[vm], fit_score(Xs, ys, fm, vm, seeds=3)))
            st = fit_score(Xs, ys, trn_s, tst_s)
            yts = ys[tst_s]
            out[name] = st
            lo, hi = ci(yts, st)
            p100 = yts[np.argsort(-st)[:100]].mean()
            line = (f"{name:<40}{X.shape[1]:>5}{vals[0]:>10.3f}{vals[1]:>10.3f}{roc_auc_score(yts, st):>10.3f}"
                    f"   [{lo:.3f}-{hi:.3f}]{p100:>8.1%}")
            if base is None:
                base = st
            else:
                m, (dlo, dhi), p = paired(yts, base, st)
                line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>16.3f}"
            print(line, flush=True)
        print()
        return out

    full = run(np.ones(len(y), bool), "full sample -- market holes left as NaN (optimistic)", "fin")
    if M is not None:
        covered = np.isfinite(M[:, 0])
        print(f"market coverage: {covered.sum():,} of {len(y):,} filings; restatement rate covered "
              f"{y[covered].mean():.2%} vs not covered {y[~covered].mean():.2%}\n")
        cov = run(covered, "covered subsample -- every stage on the same rows (honest)", "fin")

    # Per-year view of the final stage against the financial-only model, full sample.
    last = stages[-1][0]
    a, b = full[stages[0][0]], full[last]
    ty = years[tst]
    print(f"per year, full sample: financial-only vs '{last}'")
    for yr in sorted(set(ty)):
        m = ty == yr
        print(f"  {yr}  {roc_auc_score(yt[m], a[m]):.3f} -> {roc_auc_score(yt[m], b[m]):.3f}  "
              f"({roc_auc_score(yt[m], b[m]) - roc_auc_score(yt[m], a[m]):+.3f})")
    np.savez("data/out/scores_record.npz", adsh=adsh[tst], y=yt,
             **{k.replace(" ", "_").replace("/", "").replace("+", "plus").replace("(", "").replace(")", "")
                .replace(",", ""): v for k, v in full.items()})


if __name__ == "__main__":
    main()
