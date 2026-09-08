#!/usr/bin/env python3
"""Little r: the restatements nobody announces, read off the filings themselves.

A 10-K reports fiscal year P. The next year's 10-K reports P again, as the
comparative column. If net income, total assets, equity or revenue for P differ
between the two filings by more than a small tolerance, the company revised its
prior year without an Item 4.02 -- a "little r" revision, invisible to the label
used everywhere else in this project.

Totals are used because reclassifications move line items but not net income,
assets or equity; a change in those is a correction (or a retrospective
standard adoption, which is the noise this label carries). Both filings must be
plain 10-Ks from the same company with period ends exactly a year apart.

Self-check built in: firms that later filed an Item 4.02 should show mismatches
at a much higher rate than firms that did not. If they do not, the label is
not measuring what it claims.

  python littler.py     -> data/out/labels_littler.npz  (adsh, y_r, y_bigR, y_union, has_successor)
"""

import numpy as np
from datetime import date, datetime
from pathlib import Path

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score

from edge import impute
from events_test import joined
from fscore import is_financial, read_quarter
from final import ci, paired

RAW = Path("data/raw")
ITEMS = ["net_income", "assets", "liabilities", "revenue"]     # totals; equity = assets - liabilities
TOL = 0.01          # relative
FLOOR = 100_000.0   # dollars; below this a difference is rounding


def year_after(ddate):
    return f"{int(ddate[:4]) + 1}{ddate[4:]}"


def main():
    # 1. every 10-K's figures for its own period, and its comparatives for the prior period
    own, comp, meta = {}, {}, {}
    for zp in sorted(RAW.glob("*q?.zip")):
        subs, facts = read_quarter(zp)
        for adsh, s in subs.items():
            if s["form"] != "10-K" or is_financial(s["sic"]) or not s["period"]:
                continue
            d = facts.get(adsh)
            if not d:
                continue
            P = s["period"]
            prev = f"{int(P[:4]) - 1}{P[4:]}"
            cik = str(int(s["cik"]))
            meta[adsh] = (cik, P, s["filed"])
            if P in d:
                own[adsh] = d[P]
            if prev in d:
                comp[(cik, prev)] = (adsh, d[prev])
        print(f"  {zp.stem}", flush=True)

    # 2. match each filing to the successor that restates its period
    rows = {}
    for adsh, (cik, P, filed) in meta.items():
        succ = comp.get((cik, P))
        cur = own.get(adsh)
        if succ is None or cur is None:
            rows[adsh] = (None, None)
            continue
        succ_adsh, later = succ
        flags, n_compared = [], 0
        for k in ITEMS + ["equity"]:
            if k == "equity":
                a = (cur.get("assets") - cur.get("liabilities")) if cur.get("assets") is not None and cur.get("liabilities") is not None else None
                b = (later.get("assets") - later.get("liabilities")) if later.get("assets") is not None and later.get("liabilities") is not None else None
            else:
                a, b = cur.get(k), later.get(k)
            if a is None or b is None:
                continue
            n_compared += 1
            diff = abs(a - b)
            flags.append(diff > FLOOR and diff > TOL * max(abs(a), abs(b), 1.0))
        rows[adsh] = (int(any(flags)) if n_compared >= 2 else None, n_compared)

    # 3. line up with the feature cache and the Big-R label
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y_big, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    y_r = np.full(len(adsh), -1)
    for i, a in enumerate(adsh.tolist()):
        v, _ = rows.get(a, (None, None))
        if v is not None:
            y_r[i] = v
    has = y_r >= 0
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    print(f"\n{has.sum():,} of {len(adsh):,} filings have a successor 10-K with comparatives")
    print(f"little-r rate: {y_r[has].mean():.2%}    Big-R rate on the same filings: {y_big[has].mean():.2%}")
    both = has & (y_r == 1) & (y_big == 1)
    print(f"\nself-check -- comparative mismatch rate among filings that later drew an Item 4.02: "
          f"{y_r[has & (y_big == 1)].mean():.1%}   among the rest: {y_r[has & (y_big == 0)].mean():.1%}")
    print(f"of {int((has & (y_r == 1)).sum()):,} little-r filings, {int(both.sum()):,} are also Big R "
          f"({100*both.sum()/max(1,(has & (y_r == 1)).sum()):.0f}%); the remainder are silent revisions")
    print(f"\n{'year':<8}{'filings':>9}{'little r':>10}{'Big R':>8}{'union':>8}")
    for yr in sorted(set(years[has])):
        m = has & (years == yr)
        print(f"{yr:<8}{m.sum():>9,}{y_r[m].mean():>10.2%}{y_big[m].mean():>8.2%}{np.maximum(y_r[m], y_big[m]).mean():>8.2%}")

    y_union = np.where(has, np.maximum(y_r, y_big), y_big)
    Path("data/out").mkdir(exist_ok=True)
    np.savez("data/out/labels_littler.npz", adsh=adsh, y_r=y_r, y_bigR=y_big, y_union=y_union, has_successor=has)

    # 4. can the same features predict the silent revisions?
    R, _ = joined(adsh, "data/out/features_raw.npz")
    R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    R2 = R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]
    X = np.hstack([z["X"], R, R2])
    ts = np.datetime64("2019-01-01")
    trn, tst = has & (filed < ts), has & (filed >= ts)
    print(f"\ntrain {trn.sum():,}  test {tst.sum():,} (filings with a successor; test years 2019-2022)")

    def brf(seed):
        return BalancedRandomForestClassifier(n_estimators=600, min_samples_leaf=5, sampling_strategy="all",
                                              replacement=True, bootstrap=False, random_state=seed, n_jobs=-1)
    Xtr, Xte = impute(X[trn], X[tst])
    print(f"\n{'trained on':<12}{'scored on':<12}{'AUC':>7}{'95% CI':>17}{'positives':>11}")
    scores = {}
    for tn, ytr in (("Big R", y_big), ("little r", y_r), ("union", y_union)):
        s = np.mean([brf(k).fit(Xtr, ytr[trn]).predict_proba(Xte)[:, 1] for k in range(3)], axis=0)
        scores[tn] = s
        for en, yte in (("Big R", y_big), ("little r", y_r), ("union", y_union)):
            lo, hi = ci(yte[tst], s)
            print(f"{tn:<12}{en:<12}{roc_auc_score(yte[tst], s):>7.3f}   [{lo:.3f}-{hi:.3f}]{int(yte[tst].sum()):>11,}", flush=True)
    m, (lo, hi), p = paired(y_big[tst], scores["Big R"], scores["union"])
    print(f"\nunion-trained vs Big-R-trained, scored on Big R: {m:+.3f} [{lo:+.3f},{hi:+.3f}]  P(<=0) {p:.3f}")
    np.savez("data/out/scores_littler.npz", adsh=adsh[tst], **{k.replace(" ", "_"): v for k, v in scores.items()})


if __name__ == "__main__":
    main()
