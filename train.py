#!/usr/bin/env python3
"""Can anything beat the F-score at predicting restatements?

Three models on the same filings, the same split, the same labels:

  1. the published F-score            -- 2011 variables, 2011 coefficients
  2. logistic regression on those     -- same variables, coefficients refit here
  3. gradient boosting on all of them -- the extra block as well

The middle model is the one that makes the comparison interpretable. If refitting
the original seven variables closes most of the gap, the F-score's problem was
stale coefficients. If it does not, the variables themselves are the limit and
more of them is the only way forward.

The split is by time, not at random: train on filings made before the cutoff,
test on filings made after. A random split would let the model learn from a
company's 2018 filing to predict its 2017 one, and would flatter every number
here. Companies do appear on both sides, which is realistic -- deployment means
scoring next year the same firms you scored last year.

  python train.py --cutoff 2017-01-01
"""

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, FSCORE_VARS, extra, vector
from fscore import fscore, is_financial, read_quarter, variables

RAW = Path("data/raw")


def load_labels(drop_years=()):
    """cik -> Item 4.02 dates, optionally dropping whole years.

    2021 is the year worth dropping. The SEC's April 2021 statement on SPAC
    warrant accounting, and its November statement on Class A share
    classification, forced several hundred blank-check companies to restate the
    same line item within weeks. Those are regulatory events, not accounting
    failures, and nothing in a balance sheet predicts them.
    """
    by_cik = defaultdict(list)
    for line in (RAW / "restatements.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if not r.get("file_date"):
            continue
        if r["file_date"][:4] in drop_years:
            continue
        d = datetime.strptime(r["file_date"], "%Y-%m-%d").date()
        for cik in r.get("ciks") or []:
            by_cik[str(int(cik))].append(d)
    return by_cik


def build(window_days, drop_years=()):
    labels = load_labels(drop_years)
    rows = []
    for zp in sorted(RAW.glob("*q?.zip")):
        subs, facts = read_quarter(zp)
        n = 0
        for adsh, s in subs.items():
            if s["form"] != "10-K" or is_financial(s["sic"]):
                continue
            d = facts.get(adsh)
            if not d or len(d) < 2:
                continue
            ds = sorted(d)
            cur, pri = d[ds[-1]], d[ds[-2]]
            fv = variables(cur, pri)
            if fv is None:
                continue
            xv = extra(cur, pri)
            try:
                filed = datetime.strptime(s["filed"], "%Y%m%d").date()
            except ValueError:
                continue
            cik = str(int(s["cik"]))
            hit = any(0 <= (x - filed).days <= window_days
                      for x in labels.get(cik, []))
            rows.append({"adsh": adsh, "cik": cik, "name": s["name"],
                         "filed": filed, "y": int(hit),
                         "f": fscore(fv)[1], "fv": fv,
                         "x": vector(fv, xv)})
            n += 1
        print(f"  {zp.stem}  {n:>4}", flush=True)
    return rows


def report(name, scores, y, base):
    a = roc_auc_score(y, scores)
    order = np.argsort(-np.asarray(scores))
    line = f"{name:<34} AUC {a:.3f}"
    for k in (100, 500):
        if k <= len(order):
            hit = int(np.asarray(y)[order[:k]].sum())
            line += f"   p@{k} {hit/k:.2%} ({(hit/k)/base:.1f}x)"
    print(line)
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default="2017-01-01")
    ap.add_argument("--window-days", type=int, default=1095)
    ap.add_argument("--drop-years", default="")
    args = ap.parse_args()
    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d").date()

    print("building features...")
    drop = tuple(y for y in args.drop_years.split(',') if y)
    rows = build(args.window_days, drop)
    train = [r for r in rows if r["filed"] < cutoff]
    test = [r for r in rows if r["filed"] >= cutoff]
    print(f"\ntrain {len(train):,} filings ({sum(r['y'] for r in train)} restated)")
    print(f"test  {len(test):,} filings ({sum(r['y'] for r in test)} restated)")
    if not test or not sum(r["y"] for r in test):
        print("no positives in the test period - adjust the cutoff or fetch more labels")
        return

    ytr = np.array([r["y"] for r in train])
    yte = np.array([r["y"] for r in test])
    base = yte.mean()
    print(f"test base rate {base:.2%}\n")

    report("1. F-score (published)", [r["f"] for r in test], yte, base)

    # 2. Same seven variables, coefficients refit on the training period.
    idx = [FEATURE_NAMES.index(k) for k in FSCORE_VARS]
    Xtr7 = np.array([[r["x"][i] for i in idx] for r in train])
    Xte7 = np.array([[r["x"][i] for i in idx] for r in test])
    sc = StandardScaler().fit(Xtr7)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(sc.transform(Xtr7), ytr)
    report("2. logistic, F-score vars refit",
           lr.predict_proba(sc.transform(Xte7))[:, 1], yte, base)

    # 3. Everything.
    Xtr = np.array([r["x"] for r in train])
    Xte = np.array([r["x"] for r in test])
    gb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
        l2_regularization=1.0, min_samples_leaf=40,
        class_weight="balanced", random_state=0)
    gb.fit(Xtr, ytr)
    report("3. gradient boosting, all vars",
           gb.predict_proba(Xte)[:, 1], yte, base)

    print(f"\n{len(FEATURE_NAMES)} features, {len(train):,} training filings")
    Path("data/out").mkdir(parents=True, exist_ok=True)
    json.dump({"cutoff": args.cutoff, "window_days": args.window_days,
               "n_train": len(train), "n_test": len(test),
               "test_base_rate": float(base)},
              open("data/out/run.json", "w"), indent=2)


if __name__ == "__main__":
    main()
