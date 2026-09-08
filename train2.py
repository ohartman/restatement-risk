#!/usr/bin/env python3
"""Does history or industry context beat the flat feature set?

Four models, added one block at a time, so each gain is attributable:

  1. F-score                      -- the published baseline
  2. flat features                -- the current best, AUC ~0.696
  3. + company history            -- prior restatements, multi-year trends
  4. + industry percentiles       -- every ratio ranked within its SIC and year

Adding blocks cumulatively is the only way to say which one earned the gain. A
single model with everything in it would show a number and explain nothing.

  python train2.py --cutoff 2019-01-01
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from features import ALL_VARS, extra
from fscore import fscore, is_financial, read_quarter, variables
from history import HIST_VARS, build_history, industry_percentiles
from train import load_labels

RAW = Path("data/raw")
PCT_KEYS = ["roa", "net_margin", "leverage", "accruals_ta", "rev_growth",
            "asset_growth", "soft_assets", "rec_rev", "inv_rev", "rsst_acc"]
RNG = np.random.default_rng(0)


def collect(window_days):
    labels = load_labels()
    rows = []
    for zp in sorted(RAW.glob("*q?.zip")):
        subs, facts = read_quarter(zp)
        for adsh, s in subs.items():
            if s["form"] != "10-K" or is_financial(s["sic"]):
                continue
            d = facts.get(adsh)
            if not d or len(d) < 2:
                continue
            ds = sorted(d)
            fv = variables(d[ds[-1]], d[ds[-2]])
            if fv is None:
                continue
            xv = extra(d[ds[-1]], d[ds[-2]]) or {}
            try:
                filed = datetime.strptime(s["filed"], "%Y%m%d").date()
            except ValueError:
                continue
            cik = str(int(s["cik"]))
            feat = dict(fv)
            feat.update(xv)
            try:
                sic2 = int(s["sic"]) // 100
            except (TypeError, ValueError):
                sic2 = None
            rows.append({"cik": cik, "filed": filed, "sic2": sic2,
                         "name": s["name"], "feat": feat,
                         "f": fscore(fv)[1],
                         "y": int(any(0 <= (x - filed).days <= window_days
                                      for x in labels.get(cik, [])))})
        print(f"  {zp.stem}", flush=True)
    return rows, labels


def matrix(rows, blocks):
    """Feature matrix; missing values are NaN, which the booster handles natively."""
    cols = []
    for r in rows:
        v = []
        if "flat" in blocks:
            v += [r["feat"].get(k, np.nan) for k in ALL_VARS]
            v.append(np.log1p(min(max(r["f"], 0.0), 500.0)))
        if "hist" in blocks:
            v += [r["hist"].get(k, np.nan) for k in HIST_VARS]
        if "pct" in blocks:
            v += [r["pct"].get(k, np.nan) for k in PCT_KEYS]
        cols.append([np.nan if x is None else float(x) for x in v])
    return np.array(cols, dtype=float)


def boot_auc(y, s, n=400):
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if y[i].sum() and y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def run(name, Xtr, ytr, Xte, yte):
    gb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
        min_samples_leaf=40, random_state=0).fit(Xtr, ytr)
    s = gb.predict_proba(Xte)[:, 1]
    a = roc_auc_score(yte, s)
    lo, hi = boot_auc(yte, s)
    order = np.argsort(-s)
    p100 = yte[order[:100]].mean()
    print(f"{name:<34} AUC {a:.3f} [{lo:.3f}-{hi:.3f}]   p@100 {p100:.1%}")
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default="2019-01-01")
    ap.add_argument("--window-days", type=int, default=1095)
    args = ap.parse_args()
    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d").date()

    print("building features...")
    rows, labels = collect(args.window_days)
    rows = build_history(rows, labels)
    train = [r for r in rows if r["filed"] < cutoff]
    test = [r for r in rows if r["filed"] >= cutoff]
    industry_percentiles(rows, train, PCT_KEYS)

    ytr = np.array([r["y"] for r in train])
    yte = np.array([r["y"] for r in test])
    print(f"\ntrain {len(train):,} ({ytr.sum()} restated)   "
          f"test {len(test):,} ({yte.sum()} restated, {yte.mean():.2%})")
    print(f"filings with a prior restatement on record: "
          f"{sum(1 for r in rows if r['hist']['prior_restatement']):,}\n")

    fte = np.array([r["f"] for r in test])
    a = roc_auc_score(yte, fte)
    lo, hi = boot_auc(yte, fte)
    print(f"{'1. F-score (published)':<34} AUC {a:.3f} [{lo:.3f}-{hi:.3f}]   "
          f"p@100 {yte[np.argsort(-fte)[:100]].mean():.1%}")

    for name, blocks in (("2. flat features", ("flat",)),
                         ("3. + company history", ("flat", "hist")),
                         ("4. + industry percentiles", ("flat", "hist", "pct"))):
        run(name, matrix(train, blocks), ytr, matrix(test, blocks), yte)

    Path("data/out").mkdir(parents=True, exist_ok=True)
    json.dump({"cutoff": args.cutoff, "n_train": len(train), "n_test": len(test),
               "test_base_rate": float(yte.mean())},
              open("data/out/train2.json", "w"), indent=2)


if __name__ == "__main__":
    main()
