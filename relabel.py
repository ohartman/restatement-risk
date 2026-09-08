#!/usr/bin/env python3
"""Tighten the labels to the periods each Item 4.02 actually condemns, and measure.

Loose label (current):  a 10-K is positive if the same company files any
                        Item 4.02 within three years of it.
Tight label (this):     a 10-K is positive only if an Item 4.02 filed after it
                        names the fiscal year that 10-K covers.

The loose label calls a clean 2018 10-K "restated" because the company withdrew
its 2020 statements. That is label noise, and label noise is a ceiling the model
cannot train through. If tightening the labels raises AUC, the ceiling was in the
labels; if it does not, the parser is not recovering enough to matter.

Same features, same temporal split, same model -- only the label changes. The
feature cache from tune.py is reused; the fiscal period of each 10-K comes from
sub.txt, which is cheap to read without touching num.txt.

  python relabel.py --test-start 2019-01-01
"""

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

RAW = Path("data/raw")
RNG = np.random.default_rng(3)


def submissions():
    """adsh -> (cik, filed, period_year) from every quarter's sub.txt."""
    out = {}
    for zp in sorted(RAW.glob("*q?.zip")):
        z = zipfile.ZipFile(zp)
        with z.open("sub.txt") as f:
            for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"),
                                    delimiter="\t"):
                try:
                    filed = datetime.strptime(r["filed"], "%Y%m%d").date()
                    pyear = int(r["period"][:4]) if r["period"] else None
                except ValueError:
                    continue
                out[r["adsh"]] = (str(int(r["cik"])), filed, pyear)
    return out


def tight_labels():
    """cik -> list of (announcement date, set of condemned fiscal years)."""
    by = defaultdict(list)
    parsed = total = 0
    for l in (RAW / "restatement_periods.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        total += 1
        if not r.get("file_date"):
            continue
        d = datetime.strptime(r["file_date"], "%Y-%m-%d").date()
        ys = set(r.get("years") or [])
        if ys:
            parsed += 1
        by[r["cik"]].append((d, ys))
    print(f"Item 4.02 filings {total:,}, with parseable years {parsed:,} "
          f"({100*parsed/max(1,total):.0f}%)")
    return by


def boot(y, s, n=400):
    v = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-start", default="2019-01-01")
    ap.add_argument("--window-days", type=int, default=1095)
    args = ap.parse_args()
    ts = np.datetime64(args.test_start)

    z = np.load("data/out/features.npz", allow_pickle=True)
    X, y_loose, filed, adsh = z["X"], z["y"], z["filed"].astype("datetime64[D]"), z["adsh"]
    subs = submissions()
    labels = tight_labels()

    y_tight = np.zeros_like(y_loose)
    unparsed_fallback = 0
    for i, a in enumerate(adsh.tolist()):
        cik, fdate, pyear = subs.get(a, (None, None, None))
        if cik is None:
            continue
        for ann, years in labels.get(cik, []):
            days = (ann - fdate).days
            if not (0 <= days <= args.window_days):
                continue
            if years:
                # The announcement must name the year this 10-K covers.
                if pyear in years:
                    y_tight[i] = 1
                    break
            else:
                # Nothing parseable: fall back to the loose rule for this one.
                y_tight[i] = 1
                unparsed_fallback += 1
                break

    trn, tst = filed < ts, filed >= ts
    print(f"\n{'':<22}{'positives (train)':>18}{'positives (test)':>17}{'test rate':>11}")
    for name, yy in (("loose label", y_loose), ("tight label", y_tight)):
        print(f"{name:<22}{int(yy[trn].sum()):>18,}{int(yy[tst].sum()):>17,}"
              f"{yy[tst].mean():>11.2%}")
    flipped = int((y_loose != y_tight).sum())
    print(f"\nlabels changed by tightening: {flipped:,}  "
          f"(unparsed announcements kept loose: {unparsed_fallback:,})\n")

    params = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                  l2_regularization=1.0, min_samples_leaf=40, random_state=0)
    print(f"{'trained on / scored on':<30}{'AUC':>7}{'95% CI':>17}{'p@100':>8}")
    for tn, ytr in (("loose", y_loose), ("tight", y_tight)):
        m = HistGradientBoostingClassifier(**params).fit(X[trn], ytr[trn])
        s = m.predict_proba(X[tst])[:, 1]
        for en, yte in (("loose", y_loose), ("tight", y_tight)):
            a = roc_auc_score(yte[tst], s)
            lo, hi = boot(yte[tst], s)
            p100 = yte[tst][np.argsort(-s)[:100]].mean()
            print(f"  {tn:<7} / {en:<20}{a:>7.3f}   [{lo:.3f}-{hi:.3f}]{p100:>8.1%}")

    np.savez("data/out/features_tight.npz", X=X, y=y_tight, filed=z["filed"],
             f=z["f"], adsh=adsh)
    print("\nwrote data/out/features_tight.npz")


if __name__ == "__main__":
    main()
