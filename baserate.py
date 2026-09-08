#!/usr/bin/env python3
"""Is the AUC real, or is the test period simply richer in restatements?

Training filings restate at 3.68%, test filings at 5.22%. A model can look good
on a shifted base rate without having learned anything transferable, so this
checks three things before any more tuning happens.

  1. The rate by filing year, to see whether the shift is a trend or an artifact.
  2. Whether the label window is even -- a 2014 filing has had a decade to be
     restated and its label is complete; a 2023 filing has had two years, and
     restatements announced in 2027 have not happened yet. Right-censoring
     depresses the apparent rate for recent filings, which would push the
     shift in the opposite direction and is worth knowing about.
  3. AUC computed within each test year separately. If the model only works
     because later years have more positives, the per-year numbers collapse
     toward 0.5. If it holds year by year, the result is real.

  python baserate.py --cutoff 2019-01-01
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from train import build


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default="2019-01-01")
    ap.add_argument("--window-days", type=int, default=1095)
    args = ap.parse_args()
    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d").date()

    rows = build(args.window_days)
    by_year = defaultdict(lambda: [0, 0])
    for r in rows:
        y = r["filed"].year
        by_year[y][0] += 1
        by_year[y][1] += r["y"]

    print(f"\n{'year':<8}{'filings':>9}{'restated':>10}{'rate':>8}   window")
    last_label = max((max(r["filed"] for r in rows)).year, 2026)
    for y in sorted(by_year):
        n, k = by_year[y]
        # A filing is only fully observed if its whole window has elapsed and
        # labels exist for that span.
        complete = "full" if y + args.window_days // 365 <= last_label else "CENSORED"
        print(f"{y:<8}{n:>9,}{k:>10,}{k/max(1,n):>8.2%}   {complete}")

    train = [r for r in rows if r["filed"] < cutoff]
    test = [r for r in rows if r["filed"] >= cutoff]
    print(f"\ntrain base {np.mean([r['y'] for r in train]):.2%}   "
          f"test base {np.mean([r['y'] for r in test]):.2%}")

    Xtr = np.array([r["x"] for r in train]); ytr = np.array([r["y"] for r in train])
    ftr = np.array([r["f"] for r in train])
    Xtr = np.hstack([Xtr, np.log1p(np.clip(ftr, 0, 500)).reshape(-1, 1)])
    gb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
        min_samples_leaf=40, random_state=0).fit(Xtr, ytr)

    print("\nAUC within each test year (a real model holds up year by year):")
    print(f"  {'year':<8}{'filings':>9}{'restated':>10}{'rate':>8}{'F-score':>10}{'model':>9}")
    for y in sorted({r['filed'].year for r in test}):
        sub = [r for r in test if r["filed"].year == y]
        ys = np.array([r["y"] for r in sub])
        if ys.sum() < 10 or ys.sum() == len(ys):
            print(f"  {y:<8}{len(sub):>9,}{int(ys.sum()):>10}{'  too few positives':>27}")
            continue
        X = np.array([r["x"] for r in sub])
        f = np.array([r["f"] for r in sub])
        X = np.hstack([X, np.log1p(np.clip(f, 0, 500)).reshape(-1, 1)])
        a_m = roc_auc_score(ys, gb.predict_proba(X)[:, 1])
        a_f = roc_auc_score(ys, f)
        print(f"  {y:<8}{len(sub):>9,}{int(ys.sum()):>10}{ys.mean():>8.2%}"
              f"{a_f:>10.3f}{a_m:>9.3f}")


if __name__ == "__main__":
    main()
