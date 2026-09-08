#!/usr/bin/env python3
"""Chase precision at the head of the ranking, where anyone would actually act.

The gradient booster wins on AUC and loses on precision@100. AUC scores the
whole distribution; precision@100 scores the only part a human ever reads. They
are different objectives and the booster is optimising the wrong one.

Before optimising anything, this bootstraps the metric. Precision@100 rests on
about ten positive filings, and a difference between six and thirteen may be
noise -- exactly the kind of small-count comparison that has misled this project
before. The confidence intervals come first; the experiments only matter if the
gap survives them.

Then four cheap attempts at the head:
  b. the booster without class balancing, which flattens the top of the ranking
  c. the F-score handed to the booster as a feature in its own right
  d. rank-averaging the two, which needs no training at all
  e. training only on the filings the F-score already finds suspicious

  python head.py --cutoff 2017-01-01
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from features import FEATURE_NAMES
from train import build

RNG = np.random.default_rng(0)


def p_at(scores, y, k=100):
    order = np.argsort(-np.asarray(scores))
    return float(np.asarray(y)[order[:k]].sum()) / k


def boot_p_at(scores, y, k=100, n=2000):
    """Bootstrap the test set to put an interval around precision@k."""
    scores, y = np.asarray(scores), np.asarray(y)
    m = len(y)
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, m, m)
        vals.append(p_at(scores[idx], y[idx], k))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def line(name, scores, y, base):
    a = roc_auc_score(y, scores)
    p100, lo, hi = boot_p_at(scores, y, 100)
    p500 = p_at(scores, y, 500)
    print(f"{name:<38} AUC {a:.3f}  p@100 {p100:.1%} [{lo:.1%}-{hi:.1%}]  "
          f"p@500 {p500:.1%} ({p500/base:.1f}x)")
    return a, p100


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default="2017-01-01")
    ap.add_argument("--window-days", type=int, default=1095)
    ap.add_argument("--drop-years", default="",
                    help="comma-separated years of labels to ignore")
    args = ap.parse_args()
    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d").date()

    print("building features...")
    drop = tuple(y for y in args.drop_years.split(',') if y)
    rows = build(args.window_days, drop)
    if drop:
        print(f'dropping label years: {", ".join(drop)}')
    train = [r for r in rows if r["filed"] < cutoff]
    test = [r for r in rows if r["filed"] >= cutoff]
    ytr = np.array([r["y"] for r in train])
    yte = np.array([r["y"] for r in test])
    Xtr = np.array([r["x"] for r in train])
    Xte = np.array([r["x"] for r in test])
    ftr = np.array([r["f"] for r in train])
    fte = np.array([r["f"] for r in test])
    base = yte.mean()
    print(f"\ntrain {len(train):,} / test {len(test):,}  base {base:.2%}")
    print("bootstrapped 95% intervals on precision@100\n")

    line("a. F-score (published)", fte, yte, base)

    gb_bal = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
        min_samples_leaf=40, class_weight="balanced", random_state=0).fit(Xtr, ytr)
    line("   gradient boosting (balanced)",
         gb_bal.predict_proba(Xte)[:, 1], yte, base)

    # b. Class balancing pushes the model to separate the bulk, which costs it
    #    the head of the ranking. Without it the top should sharpen.
    gb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
        min_samples_leaf=40, random_state=0).fit(Xtr, ytr)
    line("b. gradient boosting (unbalanced)",
         gb.predict_proba(Xte)[:, 1], yte, base)

    # c. The F-score is a particular nonlinear combination the model has to
    #    rediscover from the raw seven. Hand it over directly.
    Xtr_f = np.hstack([Xtr, np.log1p(np.clip(ftr, 0, 500)).reshape(-1, 1)])
    Xte_f = np.hstack([Xte, np.log1p(np.clip(fte, 0, 500)).reshape(-1, 1)])
    gb_f = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
        min_samples_leaf=40, random_state=0).fit(Xtr_f, ytr)
    line("c. + F-score as a feature",
         gb_f.predict_proba(Xte_f)[:, 1], yte, base)

    # d. No training at all: average the two rankings.
    def ranks(v):
        o = np.argsort(np.argsort(-np.asarray(v)))
        return o / max(1, len(o) - 1)
    blend = 1 - (ranks(fte) + ranks(gb_f.predict_proba(Xte_f)[:, 1])) / 2
    line("d. rank-average of F-score and (c)", blend, yte, base)

    # e. The F-score is cheap and high-recall; use it as a first pass and train
    #    the model only on what it flags, which is the population that matters.
    keep = ftr >= 1.0
    if keep.sum() > 200 and ytr[keep].sum() > 20:
        gb_s = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, min_samples_leaf=20,
            random_state=0).fit(Xtr_f[keep], ytr[keep])
        staged = gb_s.predict_proba(Xte_f)[:, 1] * (fte >= 1.0)
        line("e. trained on F-score>=1 only", staged, yte, base)
        print(f"   (stage-one keeps {int((fte>=1).sum()):,} of {len(fte):,} test filings)")

    Path("data/out").mkdir(parents=True, exist_ok=True)
    json.dump({"cutoff": args.cutoff, "base_rate": float(base),
               "n_train": len(train), "n_test": len(test)},
              open("data/out/head.json", "w"), indent=2)


if __name__ == "__main__":
    main()
