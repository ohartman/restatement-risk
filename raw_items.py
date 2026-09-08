#!/usr/bin/env python3
"""Raw financial statement items as features, after Bao et al. (2020).

Their finding, which surprised the field: a booster fed the raw dollar amounts
of 28 statement line items beat the same booster fed the 14 hand-built ratios
the literature had spent twenty years on. The ratios throw information away
(a ratio cannot tell a $10M firm from a $10B one with the same margin), and
trees can build whatever ratio they need from the raw items themselves.

We have 17 consolidated items per filing from num.txt, this year and last.
Each is stored as a signed log10 so the magnitudes are comparable, along with
the year-over-year change. Missing stays NaN; the boosters read that natively.

  python raw_items.py        -> data/out/features_raw.npz  (X, adsh, names)
"""

import math
from pathlib import Path

import numpy as np

from fscore import TAGS, is_financial, read_quarter

RAW = Path("data/raw")
OUT = Path("data/out/features_raw.npz")
ITEMS = list(TAGS)


def slog(v):
    return None if v is None else math.copysign(math.log10(1 + abs(v)), v)


def main():
    names = ([f"{k}_cur" for k in ITEMS] + [f"{k}_pri" for k in ITEMS]
             + [f"{k}_chg" for k in ITEMS])
    X, adsh = [], []
    for zp in sorted(RAW.glob("*q?.zip")):
        subs, facts = read_quarter(zp)
        n = 0
        for a, s in subs.items():
            if s["form"] != "10-K" or is_financial(s["sic"]):
                continue
            d = facts.get(a)
            if not d or len(d) < 2:
                continue
            ds = sorted(d)
            cur, pri = d[ds[-1]], d[ds[-2]]
            row = []
            for k in ITEMS:
                row.append(slog(cur.get(k)))
            for k in ITEMS:
                row.append(slog(pri.get(k)))
            for k in ITEMS:
                c, p = cur.get(k), pri.get(k)
                row.append(slog(c - p) if (c is not None and p is not None) else None)
            X.append([np.nan if v is None else v for v in row])
            adsh.append(a)
            n += 1
        print(f"  {zp.stem}  {n:>4}", flush=True)
    X = np.array(X, dtype=float)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, X=X, adsh=np.array(adsh), names=np.array(names))
    print(f"\n{X.shape[0]:,} filings x {X.shape[1]} raw-item features -> {OUT}")
    print(f"coverage per item (current year): " +
          ", ".join(f"{k} {100*np.isfinite(X[:, i]).mean():.0f}%" for i, k in enumerate(ITEMS)))


if __name__ == "__main__":
    main()
