#!/usr/bin/env python3
"""The regulator's own attention: SEC comment letters and other EDGAR filing events, per 10-K.

When the SEC's Division of Corporation Finance reviews a filing and has questions,
it sends a comment letter; the exchange is posted on EDGAR as form type UPLOAD
(the SEC's letter) and CORRESP (the company's reply). A firm under review is a
firm someone is already looking at hard, and the literature finds comment
letters predict restatements (Cassell, Dreher & Myers 2013). Nothing in our
feature set knows about them. Neither do the late-filing notices (NT 10-K,
NT 10-Q) -- the formal "we cannot file on time" -- or the quarterly amendments.

All from EDGAR's quarterly form indices 2010-2023, already on disk, using only
filings dated before each 10-K:

  sec_letters_1y, sec_letters_2y    UPLOAD filings (SEC letters) in the prior 1 / 2 years
  corresp_2y                        the company's CORRESP replies in the prior 2 years
  days_since_letter                 days since the last SEC letter (capped at 5 years)
  nt_10k_2y, nt_10q_2y              late-filing notices in the prior 2 years
  n_10ka_3y, n_10qa_3y              10-K/A and 10-Q/A amendments in the prior 3 years
  n_8k_1y                           8-Ks in the prior year
  n_filings_1y                      everything in the prior year

  python letters_features.py     -> data/out/features_letters.npz (X, adsh, names)
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from edgar_behavior import parse_index, LINE
from relabel import submissions

IDX = Path("data/raw/edgar_index")
NAMES = ["sec_letters_1y", "sec_letters_2y", "corresp_2y", "days_since_letter", "nt_10k_2y", "nt_10q_2y",
         "n_10ka_3y", "n_10qa_3y", "n_8k_1y", "n_filings_1y"]
KEEP = ("UPLOAD", "CORRESP", "NT 10-K", "NT 10-Q", "10-K/A", "10-Q/A", "8-K")


PUBLIC = Path("data/raw/edgar_daily/dissemination.jsonl")   # accession -> the day it was disseminated
ACC = re.compile(r"\d{10}-\d{2}-\d{6}")


def public_dates():
    """Comment letters (UPLOAD) and replies (CORRESP) are dated by the day they were written but
    released weeks after the review closes; the daily indices give the day they became public."""
    out = {}
    if PUBLIC.exists():
        for l in PUBLIC.open(encoding="utf-8"):
            r = json.loads(l)
            if r["acc"] not in out:
                out[r["acc"]] = date.fromisoformat(r["public"])
    return out


def main():
    pub = public_dates(); print(f"{len(pub):,} letters with a public date; letters without one are moved 31 days later (the observed release lag)")
    filings = defaultdict(list); moved = missing = 0
    for p in sorted(IDX.glob("form_20[12]*_Q*.idx")):
        n = 0
        with p.open(encoding="latin-1") as fh:
            for l in fh:
                m = LINE.match(l.rstrip())
                if not m:
                    continue
                form, cik, d = m.group(1).strip(), int(m.group(3)), date.fromisoformat(m.group(4))
                if form in ("UPLOAD", "CORRESP"):
                    a = ACC.search(l.rsplit(None, 1)[-1])          # the file path at the end of the row
                    if a and a.group(0) in pub:
                        d = pub[a.group(0)]; moved += 1
                    else:
                        d = d + timedelta(days=31); missing += 1
                filings[cik].append((d, form)); n += 1
        print(f"  {p.name}: {n:,}", flush=True)
    print(f"letters dated by their public day: {moved:,}; by letter date + 31 days: {missing:,}")
    for v in filings.values():
        v.sort()
    print(f"{sum(len(v) for v in filings.values()):,} filings 2010-2023 for {len(filings):,} CIKs")

    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y = z["adsh"], z["y"]
    subs = submissions()
    X = np.full((len(adsh), len(NAMES)), np.nan)
    for i, a in enumerate(adsh.tolist()):
        cik, filed, _ = subs.get(a, (None, None, None))
        if cik is None:
            continue
        fl = filings.get(int(cik))
        if not fl:
            X[i] = [0, 0, 0, 1825, 0, 0, 0, 0, 0, 0]
            continue
        dates = [d for d, _ in fl]
        hi = bisect_left(dates, filed)                       # strictly before the 10-K
        def window(days):
            lo = bisect_left(dates, filed - timedelta(days=days))
            return fl[lo:hi]
        w1, w2, w3 = window(365), window(730), window(1095)
        letters = [d for d, f in fl[:hi] if f == "UPLOAD"]
        X[i] = [sum(1 for _, f in w1 if f == "UPLOAD"), sum(1 for _, f in w2 if f == "UPLOAD"),
                sum(1 for _, f in w2 if f == "CORRESP"),
                min((filed - letters[-1]).days, 1825) if letters else 1825,
                sum(1 for _, f in w2 if f.startswith("NT 10-K")), sum(1 for _, f in w2 if f.startswith("NT 10-Q")),
                sum(1 for _, f in w3 if f in ("10-K/A", "10-K405/A")), sum(1 for _, f in w3 if f == "10-Q/A"),
                sum(1 for _, f in w1 if f.startswith("8-K")), len(w1)]
        if i % 10000 == 0:
            print(f"  {i:,}/{len(adsh):,}", flush=True)
    Path("data/out").mkdir(exist_ok=True)
    np.savez("data/out/features_letters.npz", X=X, adsh=adsh, names=np.array(NAMES))
    from sklearn.metrics import roc_auc_score
    has = np.isfinite(X[:, 0])
    lt = X[:, 1] > 0
    print(f"\nfilings with an SEC comment letter in the prior two years: {lt.sum():,} ({100*lt.mean():.1f}%); "
          f"restatement rate {y[lt].mean():.2%} vs {y[~lt].mean():.2%}")
    nt = X[:, 4] > 0
    print(f"filings with an NT 10-K in the prior two years: {nt.sum():,} ({100*nt.mean():.1f}%); "
          f"restatement rate {y[nt].mean():.2%} vs {y[~nt].mean():.2%}")
    for k, n in enumerate(NAMES):
        v = X[has, k]
        print(f"  {n:<20} AUC alone {roc_auc_score(y[has], v):.3f}")


if __name__ == "__main__":
    main()
