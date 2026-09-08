#!/usr/bin/env python3
"""Turn the harvested 8-K events into per-10-K features, using only the past.

For each 10-K in the feature cache, look at the same company's Item 4.01
(auditor change) and CFO-related Item 5.02 (officer turnover) 8-Ks filed
*before* the 10-K's own filing date, and record how many there were in the
prior one and three years and how long ago the most recent one was. Nothing
filed after the 10-K is used; that would be looking at the answer.

  python event_features.py      -> data/out/features_events.npz (X, adsh, names)
"""

import json
from bisect import bisect_left
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

from relabel import submissions

RAW = Path("data/raw")
OUT = Path("data/out/features_events.npz")
EVENTS = {"aud": RAW / "events_401.jsonl", "cfo": RAW / "events_502.jsonl"}


def load_events(path):
    by = defaultdict(list)
    n = 0
    if not path.exists():
        print(f"  {path} missing; its features will be all-NaN")
        return by
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if not r.get("file_date"):
            continue
        d = datetime.strptime(r["file_date"], "%Y-%m-%d").date()
        for cik in r.get("ciks") or []:
            by[str(int(cik))].append(d)
            n += 1
    for v in by.values():
        v.sort()
    print(f"  {path.name}: {n:,} company-events across {len(by):,} companies")
    return by


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh = z["adsh"]
    subs = submissions()
    names, cols = [], []
    for key, path in EVENTS.items():
        by = load_events(path)
        n1 = np.full(len(adsh), np.nan); n3 = np.full(len(adsh), np.nan)
        since = np.full(len(adsh), np.nan)
        for i, a in enumerate(adsh.tolist()):
            cik, filed, _ = subs.get(a, (None, None, None))
            if cik is None:
                continue
            ds = by.get(cik, [])
            # Events strictly before the 10-K filing date.
            k = bisect_left(ds, filed)
            past = ds[:k]
            n1[i] = sum(1 for d in past if (filed - d).days <= 365)
            n3[i] = sum(1 for d in past if (filed - d).days <= 1095)
            # Days since the last event, capped at ten years; none ever -> cap.
            since[i] = min((filed - past[-1]).days, 3650) if past else 3650
        names += [f"{key}_n1y", f"{key}_n3y", f"{key}_days_since"]
        cols += [n1, n3, since]
        have = np.isfinite(n1)
        print(f"  {key}: {int((n3[have] > 0).sum()):,} of {int(have.sum()):,} 10-Ks had an event "
              f"in the prior three years ({100*(n3[have] > 0).mean():.1f}%)")
    X = np.column_stack(cols)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, X=X, adsh=adsh, names=np.array(names))
    print(f"\n{X.shape[0]:,} filings x {X.shape[1]} event features -> {OUT}")


if __name__ == "__main__":
    main()
