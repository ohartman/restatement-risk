#!/usr/bin/env python3
"""Board and officer interlocks, from the Forms 3/4/5 data sets.

Every insider filing carries the reporting person's own CIK, so the bulk data
is a person-firm graph over time: who sat where, when. Misconduct travels along
that graph -- a director who has watched one company restate is more likely to
be sitting at the next one (Chiu, Teoh & Tian 2013). Nobody has built this from
free data, and it costs nothing here because the files are already on disk.

For each 10-K filed at date t, using only what was public before t:
  n_insiders          people who filed as officer/director of this firm in the prior 3 years
  n_interlocked       of those, how many also filed as insider of another firm in that window
  n_linked_firms      distinct other firms reached through them
  n_linked_402        of those firms, how many had announced an Item 4.02 before t
  frac_linked_402     n_linked_402 / n_linked_firms
  any_linked_402      1 if any linked firm had announced before t
  n_linked_402_officer  same, counting only links through officers (not outside directors)

  python interlock_features.py    -> data/out/features_interlock.npz (X, adsh, names)
"""

import io
import json
import zipfile
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from relabel import submissions

RAW = Path("data/raw/insider")
NAMES = ["n_insiders", "n_interlocked", "n_linked_firms", "n_linked_402", "frac_linked_402",
         "any_linked_402", "n_linked_402_officer"]


def load_links():
    """(person cik, firm cik, filing date, is_officer) for every insider filing."""
    parts = []
    for zp in sorted(RAW.glob("*_form345.zip")):
        z = zipfile.ZipFile(zp)
        sub = pd.read_csv(io.BytesIO(z.read("SUBMISSION.tsv")), sep="\t",
                          usecols=["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK"], dtype=str)
        own = pd.read_csv(io.BytesIO(z.read("REPORTINGOWNER.tsv")), sep="\t",
                          usecols=["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"], dtype=str)
        df = own.merge(sub, on="ACCESSION_NUMBER", how="inner")
        rel = df.RPTOWNER_RELATIONSHIP.fillna("")
        df = df[rel.str.contains("Officer|Director", case=False)]
        df["officer"] = rel.loc[df.index].str.contains("Officer", case=False)
        df["filed"] = pd.to_datetime(df.FILING_DATE, format="%d-%b-%Y", errors="coerce")
        df["person"] = df.RPTOWNERCIK.str.lstrip("0")
        df["firm"] = df.ISSUERCIK.str.lstrip("0")
        parts.append(df[["person", "firm", "filed", "officer"]].dropna())
        print(f"  {zp.stem}: {len(df):,} insider filings", flush=True)
    links = pd.concat(parts, ignore_index=True).drop_duplicates()
    return links


def main():
    links = load_links()
    print(f"\n{len(links):,} person-firm-date links, {links.person.nunique():,} people, {links.firm.nunique():,} firms")
    # firm -> sorted array of (date, person, officer); person -> sorted array of (date, firm)
    by_firm = {f: g.sort_values("filed") for f, g in links.groupby("firm")}
    by_person = {p: g.sort_values("filed") for p, g in links.groupby("person")}

    # Item 4.02 announcements by firm, sorted -- only those BEFORE the 10-K count.
    ann = defaultdict(list)
    for l in open("data/raw/restatements.jsonl", encoding="utf-8"):
        r = json.loads(l)
        for c in r.get("ciks") or []:
            if r.get("file_date"):
                ann[str(int(c))].append(datetime.strptime(r["file_date"], "%Y-%m-%d").date())
    for v in ann.values():
        v.sort()

    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y = z["adsh"], z["y"]
    subs = submissions()
    X = np.full((len(adsh), len(NAMES)), np.nan)
    for i, a in enumerate(adsh.tolist()):
        cik, filed, _ = subs.get(a, (None, None, None))
        if cik is None:
            continue
        g = by_firm.get(cik)
        if g is None:
            X[i] = [0, 0, 0, 0, np.nan, 0, 0]
            continue
        t = pd.Timestamp(filed)
        lo, hi = t - pd.Timedelta(days=1095), t
        w = g[(g.filed >= lo) & (g.filed < hi)]
        people = w.drop_duplicates("person")[["person", "officer"]]
        linked, linked_officer = {}, set()
        n_inter = 0
        for person, is_off in people.itertuples(index=False):
            h = by_person.get(person)
            if h is None:
                continue
            hw = h[(h.filed >= lo) & (h.filed < hi) & (h.firm != cik)]
            if hw.empty:
                continue
            n_inter += 1
            for f in hw.firm.unique():
                linked[f] = True
                if is_off:
                    linked_officer.add(f)
        fd = filed
        n402 = sum(1 for f in linked if ann.get(f) and bisect_left(ann[f], fd) > 0)
        n402_off = sum(1 for f in linked_officer if ann.get(f) and bisect_left(ann[f], fd) > 0)
        X[i] = [len(people), n_inter, len(linked), n402,
                n402 / len(linked) if linked else np.nan, float(n402 > 0), n402_off]
        if i % 5000 == 0:
            print(f"  {i:,}/{len(adsh):,}", flush=True)

    Path("data/out").mkdir(exist_ok=True)
    np.savez("data/out/features_interlock.npz", X=X, adsh=adsh, names=np.array(NAMES))
    from sklearn.metrics import roc_auc_score
    has = np.isfinite(X[:, 0])
    print(f"\ninterlock features for {has.sum():,} filings")
    lk = X[:, 2] > 0
    print(f"filings with any interlock: {lk.sum():,} ({100*lk.mean():.0f}%); restatement rate with {y[lk].mean():.2%} vs without {y[~lk].mean():.2%}")
    a4 = X[:, 5] == 1
    print(f"filings linked to a firm that had already announced an Item 4.02: {a4.sum():,} ({100*a4.mean():.1f}%); "
          f"restatement rate {y[a4].mean():.2%} vs {y[~a4].mean():.2%}")
    for k, n in enumerate(NAMES):
        v = X[:, k]; ok = np.isfinite(v)
        print(f"  {n:<22} coverage {100*ok.mean():3.0f}%  AUC alone {roc_auc_score(y[ok], v[ok]):.3f}")


if __name__ == "__main__":
    main()
