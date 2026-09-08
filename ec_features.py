#!/usr/bin/env python3
"""Rebuild every text-derived block from EDGAR-CRAWLER's sections, one parser for every row.

Reads data/raw/ec_items.jsonl (ec_extract.py) and writes:
  data/out/features_scrutiny_ec.npz   the ten flags + the five history columns carried over
  data/out/features_flags2_ec.npz     the thirty disclosure flags
  data/out/features_change_ec.npz     year-over-year drift and transitions, now pairable for every year
  data/raw/digest_ec.jsonl            regex-retrieved digests, one source for all filings

  python ec_features.py
"""

import json
import math
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

from change_features import NAMES as CHANGE_NAMES, SIGN_DATE, cos, summarize
from digest_text import digest
from flags2_features import NAMES as F2_NAMES, flags2
from relabel import submissions
from scrutiny_features import NAMES as S_NAMES, text_flags

EC = Path("data/raw/ec_items.jsonl")


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    alist = adsh.tolist(); pos = {a: i for i, a in enumerate(alist)}
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    subs = submissions()

    S_old = np.load("data/out/features_scrutiny.npz", allow_pickle=True)["X"]
    S = np.full((len(alist), len(S_NAMES)), np.nan); S[:, 10:] = S_old[:, 10:]   # history columns unchanged
    F = np.full((len(alist), len(F2_NAMES)), np.nan)
    summ = {}
    n = 0
    with open("data/raw/digest_ec.jsonl", "w", encoding="utf-8") as dg:
        for l in EC.open(encoding="utf-8"):
            r = json.loads(l)
            i = pos.get(r["adsh"])
            if i is None or r["status"] != "ok":
                continue
            if len(r.get("section_9A", "")) < 100 and len(r.get("section_8", "")) < 100:
                continue
            S[i, :10] = text_flags(r)
            F[i] = flags2(r, int(years[i]))
            summ[r["adsh"]] = summarize(r)
            dg.write(json.dumps({"adsh": r["adsh"], "source": "ec", "text": digest(r)}) + "\n")
            n += 1
    print(f"{n:,} filings with sections from one parser")
    np.savez("data/out/features_scrutiny_ec.npz", X=S, adsh=adsh, names=np.array(S_NAMES))
    np.savez("data/out/features_flags2_ec.npz", X=F, adsh=adsh, names=np.array(F2_NAMES))

    # change block: consecutive filings of the same firm, now every year pairable
    by_cik = defaultdict(list)
    for a in alist:
        cik, f, _ = subs.get(a, (None, None, None))
        if cik:
            by_cik[cik].append((f, a))
    for v in by_cik.values():
        v.sort()
    C = np.full((len(alist), len(CHANGE_NAMES)), np.nan)
    pairs = 0
    for cik, items in by_cik.items():
        for k in range(1, len(items)):
            (f0, a0), (f1, a1) = items[k - 1], items[k]
            if not (300 <= (f1 - f0).days <= 430):
                continue
            s0, s1 = summ.get(a0), summ.get(a1)
            if s0 is None or s1 is None:
                continue
            row = [cos(s0["bags"][kk], s1["bags"][kk]) for kk in ("section_1A", "section_7", "section_8", "section_9A")]
            row += [s1["len9a"] / s0["len9a"] if s0["len9a"] > 200 else math.nan,
                    s1["len7"] / s0["len7"] if s0["len7"] > 200 else math.nan,
                    float(s1["not_eff"] and not s0["not_eff"]), float(s1["not_eff"] and s0["not_eff"]),
                    float(s0["remed"] and s1["mw"]), float(s1["gc"] and not s0["gc"]),
                    float(len(s1["types"] - s0["types"])), float(s0["not_eff"])]
            lag = math.nan
            for m in s1["sign"]:
                try:
                    from datetime import datetime
                    d = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y").date()
                except ValueError:
                    continue
                days = (f1 - d).days
                if 0 <= days <= 200:
                    lag = float(days); break
            row.append(lag)
            C[pos[a1]] = row; pairs += 1
    np.savez("data/out/features_change_ec.npz", X=C, adsh=adsh, names=np.array(CHANGE_NAMES))
    print(f"{pairs:,} same-parser consecutive pairs (was 21,995 with two sources)")

    from sklearn.metrics import roc_auc_score
    for nm, X in (("scrutiny", S[:, :10]), ("flags2", F), ("change", C)):
        has = np.isfinite(X[:, 0])
        print(f"  {nm}: coverage {100*has.mean():.0f}%")
    for k in (0, 1, 3):
        v = S[:, k]; ok = np.isfinite(v)
        print(f"  {S_NAMES[k]:<20} AUC alone {roc_auc_score(y[ok], v[ok]):.3f}   rate when 1: {y[ok][v[ok]==1].mean():.2%} (n={int((v[ok]==1).sum()):,})")


if __name__ == "__main__":
    main()
