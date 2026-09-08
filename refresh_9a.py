#!/usr/bin/env python3
"""Recompute the ten text flags for every 2019+ filing with a saved full document, using the
improved section cutter (F-pages included), and write them into features_scrutiny.npz for
TEST rows only. Training rows keep their corpus-sourced flags."""
import gzip
from pathlib import Path
import numpy as np
from scrutiny_extend import sections
from scrutiny_features import NAMES, text_flags
z = np.load("data/out/features_scrutiny.npz", allow_pickle=True)
X, adsh, names = z["X"], z["adsh"], z["names"]
filed = np.load("data/out/features.npz", allow_pickle=True)["filed"].astype("datetime64[D]")
pos = {a: i for i, a in enumerate(adsh.tolist())}
n = 0
for p in Path("data/raw/mdna").glob("*.full.html.gz"):
    a = p.name.replace(".full.html.gz", "")
    i = pos.get(a)
    if i is None or filed[i] < np.datetime64("2021-01-01"):     # 2019-20 test rows keep corpus flags
        continue
    with gzip.open(p, "rt", encoding="utf-8") as g:
        sec = sections(g.read())
    if len(sec.get("section_9A", "")) < 200 and len(sec.get("section_8", "")) < 200:
        continue
    X[i, :10] = text_flags(sec); n += 1
    if n % 2000 == 0:
        print(f"  {n:,} refreshed", flush=True)
np.savez("data/out/features_scrutiny.npz", X=X, adsh=adsh, names=names)
print(f"refreshed flags for {n:,} 2021-23 test filings with the F-pages cutter")
