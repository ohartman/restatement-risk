#!/usr/bin/env python3
"""The v3 flag block rebuilt from a blank slate: every row cut from its own saved 10-K
(matched by accession number) with our extractor, nothing inherited from the corpus.
Rows without a document, or whose Items 9A and 8 both come out empty, stay NaN.
History columns (prior 4.02s etc.) are copied from v3; they never touched the corpus.

  python scrutiny_v3_clean.py            -> data/out/features_scrutiny_v3c.npz
"""
import gzip
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scrutiny_features import NAMES, text_flags
from scrutiny_extend import sections

MDNA = Path("data/raw/mdna")


def one(adsh):
    p = MDNA / f"{adsh}.full.html.gz"
    if not p.exists():
        return adsh, None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as g:
            sec = sections(g.read())
    except Exception:
        return adsh, None
    if len(sec["section_9A"]) < 200 and len(sec["section_8"]) < 200:
        return adsh, None
    return adsh, text_flags(sec)


def main():
    z = np.load("data/out/features.npz", allow_pickle=True); adsh = z["adsh"]
    old = np.load("data/out/features_scrutiny_v3.npz", allow_pickle=True)
    assert (old["adsh"] == adsh).all()
    X = np.full(old["X"].shape, np.nan); X[:, 10:] = old["X"][:, 10:]
    pos = {a: i for i, a in enumerate(adsh.tolist())}
    n = 0
    with Pool(6) as pool:
        for k, (a, fl) in enumerate(pool.imap_unordered(one, adsh.tolist(), chunksize=50), 1):
            if fl is not None:
                X[pos[a], :10] = fl; n += 1
            if k % 2000 == 0:
                print(f"  {k:,} read, {n:,} filled", flush=True)
    np.savez("data/out/features_scrutiny_v3c.npz", X=X, adsh=adsh, names=np.array(NAMES))
    print(f"{n:,} of {len(adsh):,} filings have flags from their own document; saved features_scrutiny_v3c.npz")


if __name__ == "__main__":
    main()
