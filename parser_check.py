#!/usr/bin/env python3
"""Do our parser and EDGAR-CORPUS produce the same flags for the same filing?

The headline mixes sources: training flags from the corpus, 2021-23 test flags
from our own section cutter. If the two disagree on "controls not effective"
for the same document, the flags are dulled on 60% of the test set and the
headline understates -- or, if they disagree in a label-correlated way,
overstates. Filings from 2019-2020 that are in both the corpus and our fetched
documents settle it: compute the ten flags both ways and compare.

  python parser_check.py
"""

import gzip
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scrutiny_extend import sections
from scrutiny_features import NAMES, text_flags

MDNA = Path("data/raw/mdna")


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"].tolist(), z["y"], z["filed"].astype("datetime64[D]")
    S = np.load("data/out/features_scrutiny.npz", allow_pickle=True)["X"]
    pos = {a: i for i, a in enumerate(adsh)}
    ours, theirs, labels = [], [], []
    for p in MDNA.glob("*.full.html.gz"):
        a = p.name.replace(".full.html.gz", "")
        i = pos.get(a)
        if i is None or not np.isfinite(S[i, 0]) or not (np.datetime64("2019-01-01") <= filed[i] < np.datetime64("2021-01-01")):
            continue
        with gzip.open(p, "rt", encoding="utf-8") as g:
            sec = sections(g.read())
        if len(sec["section_9A"]) < 200 and len(sec["section_8"]) < 200:
            continue
        ours.append(text_flags(sec)[:10]); theirs.append(S[i, :10]); labels.append(y[i])
        if len(ours) >= 1500:
            break
    ours, theirs, labels = np.array(ours), np.array(theirs), np.array(labels)
    print(f"{len(ours):,} filings from 2019-2020 with flags from both parsers ({int(labels.sum())} restated)\n")
    print(f"{'flag':<20}{'agree':>8}{'ours=1':>8}{'corpus=1':>10}{'AUC ours':>10}{'AUC corpus':>12}")
    for k, n in enumerate(NAMES[:10]):
        a, b = ours[:, k], theirs[:, k]
        binary = set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}
        agree = np.mean(a == b) if binary else np.corrcoef(a, b)[0, 1]
        line = f"{n:<20}{agree:>8.3f}"
        if binary:
            line += f"{a.mean():>8.3f}{b.mean():>10.3f}"
        else:
            line += f"{'':>18}"
        if 0 < labels.sum() < len(labels) and len(np.unique(a)) > 1 and len(np.unique(b)) > 1:
            line += f"{roc_auc_score(labels, a):>10.3f}{roc_auc_score(labels, b):>12.3f}"
        print(line)
    print("\n(agree = share of filings where the two parsers give the same value; correlation for counts/lengths)")


if __name__ == "__main__":
    main()
