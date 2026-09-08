#!/usr/bin/env python3
"""Does the F-score separate filings that were later restated from those that were not?

Every non-financial 10-K in the quarters on disk is scored, then labelled by
what happened afterwards: did the same company file an Item 4.02 non-reliance
8-K within the following window? The label comes from the future, so nothing in
the scoring can have been tuned to it.

The base rate is low -- a percent or two -- so accuracy is meaningless here and
is not reported. What matters is whether high scores are enriched for
restatements: AUC, and precision in the top slice of the ranking, which is what
anyone would actually act on.

  python evaluate.py --window-days 1095
"""

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fscore import read_quarter, variables, fscore, is_financial

RAW = Path("data/raw")


def load_restatements():
    """cik -> sorted list of Item 4.02 announcement dates."""
    by_cik = defaultdict(list)
    path = RAW / "restatements.jsonl"
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if not r.get("file_date"):
            continue
        d = datetime.strptime(r["file_date"], "%Y-%m-%d").date()
        for cik in r.get("ciks") or []:
            by_cik[str(int(cik))].append(d)
    for c in by_cik:
        by_cik[c].sort()
    return by_cik


def auc(scores, labels):
    """Rank-based AUC; ties share their averaged rank."""
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = [ranks[k] for k, (_, y) in enumerate(pairs) if y]
    n_pos, n_neg = len(pos), len(pairs) - len(pos)
    if not n_pos or not n_neg:
        return float("nan")
    return (sum(pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-days", type=int, default=1095,
                    help="how long after filing a restatement still counts")
    args = ap.parse_args()

    restate = load_restatements()
    print(f"companies with an Item 4.02 filing: {len(restate):,}")

    rows = []
    for zp in sorted(RAW.glob("*q?.zip")):
        subs, facts = read_quarter(zp)
        n = 0
        for adsh, s in subs.items():
            if s["form"] != "10-K" or is_financial(s["sic"]):
                continue
            d = facts.get(adsh)
            if not d or len(d) < 2:
                continue
            ds = sorted(d)
            v = variables(d[ds[-1]], d[ds[-2]])
            if v is None:
                continue
            f = fscore(v)[1]
            try:
                filed = datetime.strptime(s["filed"], "%Y%m%d").date()
            except ValueError:
                continue
            cik = str(int(s["cik"]))
            later = [x for x in restate.get(cik, [])
                     if 0 <= (x - filed).days <= args.window_days]
            rows.append({"adsh": adsh, "cik": cik, "name": s["name"],
                         "filed": filed.isoformat(), "f": f,
                         "label": 1 if later else 0})
            n += 1
        print(f"  {zp.stem}  {n:>4} scored 10-Ks", flush=True)

    pos = [r for r in rows if r["label"]]
    print(f"\nfilings scored     {len(rows):,}")
    print(f"later restated     {len(pos):,}  ({100*len(pos)/max(1,len(rows)):.2f}%)")
    if not pos:
        print("no positives in the window - widen it or fetch more restatement years")
        return

    fs_pos = [r["f"] for r in pos]
    fs_neg = [r["f"] for r in rows if not r["label"]]
    print(f"\nmedian F-score     restated {statistics.median(fs_pos):.2f}   "
          f"clean {statistics.median(fs_neg):.2f}")
    a = auc([r["f"] for r in rows], [r["label"] for r in rows])
    print(f"AUC                {a:.3f}   (0.50 is a coin flip)")

    ranked = sorted(rows, key=lambda r: -r["f"])
    base = len(pos) / len(rows)
    print(f"\nbase rate {base:.2%} - precision in the top slice of the ranking:")
    for k in (50, 100, 250, 500, 1000):
        if k > len(ranked):
            break
        hit = sum(r["label"] for r in ranked[:k])
        print(f"  top {k:>5}   {hit:>3} restated   precision {hit/k:.2%}   "
              f"lift {(hit/k)/base:.2f}x")

    Path("data/out").mkdir(parents=True, exist_ok=True)
    with open("data/out/scored.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote data/out/scored.jsonl ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
