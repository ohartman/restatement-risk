#!/usr/bin/env python3
"""What changed since last year: text drift and flag transitions.

Companies leave boilerplate alone until something forces an edit (Cohen, Malloy
& Nguyen 2020). For each 10-K with a prior-year 10-K from the same source, this
measures how much each section moved, and how the disclosure flags moved:

  sim_1A, sim_7, sim_8, sim_9A   cosine similarity of the section's word counts to last year's
  len_ratio_9A, len_ratio_7      this year's length / last year's
  newly_not_effective            controls not effective now, effective last year
  chronic_not_effective          not effective both years
  remediated_then_weak           remediation claimed last year, material weakness again this year
  new_going_concern              going-concern doubt now, none last year
  new_mw_types                   weakness types named now that were not named last year
  prev_not_effective             last year's flag, as its own column
  audit_report_lag               days from fiscal year-end to the date under the auditor's signature

Pairs use one parser per pair: corpus-corpus for 2014-2020, our fetched documents
for 2022 vs 2021 and 2023 vs 2022. 2021 rows (whose prior year is corpus-only)
stay NaN -- label-independent, since every 2021 filing lacks it.

  python change_features.py   -> data/out/features_change.npz (X, adsh, names)
"""

import gzip
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from flags2_features import WEAK_TYPES
from relabel import submissions
from scrutiny_extend import sections as cut_sections
from scrutiny_features import GC, MW, NOT_EFF, REMED

CORPUS = Path("data/raw/edgar_corpus")
MDNA = Path("data/raw/mdna")
ACC = re.compile(r"\d{10}-\d{2}-\d{6}")
NAMES = ["sim_1A", "sim_7", "sim_8", "sim_9A", "len_ratio_9A", "len_ratio_7", "newly_not_effective",
         "chronic_not_effective", "remediated_then_weak", "new_going_concern", "new_mw_types",
         "prev_not_effective", "audit_report_lag"]
WORD = re.compile(r"[a-z]{3,}")
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
SIGN_DATE = re.compile(rf"({MONTHS})\s+(\d{{1,2}}),\s+(20\d\d)")


def bag(text, n=20000):
    return Counter(WORD.findall(text.lower()[:200000]))


def cos(a, b):
    if not a or not b:
        return math.nan
    num = sum(v * b.get(k, 0) for k, v in a.items())
    return num / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def summarize(sec):
    """What we keep per filing: bags for four sections, lengths, and the flags that can transition."""
    s9a = sec.get("section_9A") or ""; s8 = sec.get("section_8") or ""; s7 = sec.get("section_7") or ""
    return {
        "bags": {k: bag(sec.get(k) or "") for k in ("section_1A", "section_7", "section_8", "section_9A")},
        "len9a": len(s9a), "len7": len(s7),
        "not_eff": bool(NOT_EFF.search(s9a)), "mw": bool(MW.search(s9a)), "remed": bool(REMED.search(s9a)),
        "gc": bool(GC.search(s8) or GC.search(s7)),
        "types": {k for k, p in WEAK_TYPES.items() if re.search(p, s9a, re.I)},
        "sign": [m for m in SIGN_DATE.finditer(s8[-30000:])],
    }


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh = z["adsh"].tolist()
    subs = submissions()
    want = set(adsh); pos = {a: i for i, a in enumerate(adsh)}
    by_cik_year = defaultdict(list)
    for a in adsh:
        cik, f, _ = subs.get(a, (None, None, None))
        if cik:
            by_cik_year[(cik, f.year)].append(a)

    summ, source = {}, {}
    for f in sorted(CORPUS.glob("*/*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = ACC.search(r.get("filename", ""))
                targets = [m.group(0)] if (m and m.group(0) in want) else []
                if not targets:
                    cik = str(int(r["cik"])) if str(r.get("cik", "")).isdigit() else None
                    try:
                        yr = int(r.get("year"))
                    except (TypeError, ValueError):
                        yr = None
                    targets = [a for a in by_cik_year.get((cik, yr), []) if a not in summ] if cik and yr else []
                if targets and (r.get("section_9A") or r.get("section_8")):
                    s = summarize(r)
                    for a in targets:
                        summ.setdefault(a, s); source.setdefault(a, "corpus")
        print(f"  {f.parent.name}/{f.name}: {len(summ):,}", flush=True)
    n_fetch = 0
    for p in MDNA.glob("*.full.html.gz"):
        a = p.name.replace(".full.html.gz", "")
        if a in pos and a not in summ:
            with gzip.open(p, "rt", encoding="utf-8") as g:
                sec = cut_sections(g.read())
            if len(sec.get("section_9A", "")) > 200 or len(sec.get("section_8", "")) > 200:
                summ[a] = summarize(sec); source[a] = "fetch"; n_fetch += 1
    print(f"summaries: {len(summ):,} ({n_fetch:,} from fetched documents)")

    # prior filing of the same firm, by fiscal period
    by_cik = defaultdict(list)
    for a in adsh:
        cik, filed, pyear = subs.get(a, (None, None, None))
        if cik:
            by_cik[cik].append((filed, a))
    for v in by_cik.values():
        v.sort()
    X = np.full((len(adsh), len(NAMES)), np.nan)
    n_pairs = 0
    for cik, items in by_cik.items():
        for k in range(1, len(items)):
            (f_prev, a_prev), (f_cur, a_cur) = items[k - 1], items[k]
            if not (300 <= (f_cur - f_prev).days <= 430):
                continue
            s0, s1 = summ.get(a_prev), summ.get(a_cur)
            if s0 is None or s1 is None or source[a_prev] != source[a_cur]:
                continue
            i = pos[a_cur]
            row = [cos(s0["bags"][k_], s1["bags"][k_]) for k_ in ("section_1A", "section_7", "section_8", "section_9A")]
            row += [s1["len9a"] / s0["len9a"] if s0["len9a"] > 200 else math.nan,
                    s1["len7"] / s0["len7"] if s0["len7"] > 200 else math.nan,
                    float(s1["not_eff"] and not s0["not_eff"]), float(s1["not_eff"] and s0["not_eff"]),
                    float(s0["remed"] and s1["mw"]), float(s1["gc"] and not s0["gc"]),
                    float(len(s1["types"] - s0["types"])), float(s0["not_eff"])]
            lag = math.nan
            _, _, pyear = subs[a_cur]
            for m in s1["sign"]:
                try:
                    d = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y").date()
                except ValueError:
                    continue
                days = (f_cur - d).days
                if 0 <= days <= 200:                       # signed before filing, within reason
                    lag = float((d - f_cur).days + (f_cur - subs[a_cur][1]).days) if False else float(days)
                    break
            row.append(lag)
            X[i] = row; n_pairs += 1
    np.savez("data/out/features_change.npz", X=X, adsh=np.array(adsh), names=np.array(NAMES))
    y = z["y"]
    from sklearn.metrics import roc_auc_score
    print(f"\n{n_pairs:,} filings with a same-source prior year")
    for k, nm in enumerate(NAMES):
        v = X[:, k]; ok = np.isfinite(v)
        if ok.sum() and 0 < y[ok].sum() < ok.sum() and len(np.unique(v[ok])) > 1:
            line = f"  {nm:<22} coverage {100*ok.mean():3.0f}%  AUC alone {roc_auc_score(y[ok], v[ok]):.3f}"
            if set(np.unique(v[ok])) <= {0.0, 1.0}:
                line += f"   rate when 1: {y[ok][v[ok]==1].mean():6.2%} (n={int((v[ok]==1).sum()):>6,})   when 0: {y[ok][v[ok]==0].mean():.2%}"
            print(line)


if __name__ == "__main__":
    main()
