#!/usr/bin/env python3
"""Extend the 10-K text flags to 2021-2023 from the full documents saved by fetch_mdna.py.

EDGAR-CORPUS ends in 2020, so scrutiny_features.py has Items 9A / 8 / 3 / 1A for
the training years but only two of the five test years. fetch_mdna.py kept every
fetched 10-K in full (data/raw/mdna/<adsh>.full.html.gz) for its sample: every
restated filing plus three clean ones per restated, 2014-2023. This cuts those
documents into the same items with the same regexes and fills the flags in.

The sample is label-stratified, so the flags' *presence* in 2021-23 tracks the
label. That is why scrutiny_test2.py evaluates only on rows that have flags --
a fixed set where every filing has them -- and never lets a model see whether
a flag is missing.

  python scrutiny_extend.py    -> data/out/features_scrutiny.npz updated in place
"""

import gzip
import html
import json
import re
from pathlib import Path

import numpy as np

from scrutiny_features import NAMES, text_flags
from text_features import TAG, WS

MDNA = Path("data/raw/mdna")
INVISIBLE = re.compile("[­​‌‍﻿]")
# Item headings; each section runs to the next item heading. Table-of-contents
# matches are short, so the longest candidate wins, as in fetch_mdna.py.
HEAD = {
    "section_1A": re.compile(r"item\s*1a(?![0-9])[^a-z0-9]{0,80}risk\s+factors", re.I),
    "section_3": re.compile(r"item\s*3(?![0-9a])[^a-z0-9]{0,80}legal\s+proceedings", re.I),
    "section_7": re.compile(r"item\s*7(?![0-9a])[^a-z0-9]{0,80}management", re.I),
    "section_8": re.compile(r"item\s*8(?![0-9])[^a-z0-9]{0,80}financial\s+statements", re.I),
    "section_9A": re.compile(r"item\s*9a(?:\s*\(t\))?(?![0-9])[^a-z0-9]{0,80}controls\s+and\s+procedures", re.I),
}
AUDIT_REPORT = re.compile(r"report of independent registered public accounting firm", re.I)
# A section ends only at a genuine heading -- the item number followed by that item's title --
# so a cross-reference like "see Item 10 of this report" inside Item 9A does not cut it short.
TITLES = {"1a": "risk factors", "1b": "unresolved staff", "2": "properties", "3": "legal proceedings",
          "4": "mine safety|submission of matters|removed and reserved|reserved", "5": "market for",
          "6": "selected financial|reserved", "7a": "quantitative and qualitative", "7": "management",
          "8": "financial statements", "9a": "controls and procedures", "9b": "other information",
          "9": "changes in and disagreements", "10": "directors", "11": "executive compensation",
          "12": "security ownership", "13": "certain relationships", "14": "principal account",
          "15": "exhibits"}
NEXT = re.compile("item\s*(?:" + "|".join(f"{k}(?![0-9])[^a-z0-9]{{0,80}}(?:{v})" for k, v in TITLES.items()) + ")", re.I)


# Cross-references look like headings -- "as set forth in Item 8, Financial Statements" -- and were
# cutting Item 9A off mid-sentence. A heading is a NEXT match that is not introduced by a
# referring word in the 40 characters before it.
CUE = re.compile(r"(see|refer to|referred to in|in|under|set forth in|included in|described in|discussed in|"
                 r"contained in|of|to|and|with|within|per|at)\s*[\"“(]?\s*$", re.I)


def next_heading(text, start):
    for m in NEXT.finditer(text, start):
        before = text[max(0, m.start() - 40): m.start()]
        if CUE.search(before):
            continue
        return m.start()
    return None


def sections(doc):
    text = WS.sub(" ", INVISIBLE.sub("", html.unescape(TAG.sub(" ", doc))))
    out = {}
    for key, pat in HEAD.items():
        best = ""
        for m in pat.finditer(text):
            e = next_heading(text, m.end() + 300)
            seg = text[m.start(): e] if e else text[m.start(): m.start() + 300_000]
            if len(seg) > len(best):
                best = seg
        out[key] = best
    # Many 10-Ks put the financial statements and the auditor's report in "F-pages" after
    # Item 15, outside any item heading. If the Item 8 cut has no audit report in it, append
    # everything from the first audit-report heading onward so the flags can see it.
    if not AUDIT_REPORT.search(out.get("section_8", "")):
        m = AUDIT_REPORT.search(text)
        if m:
            out["section_8"] = (out.get("section_8", "") + " " + text[m.start(): m.start() + 25_000]).strip()
    return out


def main():
    z = np.load("data/out/features_scrutiny.npz", allow_pickle=True)
    X, adsh, names = z["X"], z["adsh"], list(z["names"])
    assert names == NAMES
    pos = {a: i for i, a in enumerate(adsh.tolist())}
    text_cols = list(range(10))
    import sys
    test_only = "--test-only" in sys.argv
    all_rows = "--all" in sys.argv          # recut every row from saved documents: one parser everywhere
    filed_all = np.load("data/out/features.npz", allow_pickle=True)["filed"].astype("datetime64[D]")
    if test_only:
        print("filling test-year rows (2019+) only; training rows keep corpus-sourced flags")
    filled = tried = 0
    rows_iter = ([{"adsh": p.name.replace(".full.html.gz", ""), "status": "ok"} for p in MDNA.glob("*.full.html.gz")]
                 if all_rows else [json.loads(l) for l in (MDNA / "index.jsonl").open(encoding="utf-8")])
    for r in rows_iter:
        i = pos.get(r["adsh"])
        if i is None or r["status"] != "ok":
            continue
        if not all_rows and np.isfinite(X[i, 0]):
            continue
        if test_only and filed_all[i] < np.datetime64("2019-01-01"):
            continue
        p = MDNA / f"{r['adsh']}.full.html.gz"
        if not p.exists():
            continue
        tried += 1
        with gzip.open(p, "rt", encoding="utf-8") as g:
            sec = sections(g.read())
        if len(sec["section_9A"]) < 200 and len(sec["section_8"]) < 200:
            continue
        X[i, text_cols] = text_flags(sec)
        filled += 1
        if tried % 500 == 0:
            print(f"  {tried:,} documents read, {filled:,} filled", flush=True)
    np.savez("data/out/features_scrutiny.npz", X=X, adsh=adsh, names=np.array(names))
    y = np.load("data/out/features.npz", allow_pickle=True)["y"]
    filed = np.load("data/out/features.npz", allow_pickle=True)["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    has = np.isfinite(X[:, 0])
    print(f"\n{filled:,} filings filled from saved documents ({tried:,} read); text flags now cover {has.sum():,} filings")
    for yr in range(2019, 2024):
        m = years == yr
        print(f"  {yr}: {int((m & has).sum()):,} of {int(m.sum()):,} test filings have flags; restated among them {int(y[m & has].sum())}")


if __name__ == "__main__":
    main()
