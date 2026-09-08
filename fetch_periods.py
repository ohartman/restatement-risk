#!/usr/bin/env python3
"""Read each Item 4.02 filing and record which periods it actually condemns.

The current label is loose: a 10-K is marked "restated" if the same company
files any Item 4.02 within three years. But an 8-K filed in 2021 that withdraws
reliance on the 2020 statements says nothing about the 2018 10-K, which was
fine -- and the loose label calls it positive anyway. Noise like that is a hard
ceiling on AUC that no model can train through.

Item 4.02 filings name the periods they cover, in reasonably formulaic
language: "the fiscal years ended December 31, 2019 and 2020", "the quarterly
periods ended March 31 and June 30, 2021", "the Annual Report on Form 10-K for
the year ended ...". This fetches each 8-K's main document, isolates the Item
4.02 passage, and pulls out the fiscal years it mentions. Where nothing can be
parsed the filing is kept with an empty list, so the loose label can still be
used as a fallback and the parse rate is visible rather than hidden.

  set SEC_CONTACT=Your Name you@example.com
  python fetch_periods.py
"""

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from text_features import get, TAG, WS

IN = Path("data/raw/restatements.jsonl")
OUT = Path("data/raw/restatement_periods.jsonl")
PAUSE = 0.15

MONTHS = ("January|February|March|April|May|June|July|August|September|October|"
          "November|December|Jan\\.?|Feb\\.?|Mar\\.?|Apr\\.?|Jun\\.?|Jul\\.?|Aug\\.?|"
          "Sep\\.?|Sept\\.?|Oct\\.?|Nov\\.?|Dec\\.?")
# "December 31, 2019", "December 31 2019", "Dec. 31, 2019"
DATE = re.compile(rf"(?:{MONTHS})\s+\d{{1,2}},?\s+(20[0-2]\d)", re.I)
# "fiscal 2019", "fiscal year 2019", "FY2019", "FY 2019"
FISCAL = re.compile(r"\b(?:fiscal(?:\s+year)?|FY)\s*(20[0-2]\d)\b", re.I)
# "the years ended December 31, 2019 and 2020" -- the trailing bare year
TRAILING = re.compile(r"(20[0-2]\d)\s*(?:,|and|&)\s*(20[0-2]\d)\b")
# 10-K / 10-Q references usually sit right next to the period they cover
FORM_REF = re.compile(r"Form\s+10-[KQ](?:/A)?\b", re.I)
ITEM = re.compile(r"Item\s*4\.02", re.I)
NEXT_ITEM = re.compile(r"Item\s*(?:[5-9]\.\d\d|4\.0[3-9])", re.I)


def contact():
    c = os.environ.get("SEC_CONTACT", "").strip()
    if not c or "@" not in c:
        sys.exit("Set SEC_CONTACT to a name and email; the SEC returns 403 without one.")
    return c


XBRL_ARTIFACT = re.compile(r"^R\d+\.htm$|FilingSummary|Financial_Report|MetaLinks|"
                           r"\.xsd$|_cal\.xml|_def\.xml|_lab\.xml|_pre\.xml|_htm\.xml", re.I)
EXHIBIT = re.compile(r"(^|[-_])ex[-_]?\d|exhibit|graphic|logo|image|-index", re.I)


def main_document(cik, adsh, ua):
    """The 8-K body itself.

    Since inline XBRL became mandatory for 8-Ks, a filing directory also holds
    R1.htm, R2.htm and friends -- rendered XBRL pages that can be larger than the
    8-K and contain only cover-page data. Picking "the biggest .htm" chose those
    for nearly every filing from 2021 on, which is how half the announcements
    parsed to nothing but their own filing date. Prefer a file whose name says
    8-K, never take an XBRL artifact, and only then fall back to size.
    """
    nod = adsh.replace("-", "")
    idx = f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/index.json"
    d = json.loads(get(idx, ua).decode("utf-8", "replace"))
    items = [(i["name"], int(i.get("size") or 0)) for i in d["directory"]["item"]]
    htm = [(n, sz) for n, sz in items
           if n.lower().endswith((".htm", ".html"))
           and not XBRL_ARTIFACT.search(n) and not EXHIBIT.search(n)]
    named = [(n, sz) for n, sz in htm if re.search(r"8-?k", n, re.I)]
    pool = named or htm
    if pool:
        name = max(pool, key=lambda x: x[1])[0]
    else:
        txt = [n for n, _ in items if n.lower().endswith(".txt")
               and not n.lower().endswith("-index.txt")]
        if not txt:
            return None
        name = txt[0]
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/{name}"


def passage(html):
    """Text of the Item 4.02 section, or the whole document if it can't be cut."""
    text = TAG.sub(" ", html)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&#160;", " ").replace("&#8217;", "'"))
    text = WS.sub(" ", text)
    m = ITEM.search(text)
    if not m:
        return text, False
    start = m.start()
    n = NEXT_ITEM.search(text, m.end())
    end = n.start() if n else min(len(text), start + 6000)
    return text[start:end], True


def years_in(passage_text):
    """Fiscal years the passage names, 2005-2026 only."""
    ys = set()
    for m in DATE.finditer(passage_text):
        ys.add(int(m.group(1)))
    for m in FISCAL.finditer(passage_text):
        ys.add(int(m.group(1)))
    for m in TRAILING.finditer(passage_text):
        ys.add(int(m.group(1))); ys.add(int(m.group(2)))
    return sorted(y for y in ys if 2005 <= y <= 2026)


def main():
    ua = contact()
    rows = [json.loads(l) for l in IN.open(encoding="utf-8")]
    done = set()
    if OUT.exists():
        for l in OUT.open(encoding="utf-8"):
            try:
                done.add(json.loads(l)["adsh"])
            except Exception:
                pass
        print(f"resuming: {len(done)} already parsed")

    ok = parsed = fail = 0
    reasons = Counter()
    started = time.time()
    with OUT.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            if r["adsh"] in done:
                continue
            cik = (r.get("ciks") or [None])[0]
            if not cik:
                fail += 1; reasons["no cik"] += 1
                continue
            try:
                url = main_document(cik, r["adsh"], ua)
                time.sleep(PAUSE)
                if not url:
                    fail += 1; reasons["no document"] += 1
                    continue
                html = get(url, ua).decode("utf-8", "replace")
                text, cut = passage(html)
                ys = years_in(text)
                # Flag the filing type words, which help interpret the years.
                mentions_10k = bool(re.search(r"10-K", text))
                mentions_10q = bool(re.search(r"10-Q", text))
                fh.write(json.dumps({
                    "adsh": r["adsh"], "cik": str(int(cik)),
                    "file_date": r.get("file_date"),
                    "item_section_found": cut,
                    "years": ys, "mentions_10k": mentions_10k,
                    "mentions_10q": mentions_10q,
                    "snippet": text[:600],
                }) + "\n")
                ok += 1
                if ys:
                    parsed += 1
            except Exception as exc:
                fail += 1
                reasons[type(exc).__name__] += 1
                if fail <= 3:
                    print(f"    failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(PAUSE)
            if i % 100 == 0:
                fh.flush()
                rate = i / (time.time() - started)
                print(f"  {i}/{len(rows)}  fetched {ok}  with years {parsed}  "
                      f"failed {fail}  ({rate:.1f}/s, {(len(rows)-i)/max(rate,.01)/60:.0f} min left)",
                      flush=True)

    print(f"\n{ok} fetched, {parsed} with parseable years, {fail} failed -> {OUT}")
    if reasons:
        print("failure reasons:", dict(reasons))


if __name__ == "__main__":
    main()
