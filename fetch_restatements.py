#!/usr/bin/env python3
"""Harvest Item 4.02 non-reliance 8-Ks -- the restatement labels.

When a company concludes that previously issued financial statements can no
longer be relied upon, it must file an 8-K under Item 4.02. That filing is the
label: it says, in the company's own words and at a known date, that an earlier
filing was wrong.

EDGAR full-text search returns a structured `items` list on every hit, so the
match is on the actual item number rather than on the phrase "Item 4.02"
appearing somewhere in the text -- a document can discuss the item without being
one.

These labels arrive from the future relative to the filings they condemn, which
is what makes the evaluation honest: nothing about a 2016 filing can be tuned to
fit a label that did not exist until 2018.

  set SEC_CONTACT=Your Name you@example.com
  python fetch_restatements.py 2016 2017 2018 2019
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

FTS = "https://efts.sec.gov/LATEST/search-index"
OUT = Path("data/raw/restatements.jsonl")
PAGE = 10          # the API's page size
PAUSE = 0.3        # well inside the SEC's ten-per-second guidance


def contact():
    c = os.environ.get("SEC_CONTACT", "").strip()
    if not c or "@" not in c:
        sys.exit("Set SEC_CONTACT to a name and email; the SEC returns 403 without one.")
    return c


def search(ua, start, end, offset):
    params = urllib.parse.urlencode({
        "q": '"Item 4.02"', "forms": "8-K",
        "startdt": start, "enddt": end, "from": offset})
    req = urllib.request.Request(f"{FTS}?{params}",
                                 headers={"User-Agent": ua, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def quarter_windows(years):
    for y in sorted(years):
        for a, b in (("01-01", "03-31"), ("04-01", "06-30"),
                     ("07-01", "09-30"), ("10-01", "12-31")):
            yield f"{y}-{a}", f"{y}-{b}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("years", nargs="+", type=int)
    args = ap.parse_args()
    ua = contact()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            try:
                seen.add(json.loads(line)["adsh"])
            except Exception:
                pass
        print(f"resuming: {len(seen)} already collected")

    kept = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for start, end in quarter_windows(args.years):
            offset, total, quarter_kept = 0, None, 0
            while True:
                try:
                    d = search(ua, start, end, offset)
                except Exception as exc:
                    print(f"  {start}  search failed ({type(exc).__name__}: {exc})")
                    break
                hits = d.get("hits", {}).get("hits", [])
                if total is None:
                    total = d["hits"]["total"]["value"]
                for h in hits:
                    src = h.get("_source", {})
                    # The structured item list, not a phrase match.
                    if "4.02" not in (src.get("items") or []):
                        continue
                    adsh = src.get("adsh") or h["_id"].split(":")[0]
                    if adsh in seen:
                        continue
                    seen.add(adsh)
                    ciks = [c.lstrip("0") for c in (src.get("ciks") or [])]
                    fh.write(json.dumps({
                        "adsh": adsh, "ciks": ciks,
                        "file_date": src.get("file_date"),
                        "period_ending": src.get("period_ending"),
                        "name": (src.get("display_names") or [None])[0],
                        "sic": (src.get("sics") or [None])[0],
                        "items": src.get("items"),
                    }) + "\n")
                    kept += 1
                    quarter_kept += 1
                offset += PAGE
                if not hits or offset >= min(total or 0, 9990):
                    break
                time.sleep(PAUSE)
            fh.flush()
            print(f"  {start[:7]}  {total or 0:>4} matches -> {quarter_kept:>3} with item 4.02",
                  flush=True)
            time.sleep(PAUSE)

    print(f"\n{kept} new restatement announcements -> {OUT}")


if __name__ == "__main__":
    main()
