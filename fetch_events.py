#!/usr/bin/env python3
"""Harvest the 8-K events that precede restatements: auditor changes and CFO turnover.

A restatement is a misstatement that somebody found. The financial statements
say something about whether there was a misstatement; they say nothing about
whether anyone was looking. Two public events do:

  Item 4.01  Changes in Registrant's Certifying Accountant -- the auditor
             resigned or was dismissed. A new auditor re-examines the books.
  Item 5.02  Departure/appointment of officers -- restricted here to filings
             that mention the chief financial officer, so it reads as CFO
             turnover. A new CFO inherits the old one's judgment calls.

Both come from the same EDGAR full-text search that supplied the labels, and
both are filtered on the structured `items` field, not the phrase. Every event
is stored with its filing date so that features can use only the events that
happened before each 10-K was filed.

  set SEC_CONTACT=Your Name you@example.com
  python fetch_events.py 4.01 2011 2023
  python fetch_events.py 5.02 2011 2023
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from calendar import monthrange
from pathlib import Path

FTS = "https://efts.sec.gov/LATEST/search-index"
PAUSE = 0.25
QUERY = {"4.01": '"Item 4.01"', "5.02": '"Chief Financial Officer" departure'}


def contact():
    c = os.environ.get("SEC_CONTACT", "").strip()
    if not c or "@" not in c:
        sys.exit("Set SEC_CONTACT to a name and email; the SEC returns 403 without one.")
    return c


def search(ua, query, start, end, offset):
    params = urllib.parse.urlencode({"q": query, "forms": "8-K",
                                     "startdt": start, "enddt": end, "from": offset})
    req = urllib.request.Request(f"{FTS}?{params}",
                                 headers={"User-Agent": ua, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def month_windows(y0, y1):
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            yield f"{y}-{m:02d}-01", f"{y}-{m:02d}-{monthrange(y, m)[1]:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("item", choices=sorted(QUERY))
    ap.add_argument("start_year", type=int)
    ap.add_argument("end_year", type=int)
    args = ap.parse_args()
    ua = contact()
    out = Path(f"data/raw/events_{args.item.replace('.', '')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    seen, done_months = set(), set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                seen.add(r["adsh"]); done_months.add(r["file_date"][:7])
            except Exception:
                pass
        print(f"resuming: {len(seen)} events already collected")

    kept, started = 0, time.time()
    with out.open("a", encoding="utf-8") as fh:
        for start, end in month_windows(args.start_year, args.end_year):
            # A month whose last day is already on disk was finished earlier.
            if start[:7] in done_months and start[:7] != max(done_months):
                continue
            offset, total, month_kept = 0, None, 0
            while True:
                try:
                    d = search(ua, QUERY[args.item], start, end, offset)
                except Exception as exc:
                    print(f"  {start[:7]}  search failed ({type(exc).__name__}: {exc})", flush=True)
                    time.sleep(5)
                    break
                hits = d.get("hits", {}).get("hits", [])
                if total is None:
                    total = d["hits"]["total"]["value"]
                for h in hits:
                    src = h.get("_source", {})
                    if args.item not in (src.get("items") or []):
                        continue
                    adsh = src.get("adsh") or h["_id"].split(":")[0]
                    if adsh in seen:
                        continue
                    seen.add(adsh)
                    fh.write(json.dumps({
                        "adsh": adsh,
                        "ciks": [c.lstrip("0") for c in (src.get("ciks") or [])],
                        "file_date": src.get("file_date"),
                        "items": src.get("items"),
                        "sic": (src.get("sics") or [None])[0],
                    }) + "\n")
                    kept += 1; month_kept += 1
                if not hits:
                    break
                offset += len(hits)
                if offset >= min(total or 0, 9990):
                    break
                time.sleep(PAUSE)
            fh.flush()
            capped = " (CAPPED)" if (total or 0) > 9990 else ""
            print(f"  {start[:7]}  {total or 0:>5} matches -> {month_kept:>4} with item {args.item}"
                  f"{capped}   [{(time.time()-started)/60:.0f} min]", flush=True)
            time.sleep(PAUSE)
    print(f"\n{kept} new item-{args.item} events -> {out}")


if __name__ == "__main__":
    main()
