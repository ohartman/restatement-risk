#!/usr/bin/env python3
"""Download SEC Financial Statement Data Sets, one flat file per quarter.

The SEC publishes every XBRL number from every filing as quarterly flat files.
No API, no scraping, no rate-limit games -- the same lesson as last time: when a
bulk archive exists, take it instead of the endpoint.

Access is conditional on declaring who you are. The SEC returns 403 for a
browser user-agent and 200 for one carrying a name and contact address, so the
contact is read from SEC_CONTACT rather than hardcoded, and it stays out of the
repository.

  set SEC_CONTACT=Your Name you@example.com
  python fetch_sec.py 2014 2015 2016
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
RAW = Path("data/raw")
# The SEC asks for no more than ten requests a second; these are 120 MB files,
# so one every couple of seconds is already far inside that.
PAUSE = 2.0


def contact():
    c = os.environ.get("SEC_CONTACT", "").strip()
    if not c or "@" not in c:
        sys.exit("Set SEC_CONTACT to a name and email, e.g.\n"
                 '  set SEC_CONTACT=Jane Doe jane@example.com\n'
                 "The SEC returns 403 without one.")
    return c


def fetch(quarter, ua):
    out = RAW / f"{quarter}.zip"
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"  {quarter}  have it ({out.stat().st_size/1e6:.0f} MB)")
        return True
    url = f"{BASE}/{quarter}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": ua,
                                               "Accept-Encoding": "gzip, deflate"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r, out.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except Exception as exc:
        print(f"  {quarter}  FAILED ({type(exc).__name__}: {exc})")
        out.unlink(missing_ok=True)
        return False
    print(f"  {quarter}  {out.stat().st_size/1e6:.0f} MB")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("years", nargs="+", type=int)
    args = ap.parse_args()

    ua = contact()
    RAW.mkdir(parents=True, exist_ok=True)
    quarters = [f"{y}q{q}" for y in sorted(args.years) for q in (1, 2, 3, 4)]
    print(f"fetching {len(quarters)} quarters as {ua!r}")
    ok = 0
    for i, q in enumerate(quarters):
        if fetch(q, ua):
            ok += 1
        if i < len(quarters) - 1:
            time.sleep(PAUSE)
    print(f"\n{ok}/{len(quarters)} quarters on disk")


if __name__ == "__main__":
    main()
