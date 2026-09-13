#!/usr/bin/env python3
"""Every federal docket with nature of suit 850 (securities, commodities, exchange) since 2014,
from CourtListener's free search API: case name, court, filing date, docket number.
Securities class actions name the company as defendant ("Smith v. Acme Corp."); SEC
enforcement actions and suits against individuals are in the same code and are kept, tagged.

  python signals/fetch_courtlistener.py   -> data/raw/courtlistener/nos850.jsonl
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/raw/courtlistener"; OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://www.courtlistener.com/api/rest/v4/search/"


def get(url, token=None):
    h = {"User-Agent": "Mozilla/5.0 (research; restatement-risk)"}
    if token:
        h["Authorization"] = f"Token {token}"
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    dst = OUT / "nos850.jsonl"
    seen = set()
    if dst.exists():
        for l in dst.open(encoding="utf-8"):
            seen.add(json.loads(l)["docket_id"])
    import datetime as dt
    windows = []
    import os, sys
    y0, y1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2014, 2022)
    start = dt.date(y0, 1, 1)
    while start < dt.date(y1 + 1, 1, 1):
        end = start + dt.timedelta(days=31)                  # monthly windows: short cursor chains survive the throttling better
        end = dt.date(end.year, end.month, 1)
        windows.append((start, end)); start = end
    PAUSE = float(os.environ.get("CL_PAUSE", "3"))
    token = os.environ.get("CL_TOKEN")                        # a free CourtListener account lifts the anonymous throttle
    n = 0; t0 = time.time()
    with dst.open("a", encoding="utf-8") as fh:
      for w0, w1 in windows:
        url = BASE + "?" + urllib.parse.urlencode({"type": "r", "q": 'suitNature:"850"', "filed_after": w0.isoformat(), "filed_before": (w1 - dt.timedelta(days=1)).isoformat(), "order_by": "dateFiled desc"})
        print(f"window {w0} to {w1}", flush=True)
        while url:
            for attempt in range(12):
                try:
                    d = get(url, token); break
                except Exception as exc:
                    print(f"  retry {attempt + 1}: {str(exc)[:60]}", flush=True); time.sleep(60 * (attempt + 1))
            else:
                print("giving up at", url[:80]); break
            for r in d["results"]:
                if r["docket_id"] in seen:
                    continue
                seen.add(r["docket_id"]); n += 1
                fh.write(json.dumps({"docket_id": r["docket_id"], "case": r.get("caseName"), "court": r.get("court_id"), "filed": r.get("dateFiled"),
                                     "terminated": r.get("dateTerminated"), "docket": r.get("docketNumber"), "cause": r.get("cause"), "nos": r.get("suitNature")}) + "\n")
            url = d.get("next")
            fh.flush()
            if n and n % 500 < 20:
                print(f"  {n:,} dockets ({time.time() - t0:.0f} s)", flush=True)
            time.sleep(PAUSE)
    print(f"{n:,} new dockets -> {dst} ({len(seen):,} total)")
    print("COURTLISTENER DONE")


if __name__ == "__main__":
    main()
