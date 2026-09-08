#!/usr/bin/env python3
"""When was each SEC comment letter actually made public?

EDGAR's quarterly form index dates an UPLOAD (the SEC's letter) or CORRESP (the company's
reply) by the day it was written, but the SEC releases correspondence weeks after the review
closes. The daily indices list filings on the day they were disseminated, so the first daily
index an accession appears in is its public date.

  set SEC_CONTACT=Your Name you@example.com
  python fetch_daily_index.py 2013 2024   -> data/raw/edgar_daily/dissemination.jsonl
"""
import datetime as dt
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_periods import contact
from text_features import get

OUT = Path("data/raw/edgar_daily"); OUT.mkdir(parents=True, exist_ok=True)
KEEP = ("UPLOAD", "CORRESP")
ACC = re.compile(r"\d{10}-\d{2}-\d{6}")
_lock = threading.Lock(); _last = [0.0]


def throttle(gap):
    with _lock:
        wait = _last[0] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def main():
    y0, y1 = int(sys.argv[1]), int(sys.argv[2])
    ua = contact(); workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3; gap = 0.13 * workers
    days = [dt.date(y0, 1, 1) + dt.timedelta(days=k) for k in range((dt.date(y1, 12, 31) - dt.date(y0, 1, 1)).days + 1)]
    days = [d for d in days if d.weekday() < 5 and d <= dt.date.today()]
    done = set()
    log = OUT / "days_done.txt"
    if log.exists():
        done = set(log.read_text().split())
    todo = [d for d in days if d.isoformat() not in done]
    print(f"{len(todo):,} business days to fetch ({len(done):,} done)", flush=True)

    def work(d):
        q = (d.month - 1) // 3 + 1
        throttle(gap)
        try:
            txt = get(f"https://www.sec.gov/Archives/edgar/daily-index/{d.year}/QTR{q}/form.{d:%Y%m%d}.idx", ua).decode("latin-1")
        except Exception as exc:
            return d, None, str(exc)[:60]
        rows = []
        for line in txt.splitlines():
            if line.startswith(KEEP):
                m = ACC.search(line)
                if m:
                    rows.append((line.split()[0], m.group(0)))
        return d, rows, None

    n = 0
    with (OUT / "dissemination.jsonl").open("a", encoding="utf-8") as fh, log.open("a") as lg, ThreadPoolExecutor(workers) as ex:
        for i, (d, rows, err) in enumerate(ex.map(work, todo), 1):
            if rows is None:
                if "404" not in (err or ""):
                    print(f"  {d}: {err}", flush=True)
                lg.write(d.isoformat() + "\n"); continue
            for form, acc in rows:
                fh.write(json.dumps({"acc": acc, "form": form, "public": d.isoformat()}) + "\n"); n += 1
            lg.write(d.isoformat() + "\n")
            if i % 200 == 0:
                fh.flush(); lg.flush(); print(f"  {i:,}/{len(todo):,} days, {n:,} letter rows", flush=True)
    print(f"{n:,} UPLOAD/CORRESP dissemination rows -> {OUT / 'dissemination.jsonl'}")


if __name__ == "__main__":
    main()
