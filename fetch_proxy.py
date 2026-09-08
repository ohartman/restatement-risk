#!/usr/bin/env python3
"""Fetch the proxy statement (DEF 14A / 14C) that follows each 10-K in the panel.

Two-thirds of 10-Ks incorporate Item 14 (auditor fees) by reference to the proxy.
The EDGAR form indices already on disk give every proxy by CIK and date; the
first one filed within 240 days after the 10-K is the one that covers it. The
proxy's main document is taken from the filing's -index.htm, where EDGAR states
each document's type.

  set SEC_CONTACT=Your Name you@example.com
  python fetch_proxy.py [--workers 3] [--limit N]
"""
import argparse, datetime as dt, glob, gzip, json, re, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from fetch_periods import contact
from relabel import submissions
from text_features import get

OUT = Path("data/raw/proxy"); OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "fetch_proxy.jsonl"; MAP = OUT / "proxy_for_10k.jsonl"
ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I); CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I); HREF = re.compile(r'href="([^"]+)"', re.I)
_lock = threading.Lock(); _last = [0.0]


def throttle(gap):
    with _lock:
        now = time.time(); wait = _last[0] + gap - now
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()


def main_doc_url(cik, adsh, ua, gap):
    nod = adsh.replace("-", "")
    throttle(gap)
    page = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/{adsh}-index.htm", ua).decode("utf-8", "replace")
    for row in ROW.findall(page):
        cells = CELL.findall(row)
        if len(cells) < 4: continue
        typ = re.sub(r"<[^>]+>", "", cells[3]).strip().upper(); href = HREF.search(cells[2])
        if href and typ.startswith("DEF 14"):
            link = href.group(1); link = link.split("ix?doc=")[-1] if "ix?doc=" in link else link
            if link.lower().endswith((".htm", ".html", ".txt")):
                return "https://www.sec.gov" + link if link.startswith("/") else link
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(); ua = contact(); gap = 0.12 * args.workers     # ~8 requests/s across workers, under the SEC's 10
    z = np.load("data/out/features.npz", allow_pickle=True); adsh = z["adsh"].tolist()
    subs = submissions(); ciks = {subs[a][0] for a in adsh if a in subs}
    proxies = defaultdict(list)
    for f in sorted(glob.glob("data/raw/edgar_index/form_20[12]*_Q*.idx")):
        for line in open(f, encoding="latin-1"):
            if line.startswith("DEF 14A") or line.startswith("DEF 14C"):
                parts = line.rstrip("\n").split(); fname, date, cik = parts[-1], parts[-2], parts[-3]
                if cik in ciks: proxies[cik].append((date, fname))
    pairs = {}
    for a in adsh:
        cik, fd, _ = subs.get(a, (None, None, None))
        if not cik: continue
        best = None
        for d, fn in sorted(proxies.get(cik, [])):
            days = (dt.date.fromisoformat(d) - fd).days
            if 0 <= days <= 240: best = (d, fn); break
        if best:
            pacc = re.search(r"\d{10}-\d{2}-\d{6}", best[1]).group(0)
            pairs[a] = (cik, pacc, best[0])
    with MAP.open("w", encoding="utf-8") as fh:
        for a, (cik, pacc, d) in pairs.items(): fh.write(json.dumps({"adsh": a, "cik": cik, "proxy": pacc, "proxy_date": d}) + "\n")
    done = set()
    if LOG.exists():
        for l in LOG.open(encoding="utf-8"):
            try: r = json.loads(l); done.add(r["proxy"])
            except Exception: pass
    todo = sorted({(cik, pacc) for cik, pacc, _ in pairs.values() if pacc not in done and not (OUT / f"{pacc}.html.gz").exists()})
    if args.limit: todo = todo[:args.limit]
    print(f"{len(pairs):,} 10-Ks have a proxy within 240 days; {len(todo):,} proxies to fetch ({len(done):,} logged already)", flush=True)
    t0 = time.time(); ok = bad = 0; lock = threading.Lock()

    def work(job):
        cik, pacc = job; status = "ok"
        try:
            url = main_doc_url(cik, pacc, ua, gap)
            if not url: status = "no DEF 14 document"
            else:
                throttle(gap); doc = get(url, ua).decode("utf-8", "replace")
                with gzip.open(OUT / f"{pacc}.html.gz", "wt", encoding="utf-8") as g: g.write(doc)
        except Exception as exc:
            status = f"{type(exc).__name__}: {str(exc)[:80]}"
        return pacc, status

    with LOG.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as ex:
        for i, (pacc, status) in enumerate(ex.map(work, todo), 1):
            fh.write(json.dumps({"proxy": pacc, "status": status}) + "\n")
            ok += status == "ok"; bad += status != "ok"
            if i % 200 == 0 or i == len(todo):
                fh.flush(); rate = i / max(1, time.time() - t0)
                print(f"  {i:,}/{len(todo):,}  ok {ok:,}  failed {bad:,}  ({rate:.2f}/s, {(len(todo) - i) / max(rate, .01) / 60:.0f} min left)", flush=True)
    print(f"{ok:,} fetched, {bad:,} failed -> {OUT}")


if __name__ == "__main__":
    main()
