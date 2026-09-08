#!/usr/bin/env python3
"""Replace saved documents that turned out to be exhibits with the filing's real 10-K.

fetch_all_10k.py picked the main document by file name; for 296 filings the
largest non-exhibit-looking name was still an exhibit (EX-31.2, EX-21.1, ...).
This reads each filing's -index.htm, where EDGAR states every document's
Type, and saves the document typed 10-K. The old file is kept as
<adsh>.exhibit.html.gz.

  set SEC_CONTACT=Your Name you@example.com
  python refetch_main.py [data/raw/mdna/wrong_main.txt]
"""

import gzip, json, re, sys, time
from pathlib import Path

from fetch_periods import contact
from relabel import submissions
from text_features import get

OUT = Path("data/raw/mdna"); PAUSE = 0.15
ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
HREF = re.compile(r'href="([^"]+)"', re.I)


def tenk_url(cik, adsh, ua):
    nod = adsh.replace("-", "")
    page = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/{adsh}-index.htm", ua).decode("utf-8", "replace")
    best = None
    for row in ROW.findall(page):
        cells = CELL.findall(row)
        if len(cells) < 4:
            continue
        typ = re.sub(r"<[^>]+>", "", cells[3]).strip().upper()
        href = HREF.search(cells[2])
        if not href or not typ.startswith("10-K"):
            continue
        link = href.group(1)
        link = link.split("ix?doc=")[-1] if "ix?doc=" in link else link     # inline-XBRL viewer wrapper
        if link.lower().endswith((".htm", ".html", ".txt")):
            best = "https://www.sec.gov" + link if link.startswith("/") else link
            break
    return best


def main():
    ua = contact()
    src = Path(sys.argv[1] if len(sys.argv) > 1 else OUT / "wrong_main.txt")
    todo = [l.split("\t")[0] for l in src.read_text().splitlines() if l.strip()]
    subs = submissions(); ok = bad = 0; t0 = time.time()
    with (OUT / "refetch_main.jsonl").open("a", encoding="utf-8") as log:
        for i, a in enumerate(todo, 1):
            status = "ok"
            try:
                url = tenk_url(subs[a][0], a, ua); time.sleep(PAUSE)
                if not url:
                    status = "no 10-K document in index"
                else:
                    doc = get(url, ua).decode("utf-8", "replace")
                    old = OUT / f"{a}.full.html.gz"
                    if old.exists():
                        old.rename(OUT / f"{a}.exhibit.html.gz")
                    with gzip.open(old, "wt", encoding="utf-8") as g:
                        g.write(doc)
            except Exception as exc:
                status = f"{type(exc).__name__}: {str(exc)[:80]}"
            log.write(json.dumps({"adsh": a, "status": status, "url": url if status == "ok" else None}) + "\n"); log.flush()
            ok += status == "ok"; bad += status != "ok"; time.sleep(PAUSE)
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ok {ok}  failed {bad}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
