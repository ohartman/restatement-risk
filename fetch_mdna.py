#!/usr/bin/env python3
"""Fetch Item 7 (Management's Discussion and Analysis) for a sample of 10-Ks.

The one free lever with published evidence of a real gain is the MD&A text read
by a fine-tuned language model (+0.04 over a forest, Waffo Dzuyo et al. 2026).
The full corpus is ~90 GB of HTML at the SEC's ten-requests-a-second limit, so
this takes every restated filing plus three clean ones per restated, cuts each
document to its Item 7 section, and keeps only that, gzipped.

The main document is chosen the way fetch_periods.py learned to: never an
inline-XBRL artifact (R1.htm and friends), never an exhibit.

  set SEC_CONTACT=Your Name you@example.com
  python fetch_mdna.py [--neg-per-pos 3]      -> data/raw/mdna/<adsh>.txt.gz, data/raw/mdna/index.jsonl
"""

import argparse
import gzip
import html
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from fetch_periods import EXHIBIT, XBRL_ARTIFACT, contact
from relabel import submissions
from text_features import TAG, WS, get

OUT = Path("data/raw/mdna")
PAUSE = 0.15
# "Item 7" then the MD&A heading, with anything but letters/digits allowed in between
# (dots, commas, dashes, table cells); the section ends at Item 7A or Item 8. The table of
# contents also matches, so the longest candidate section wins.
ITEM7 = re.compile(r"item\s*7(?![0-9a])[^a-z0-9]{0,80}management", re.I)
ITEM7_END = re.compile(r"item\s*7a(?![0-9])[^a-z0-9]{0,80}quantitative|item\s*8(?![0-9])[^a-z0-9]{0,80}financial\s+statements", re.I)
# Exhibits filed as part of the 10-K carry "ex" plus a number somewhere in the name.
EXHIBIT_ANY = re.compile(r"ex-?\d|exhibit", re.I)
# Soft hyphens and zero-width characters hide inside words in some filings.
INVISIBLE = re.compile("[­​‌‍﻿]")


def main_document(cik, adsh, ua):
    nod = adsh.replace("-", "")
    d = json.loads(get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/index.json", ua).decode("utf-8", "replace"))
    items = [(i["name"], int(i.get("size") or 0)) for i in d["directory"]["item"]]
    htm = [(n, sz) for n, sz in items if n.lower().endswith((".htm", ".html"))
           and not XBRL_ARTIFACT.search(n) and not EXHIBIT.search(n) and not EXHIBIT_ANY.search(n)]
    named = [(n, sz) for n, sz in htm if re.search(r"10-?k", n, re.I)]
    pool = named or htm
    if not pool:
        return None
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/{max(pool, key=lambda x: x[1])[0]}"


def mdna(html_doc):
    text = html.unescape(TAG.sub(" ", html_doc))
    text = WS.sub(" ", INVISIBLE.sub("", text))
    best = ""
    for m in ITEM7.finditer(text):
        e = ITEM7_END.search(text, m.end() + 500)
        seg = text[m.start(): e.start()] if e else text[m.start(): m.start() + 200_000]
        if len(seg) > len(best):
            best = seg
    return best if len(best.split()) >= 300 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neg-per-pos", type=float, default=3.0)
    args = ap.parse_args()
    ua = contact()
    OUT.mkdir(parents=True, exist_ok=True)

    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y = z["adsh"].tolist(), z["y"]
    subs = submissions()
    pos = [a for a, yy in zip(adsh, y) if yy == 1]
    neg = [a for a, yy in zip(adsh, y) if yy == 0]
    rng = random.Random(11)
    rng.shuffle(neg)
    take = pos + neg[:int(len(pos) * args.neg_per_pos)]
    rng.shuffle(take)
    print(f"sample {len(take):,} filings ({len(pos):,} restated)")

    idx = OUT / "index.jsonl"
    done = set()
    if idx.exists():
        for l in idx.open(encoding="utf-8"):
            try:
                done.add(json.loads(l)["adsh"])
            except Exception:
                pass
        print(f"resuming: {len(done):,} done")

    ok = fail = 0
    t0 = time.time()
    with idx.open("a", encoding="utf-8") as fh:
        for i, a in enumerate(take, 1):
            if a in done:
                continue
            cik, filed, _ = subs.get(a, (None, None, None))
            status = "ok"
            try:
                url = main_document(cik, a, ua)
                time.sleep(PAUSE)
                if not url:
                    status = "no document"
                else:
                    html = get(url, ua).decode("utf-8", "replace")
                    # Keep the whole main document too (gzipped, ~400 KB each) so later
                    # passes -- auditor name, audit opinion, risk factors -- are local.
                    with gzip.open(OUT / f"{a}.full.html.gz", "wt", encoding="utf-8") as g:
                        g.write(html)
                    sec = mdna(html)
                    if sec is None:
                        status = "no item 7"
                    else:
                        with gzip.open(OUT / f"{a}.txt.gz", "wt", encoding="utf-8") as g:
                            g.write(sec)
            except Exception as exc:
                status = f"{type(exc).__name__}"
            fh.write(json.dumps({"adsh": a, "cik": cik, "filed": str(filed), "y": int(y[adsh.index(a)]),
                                 "status": status}) + "\n")
            if status == "ok":
                ok += 1
            else:
                fail += 1
            time.sleep(PAUSE)
            if i % 100 == 0:
                fh.flush()
                rate = (ok + fail) / max(1, time.time() - t0)
                print(f"  {i}/{len(take)}  ok {ok}  failed {fail}  ({rate:.2f}/s, "
                      f"{(len(take)-i)/max(rate,.01)/60:.0f} min left)", flush=True)
    print(f"\n{ok} MD&A sections saved, {fail} failed -> {OUT}")


if __name__ == "__main__":
    main()
