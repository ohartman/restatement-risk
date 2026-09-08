#!/usr/bin/env python3
"""Fetch Items 9A / 8 / 3 / 1A / 7 for every 10-K that still lacks text flags.

The controls, auditor and legal flags moved the model more than anything since
the raw items, but EDGAR-CORPUS stops in 2020 and the MD&A fetch was a sample.
This fills the rest -- mostly the 2021-2023 test filings -- so the headline can
be computed on the whole test set. Same document chooser and section cutter as
before; each filing's flags go straight to a jsonl, and the full document is
kept gzipped. Resumable.

  set SEC_CONTACT=Your Name you@example.com
  python fetch_9a.py            -> data/raw/mdna/flags_9a.jsonl (+ <adsh>.full.html.gz)
"""

import gzip
import json
import time
from pathlib import Path

import numpy as np

from fetch_mdna import contact, main_document
from relabel import submissions
from scrutiny_extend import sections
from scrutiny_features import NAMES, text_flags
from text_features import get

OUT = Path("data/raw/mdna")
FLAGS = OUT / "flags_9a.jsonl"
PAUSE = 0.15


def main():
    ua = contact()
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, filed = z["adsh"].tolist(), z["filed"].astype("datetime64[D]")
    S = np.load("data/out/features_scrutiny.npz", allow_pickle=True)["X"]
    need = [a for a, x, f in zip(adsh, S[:, 0], filed) if not np.isfinite(x) and f >= np.datetime64("2019-01-01")]
    # test years first, most recent last so a partial run still covers whole years
    need.sort(key=lambda a: adsh.index(a))
    done = set()
    if FLAGS.exists():
        for l in FLAGS.open(encoding="utf-8"):
            try:
                done.add(json.loads(l)["adsh"])
            except Exception:
                pass
    todo = [a for a in need if a not in done]
    print(f"{len(need):,} test filings without flags; {len(done):,} already fetched; {len(todo):,} to go")
    subs = submissions()
    ok = fail = 0
    t0 = time.time()
    with FLAGS.open("a", encoding="utf-8") as fh:
        for i, a in enumerate(todo, 1):
            cik = subs.get(a, (None,))[0]
            status, flags = "ok", None
            try:
                full = OUT / f"{a}.full.html.gz"
                if full.exists():
                    with gzip.open(full, "rt", encoding="utf-8") as g:
                        doc = g.read()
                else:
                    url = main_document(cik, a, ua)
                    time.sleep(PAUSE)
                    if not url:
                        status = "no document"
                    else:
                        doc = get(url, ua).decode("utf-8", "replace")
                        with gzip.open(full, "wt", encoding="utf-8") as g:
                            g.write(doc)
                if status == "ok":
                    sec = sections(doc)
                    if len(sec["section_9A"]) < 200 and len(sec["section_8"]) < 200:
                        status = "no sections"
                    else:
                        flags = text_flags(sec)
            except Exception as exc:
                status = type(exc).__name__
            fh.write(json.dumps({"adsh": a, "status": status, "flags": flags}) + "\n")
            ok += status == "ok"; fail += status != "ok"
            time.sleep(PAUSE)
            if i % 200 == 0:
                fh.flush()
                rate = i / max(1, time.time() - t0)
                print(f"  {i:,}/{len(todo):,}  ok {ok:,}  failed {fail:,}  ({rate:.2f}/s, {(len(todo)-i)/max(rate,.01)/60:.0f} min left)", flush=True)
    print(f"\n{ok:,} filings flagged, {fail:,} failed -> {FLAGS}")


if __name__ == "__main__":
    main()
