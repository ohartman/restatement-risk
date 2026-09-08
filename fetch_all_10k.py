#!/usr/bin/env python3
"""Save the full main document of every 10-K in the panel that is not already on disk,
so every text flag can be cut by one parser. Resumable; training years first.

  set SEC_CONTACT=Your Name you@example.com
  python fetch_all_10k.py
"""
import gzip, json, time
from pathlib import Path
import numpy as np
from fetch_mdna import contact, main_document
from relabel import submissions
from text_features import get
OUT = Path("data/raw/mdna"); LOG = OUT / "fetch_all.jsonl"; PAUSE = 0.15
ua = contact()
z = np.load("data/out/features.npz", allow_pickle=True)
adsh, filed = z["adsh"].tolist(), z["filed"].astype("datetime64[D]")
order = np.argsort(filed)                                   # oldest first: training years before test years
todo = [adsh[i] for i in order if not (OUT / f"{adsh[i]}.full.html.gz").exists()]
failed = set()
if LOG.exists():
    for l in LOG.open(encoding="utf-8"):
        r = json.loads(l)
        if r["status"] != "ok":
            failed.add(r["adsh"])
todo = [a for a in todo if a not in failed]
print(f"{len(todo):,} filings without a saved document ({len(failed):,} earlier failures skipped)")
subs = submissions(); ok = bad = 0; t0 = time.time()
with LOG.open("a", encoding="utf-8") as fh:
    for i, a in enumerate(todo, 1):
        status = "ok"
        try:
            url = main_document(subs[a][0], a, ua); time.sleep(PAUSE)
            if not url:
                status = "no document"
            else:
                doc = get(url, ua).decode("utf-8", "replace")
                with gzip.open(OUT / f"{a}.full.html.gz", "wt", encoding="utf-8") as g:
                    g.write(doc)
        except Exception as exc:
            status = type(exc).__name__
        fh.write(json.dumps({"adsh": a, "status": status}) + "\n")
        ok += status == "ok"; bad += status != "ok"; time.sleep(PAUSE)
        if i % 500 == 0:
            fh.flush(); rate = i / max(1, time.time() - t0)
            print(f"  {i:,}/{len(todo):,}  ok {ok:,}  failed {bad:,}  ({rate:.2f}/s, {(len(todo)-i)/max(rate,.01)/60:.0f} min left)", flush=True)
print(f"\n{ok:,} documents saved, {bad:,} failed")
