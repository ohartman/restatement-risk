#!/usr/bin/env python3
"""Cut every saved 10-K into items with EDGAR-CRAWLER's extractor -- the code that built
EDGAR-CORPUS -- so training and test flags come from one parser, and the good one.

Each data/raw/mdna/<adsh>.full.html.gz is decompressed into a scratch folder laid out the
way the extractor expects (raw/10-K/<adsh>.htm), passed through ExtractItems with the
corpus's settings (tables removed), and the items we use are written to one JSONL line.
Resumable; runs on several processes.

  python ec_extract.py [--limit N] [--procs 6]      -> data/raw/ec_items.jsonl
"""
import argparse, gzip, json, os, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tools" / "edgar-crawler"))
MDNA = HERE / "data" / "raw" / "mdna"
SCRATCH = HERE / "data" / "raw" / "ec_scratch"
OUT = HERE / "data" / "raw" / "ec_items.jsonl"
ITEMS = ["1", "1A", "3", "7", "8", "9", "9A", "9B"]
_ex = None


def init():
    global _ex
    import logging
    logging.disable(logging.CRITICAL)
    from extract_items import ExtractItems
    (SCRATCH / "10-K").mkdir(parents=True, exist_ok=True)
    (SCRATCH / "out").mkdir(parents=True, exist_ok=True)
    _ex = ExtractItems(remove_tables=True, items_to_extract=ITEMS, include_signature=False,
                       raw_files_folder=str(SCRATCH), extracted_files_folder=str(SCRATCH / "out"),
                       skip_extracted_filings=False)


def one(job):
    adsh, cik, filed = job
    try:
        src = MDNA / f"{adsh}.full.html.gz"
        dst = SCRATCH / "10-K" / f"{adsh}.htm"
        with gzip.open(src, "rt", encoding="utf-8") as g, dst.open("w", encoding="utf-8") as f:
            f.write(g.read())
        meta = {"CIK": cik, "Company": "", "Type": "10-K", "Date": filed, "filename": f"{adsh}.htm",
                "Period of Report": "", "SIC": "", "State of Inc": "", "State location": "", "Fiscal Year End": "",
                "html_index": "", "htm_file_link": "", "complete_text_file_link": "", "filename_link": ""}
        _ex.determine_items_to_extract(meta)
        res = _ex.extract_items(meta)
        dst.unlink(missing_ok=True)
        if not res:
            return {"adsh": adsh, "status": "no items"}
        out = {"adsh": adsh, "status": "ok"}
        for k in ITEMS:
            out[f"section_{k}"] = res.get(f"item_{k}", "") or ""
        return out
    except Exception as exc:
        return {"adsh": adsh, "status": f"{type(exc).__name__}: {str(exc)[:80]}"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    from relabel import submissions
    z = np.load(HERE / "data/out/features.npz", allow_pickle=True)
    adsh = z["adsh"].tolist()
    subs = submissions()
    done = set()
    if OUT.exists():
        for l in OUT.open(encoding="utf-8"):
            try:
                done.add(json.loads(l)["adsh"])
            except Exception:
                pass
    jobs = [(a, subs[a][0], str(subs[a][1])) for a in adsh if a in subs and a not in done and (MDNA / f"{a}.full.html.gz").exists()]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs):,} documents to extract ({len(done):,} already done)", flush=True)
    t0 = time.time(); ok = bad = 0
    with OUT.open("a", encoding="utf-8") as fh, Pool(args.procs, initializer=init) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4), 1):
            fh.write(json.dumps(r) + "\n")
            ok += r["status"] == "ok"; bad += r["status"] != "ok"
            if i % 500 == 0 or i == len(jobs):
                fh.flush(); rate = i / max(1, time.time() - t0)
                print(f"  {i:,}/{len(jobs):,}  ok {ok:,}  failed {bad:,}  ({rate:.2f}/s, {(len(jobs)-i)/max(rate,.01)/60:.0f} min left)", flush=True)
    print(f"{ok:,} extracted, {bad:,} failed -> {OUT}")


if __name__ == "__main__":
    main()
