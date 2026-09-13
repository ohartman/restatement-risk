#!/usr/bin/env python3
"""Raw inputs the live scorer needs for 10-Ks filed 2024-2025, into the main data/raw tree:
quarterly form indices, Form 3/4/5 data sets, and nothing else that a script does not already
fetch (the financial data sets, daily indices and 8-K events have their own fetchers).

  set SEC_CONTACT=Your Name you@example.com
  python live/fetch_inputs.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fetch_periods import contact  # noqa: E402
from text_features import get  # noqa: E402

IDX = ROOT / "data/raw/edgar_index"; INS = ROOT / "data/raw/insider"


def main():
    ua = contact()
    for y in (2024, 2025):
        for q in (1, 2, 3, 4):
            dst = IDX / f"form_{y}_Q{q}.idx"
            if dst.exists():
                continue
            try:
                dst.write_bytes(get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx", ua, timeout=300))
                print(f"  index {y} Q{q}: {dst.stat().st_size / 1e6:.0f} MB", flush=True)
            except Exception as exc:
                print(f"  index {y} Q{q}: {str(exc)[:60]}", flush=True)
            time.sleep(0.5)
    for y in (2024, 2025):
        for q in (1, 2, 3, 4):
            dst = INS / f"{y}q{q}_form345.zip"
            if dst.exists():
                continue
            try:
                dst.write_bytes(get(f"https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip", ua, timeout=600))
                print(f"  insider {y}q{q}: {dst.stat().st_size / 1e6:.0f} MB", flush=True)
            except Exception as exc:
                print(f"  insider {y}q{q}: {str(exc)[:60]}", flush=True)
            time.sleep(0.5)
    print("LIVE INPUTS DONE")


if __name__ == "__main__":
    main()
