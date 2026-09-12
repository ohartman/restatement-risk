#!/usr/bin/env python3
"""Fetch the SEC's Form 13F data sets (quarterly, 2013q2 on) and a CUSIP-to-ticker map.

13F holdings are keyed by CUSIP. The free map comes from the SEC's own fails-to-deliver
files, which carry CUSIP, symbol and description for every settled security; one file per
half-month, so a handful spread over the years covers the universe.

  set SEC_CONTACT=Your Name you@example.com
  python signals/fetch_13f.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fetch_periods import contact  # noqa: E402
from text_features import get  # noqa: E402

OUT = ROOT / "data/raw/form13f"; OUT.mkdir(parents=True, exist_ok=True)
FTD = ROOT / "data/raw/ftd"; FTD.mkdir(parents=True, exist_ok=True)


def main():
    ua = contact()
    quarters = [f"{y}q{q}" for y in range(2013, 2025) for q in (1, 2, 3, 4)]
    quarters = [q for q in quarters if q >= "2013q2" and q <= "2024q2"]
    for q in quarters:
        dst = OUT / f"{q}_form13f.zip"
        if dst.exists() and dst.stat().st_size > 1_000_000:
            continue
        try:
            dst.write_bytes(get(f"https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{q}_form13f.zip", ua, timeout=600))
            print(f"  {q}: {dst.stat().st_size / 1e6:.0f} MB", flush=True)
        except Exception as exc:
            print(f"  {q}: {exc}", flush=True)
        time.sleep(0.5)
    # fails-to-deliver: one file per half-month; take the first half of every June and December
    for y in range(2013, 2025):
        for mm in ("06", "12"):
            dst = FTD / f"cnsfails{y}{mm}a.zip"
            if dst.exists():
                continue
            try:
                dst.write_bytes(get(f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{y}{mm}a.zip", ua, timeout=120))
                print(f"  ftd {y}-{mm}: {dst.stat().st_size / 1e3:.0f} KB", flush=True)
            except Exception as exc:
                print(f"  ftd {y}-{mm}: {exc}", flush=True)
            time.sleep(0.3)
    print("FETCH 13F DONE")


if __name__ == "__main__":
    main()
