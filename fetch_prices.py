#!/usr/bin/env python3
"""Daily prices for every company in the 10-K panel, from a free source.

The published models that clear 0.75 all lean on market data: returns before
the filing, volatility, turnover, size. Those come from CRSP, which costs money.
Yahoo Finance gives daily prices for free through yfinance, with one serious
catch that this script is built around: Yahoo drops delisted tickers, and firms
that restate delist more often than firms that don't. So "no price history"
is correlated with the label through the future. Coverage is therefore
recorded per company, and the market features are evaluated on the covered
subsample rather than letting the forest learn from the holes.

Companies are mapped CIK -> ticker with the SEC's own company_tickers.json.

  python fetch_prices.py       -> data/raw/market/prices.parquet (date, ticker, close, adj, volume)
                                  data/raw/market/coverage.csv   (cik, ticker, first, last, n_days)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from relabel import submissions

MARKET = Path("data/raw/market")
START, END = "2012-06-01", "2024-06-30"
BATCH = 100


def main():
    tick = json.load(open(MARKET / "company_tickers.json"))
    by_cik = {}
    for v in tick.values():
        by_cik.setdefault(str(int(v["cik_str"])), v["ticker"])   # first listed ticker per CIK
    z = np.load("data/out/features.npz", allow_pickle=True)
    subs = submissions()
    ciks = sorted({subs[a][0] for a in z["adsh"].tolist() if a in subs})
    wanted = {c: by_cik[c] for c in ciks if c in by_cik}
    print(f"{len(ciks):,} companies in the panel; {len(wanted):,} have a current ticker "
          f"({100*len(wanted)/len(ciks):.0f}%) -- the rest are delisted or never listed\n")

    tickers = sorted(set(wanted.values()))
    frames, done = [], set()
    out = MARKET / "prices.parquet"
    if out.exists():
        prev = pd.read_parquet(out)
        frames.append(prev); done = set(prev.ticker.unique())
        print(f"resuming: {len(done):,} tickers already on disk")
    todo = [t for t in tickers if t not in done]
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        try:
            df = yf.download(batch + ["SPY"] if i == 0 else batch, start=START, end=END,
                             auto_adjust=False, progress=False, threads=True, group_by="ticker")
        except Exception as exc:
            print(f"  batch {i//BATCH}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(30)
            continue
        rows = []
        for t in (batch + ["SPY"] if i == 0 else batch):
            try:
                d = df[t].dropna(subset=["Close"])
            except KeyError:
                continue
            if d.empty:
                continue
            rows.append(pd.DataFrame({"date": d.index, "ticker": t, "close": d["Close"].values,
                                      "adj": d["Adj Close"].values, "volume": d["Volume"].values}))
        if rows:
            frames.append(pd.concat(rows))
        if (i // BATCH) % 5 == 4 or i + BATCH >= len(todo):
            pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        got = len(rows)
        print(f"  {min(i+BATCH, len(todo)):>5}/{len(todo)} tickers   this batch {got}/{len(batch)} with data   "
              f"[{(time.time()-t0)/60:.0f} min]", flush=True)
        time.sleep(1.0)

    prices = pd.concat(frames, ignore_index=True)
    prices.to_parquet(out, index=False)
    cov = (prices.groupby("ticker").agg(first=("date", "min"), last=("date", "max"), n_days=("date", "size"))
           .reset_index())
    cov["cik"] = cov.ticker.map({t: c for c, t in wanted.items()})
    cov.to_csv(MARKET / "coverage.csv", index=False)
    print(f"\n{len(prices):,} daily rows for {prices.ticker.nunique():,} tickers -> {out}")


if __name__ == "__main__":
    main()
