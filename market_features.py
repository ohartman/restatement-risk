#!/usr/bin/env python3
"""Market features for each 10-K, from the twelve months before it was filed.

What the market knew before the filing: how the stock did, how nervous it was,
how much of it changed hands, and how big the company is. Every window ends the
day before the filing date, so nothing here is a reaction to the 10-K itself.

  ret_12m, ret_6m, ret_3m    buy-and-hold return over the window (adjusted close)
  abn_12m                    return minus SPY over the same window
  vol_12m                    daily return standard deviation, annualised
  idio_vol                   std of daily return minus SPY return
  max_dd                     worst peak-to-trough drop in the window
  turnover                   mean daily volume / shares outstanding (XBRL)
  log_mcap                   log10 of price x shares outstanding
  btm                        book equity / market cap
  penny                      price below $5 on the last day
  neg_ret_days               share of days with a negative return
  price_cov                  fraction of the 12-month window with a quote (coverage, for the caveat)

Written as NaN for firms without price history; those rows are handled at
evaluation time, not learned from.

  python market_features.py   -> data/out/features_market.npz (X, adsh, names)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from relabel import submissions

MARKET = Path("data/raw/market")
NAMES = ["ret_12m", "ret_6m", "ret_3m", "abn_12m", "vol_12m", "idio_vol", "max_dd", "turnover",
         "log_mcap", "btm", "penny", "neg_ret_days", "price_cov"]


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh = z["adsh"]
    subs = submissions()
    tick = json.load(open(MARKET / "company_tickers.json"))
    by_cik = {}
    for v in tick.values():
        by_cik.setdefault(str(int(v["cik_str"])), v["ticker"])

    prices = pd.read_parquet(MARKET / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    prices = prices.sort_values(["ticker", "date"])
    spy = prices[prices.ticker == "SPY"].set_index("date")["adj"]
    groups = {t: g.set_index("date") for t, g in prices.groupby("ticker")}

    # Shares outstanding and book equity from the raw-item block, for turnover, size and BTM.
    zr = np.load("data/out/features_raw2.npz", allow_pickle=True)
    n2 = list(zr["names"]); XR = zr["X"]
    pos = {a: i for i, a in enumerate(zr["adsh"].tolist())}
    j_sh, j_eq = n2.index("shares_out_cur"), n2.index("equity_cur")

    X = np.full((len(adsh), len(NAMES)), np.nan)
    covered = 0
    for i, a in enumerate(adsh.tolist()):
        cik, filed, _ = subs.get(a, (None, None, None))
        t = by_cik.get(cik) if cik else None
        if t is None or t not in groups:
            continue
        g = groups[t]
        end = pd.Timestamp(filed) - pd.Timedelta(days=1)
        w = g.loc[end - pd.Timedelta(days=365): end]
        if len(w) < 60:
            continue
        covered += 1
        adj = w["adj"].values
        r = np.diff(adj) / adj[:-1]
        r = r[np.isfinite(r)]
        s = spy.reindex(w.index).ffill().values
        sr = np.diff(s) / s[:-1]
        ok = np.isfinite(r) & np.isfinite(sr[:len(r)]) if len(sr) >= len(r) else np.zeros(len(r), bool)

        def ret(days):
            ww = w.loc[end - pd.Timedelta(days=days): end]["adj"].values
            return ww[-1] / ww[0] - 1 if len(ww) > 5 and ww[0] > 0 else np.nan

        peak = np.maximum.accumulate(adj)
        row = dict(
            ret_12m=adj[-1] / adj[0] - 1, ret_6m=ret(182), ret_3m=ret(91),
            abn_12m=(adj[-1] / adj[0] - 1) - (s[-1] / s[0] - 1) if np.isfinite(s[0]) and s[0] > 0 else np.nan,
            vol_12m=r.std() * np.sqrt(252) if len(r) > 20 else np.nan,
            idio_vol=(r_all[ok] - sr[ok]).std() * np.sqrt(252) if ok.sum() > 20 else np.nan,
            max_dd=float(((adj - peak) / peak).min()),
            neg_ret_days=float((r < 0).mean()) if len(r) else np.nan,
            penny=float(w["close"].values[-1] < 5),
            price_cov=len(w) / 252,
        )
        j = pos.get(a)
        shares = 10 ** abs(XR[j, j_sh]) - 1 if j is not None and np.isfinite(XR[j, j_sh]) else np.nan
        equity = np.sign(XR[j, j_eq]) * (10 ** abs(XR[j, j_eq]) - 1) if j is not None and np.isfinite(XR[j, j_eq]) else np.nan
        mcap = w["close"].values[-1] * shares if np.isfinite(shares) and shares > 0 else np.nan
        row["turnover"] = float(w["volume"].mean() / shares) if np.isfinite(shares) and shares > 0 else np.nan
        row["log_mcap"] = np.log10(mcap) if np.isfinite(mcap) and mcap > 0 else np.nan
        row["btm"] = equity / mcap if np.isfinite(mcap) and mcap > 0 and np.isfinite(equity) else np.nan
        X[i] = [row[n] for n in NAMES]

    y = z["y"]
    has = np.isfinite(X[:, 0])
    print(f"market features for {covered:,} of {len(adsh):,} filings ({100*covered/len(adsh):.0f}%)")
    print(f"restatement rate: covered {y[has].mean():.2%}   not covered {y[~has].mean():.2%}   "
          f"<- the survivorship caveat, in numbers")
    Path("data/out").mkdir(exist_ok=True)
    np.savez("data/out/features_market.npz", X=X, adsh=adsh, names=np.array(NAMES))
    print("coverage per feature: " + ", ".join(f"{n} {100*np.isfinite(X[:, k]).mean():.0f}%" for k, n in enumerate(NAMES)))


if __name__ == "__main__":
    main()
