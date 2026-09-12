#!/usr/bin/env python3
"""The monthly price panel shared by every backtest, with one explicit data-quality screen.

Yahoo's history for a few hundred tickers is broken: recycled symbols spliced onto another
company's prices, and split factors applied to the wrong side, producing month-end closes in
the hundreds of millions of dollars and monthly "returns" of +100,000%. Those are not moves
a portfolio could have earned or lost. The screen drops a ticker's whole history if it ever
shows a monthly return above +1,000% or a month-end close above $100,000 or below $0.01 -
a series that broken once cannot be trusted elsewhere. Genuine large moves (a distressed
stock tripling) are kept; `cap` optionally winsorises what remains at +300%.

  from panel import month_panel
  ret, close, dropped = month_panel(screen="ticker", cap=None, delist=-0.30)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MARKET = ROOT / "data/raw/market"


def month_panel(screen="ticker", cap=None, delist=-0.30):
    prices = pd.read_parquet(MARKET / "prices.parquet", columns=["date", "ticker", "close", "adj"]); prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices.ticker != "SPY"]; last_day = prices.date.max()
    prices["month"] = prices.date.dt.to_period("M")
    me = prices.sort_values("date").groupby(["ticker", "month"]).agg(adj=("adj", "last"), close=("close", "last")).reset_index()
    adj = me.pivot(index="month", columns="ticker", values="adj").sort_index()
    close = me.pivot(index="month", columns="ticker", values="close").sort_index()
    ret = adj / adj.shift(1) - 1
    dropped = []
    if screen == "ticker":
        broken = (ret > 10.0).any() | (close > 1e5).any() | (close < 0.01).any()
        dropped = list(ret.columns[broken])
        ret = ret.drop(columns=dropped); close = close.drop(columns=dropped)
    dark = pd.DataFrame(False, index=ret.index, columns=ret.columns)
    for t, d in prices.groupby("ticker").date.max().items():
        if d < last_day - pd.Timedelta(days=7) and t in dark.columns:
            dark.loc[d.to_period("M"), t] = True
    ret = ret.where(~dark, (1 + ret) * (1 + delist) - 1)
    if cap is not None:
        ret = ret.clip(upper=cap)
    print(f"  month panel: {ret.shape[1]:,} tickers, screen={screen} dropped {len(dropped)} broken series, cap={cap}, delisting return {delist:+.0%}", flush=True)
    return ret, close, dropped
