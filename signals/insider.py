#!/usr/bin/env python3
"""Insider trading as a signal: do stocks that insiders buy together beat the market?

Every open-market purchase (code P) and sale (code S) by an officer, director or ten-percent
holder is on EDGAR within two business days (Form 4). The signal for month m is built from
trades FILED in month m - the public date - and the position is held in month m+1.

Three definitions, declared in advance, chosen on 2012-06 to 2017-12 and tested on
2018-01 to 2024-06 (the choice is the only fitted thing here):
  D1  cluster: net distinct insiders buying minus selling >= +2 (long) / <= -2 (short)
  D2  dollars: top / bottom decile of net dollars bought (purchases minus sales)
  D3  officers only: any officer purchase and no officer sale (long) / the reverse (short)

Universe: the panel's tickers with Yahoo prices and a month-end close of at least $5.
Reports the long leg against the equal-weighted universe, the long-short spread, and a
six-factor regression (Fama-French five plus momentum).

  python signals/insider.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "backtest"))
from insider_features import load_trades  # noqa: E402
from portfolio import factors, ols  # noqa: E402

MARKET = ROOT / "data/raw/market"
SELECT_END = pd.Period("2017-12", "M")
FLOOR = 5.0


from panel import month_panel as _panel


def month_panel(screen="ticker", cap=None):
    ret, close, _ = _panel(screen=screen, cap=cap)
    return ret, close


def summarize(name, r):
    ann = r.mean() * 12; vol = r.std() * np.sqrt(12)
    return f"{name:<52} {ann:>+7.1%}/yr  vol {vol:>5.1%}  Sharpe {ann / vol if vol > 0 else float('nan'):>5.2f}  months up {(r > 0).mean():>4.0%}  n {len(r)}"


def regress(name, r, fac):
    d = pd.concat([r.rename("r"), fac[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]]], axis=1, join="inner").dropna()
    beta, se = ols(d.r.values, d[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]].values)
    load = "  ".join(f"{n} {b:+.2f}({b / s:+.1f})" for n, b, s in zip(["Mkt", "SMB", "HML", "RMW", "CMA", "MOM"], beta[1:], se[1:]))
    print(f"  {name:<50} alpha {beta[0] * 12:>+6.1%}/yr  t {beta[0] / se[0]:>+5.2f}   {load}")


def main():
    trades = load_trades()
    tick = json.load(open(MARKET / "company_tickers.json")); by_cik = {}
    for v in tick.values():
        by_cik.setdefault(str(int(v["cik_str"])), v["ticker"])
    trades["ticker"] = trades.cik.map(by_cik); trades = trades.dropna(subset=["ticker"])
    trades["month"] = trades.filed.dt.to_period("M")
    ret, close = month_panel(); fac = factors()
    print(f"\n{len(trades):,} open-market trades with a ticker, {trades.ticker.nunique():,} tickers, {trades.month.min()} to {trades.month.max()}")

    # per issuer-month signal ingredients
    g = trades.groupby(["ticker", "month"])
    sig = pd.DataFrame({
        "buyers": g.apply(lambda d: d.loc[d.TRANS_CODE == "P", "RPTOWNERCIK"].nunique()),
        "sellers": g.apply(lambda d: d.loc[d.TRANS_CODE == "S", "RPTOWNERCIK"].nunique()),
        "net_dollars": g.apply(lambda d: d.loc[d.TRANS_CODE == "P", "value"].sum() - d.loc[d.TRANS_CODE == "S", "value"].sum()),
        "off_buy": g.apply(lambda d: bool(((d.TRANS_CODE == "P") & d.officer).any())),
        "off_sell": g.apply(lambda d: bool(((d.TRANS_CODE == "S") & d.officer).any())),
    }).reset_index()
    sig["net_traders"] = sig.buyers - sig.sellers

    months = [m for m in ret.index if pd.Period("2012-07", "M") <= m <= ret.index[-1] - 1]
    results = {k: [] for k in ("D1", "D2", "D3")}
    for m in months:
        nxt = m + 1
        if nxt not in ret.index:
            continue
        universe = close.loc[m][close.loc[m] >= FLOOR].dropna().index
        universe = universe[ret.loc[nxt].reindex(universe).notna().values]
        if len(universe) < 200:
            continue
        s = sig[(sig.month == m) & sig.ticker.isin(universe)].set_index("ticker")
        r_next = ret.loc[nxt]; r_u = r_next.reindex(universe).mean()
        sets = {
            "D1": (set(s.index[s.net_traders >= 2]), set(s.index[s.net_traders <= -2])),
            "D2": (set(s.index[s.net_dollars >= s.net_dollars.quantile(0.9)]) if len(s) >= 20 else set(),
                   set(s.index[s.net_dollars <= s.net_dollars.quantile(0.1)]) if len(s) >= 20 else set()),
            "D3": (set(s.index[s.off_buy & ~s.off_sell]), set(s.index[s.off_sell & ~s.off_buy])),
        }
        for k, (L, S) in sets.items():
            rl = r_next.reindex(list(L)).mean() if L else np.nan; rs = r_next.reindex(list(S)).mean() if S else np.nan
            results[k].append({"month": nxt, "long": rl, "short": rs, "universe": r_u, "nL": len(L), "nS": len(S)})
    out = {k: pd.DataFrame(v).set_index("month") for k, v in results.items()}

    print("\nselection period 2012-07 to 2017-12, long leg minus the equal-weighted universe:")
    best, best_s = None, -9
    for k, df in out.items():
        d = df[df.index <= SELECT_END]; r = (d.long - d.universe).dropna()
        sh = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else float("nan")
        print(f"  {k}: {summarize('long minus universe', r)}   avg names long {d.nL.mean():.0f}, short {d.nS.mean():.0f}")
        if sh > best_s:
            best, best_s = k, sh
    print(f"chosen on the selection period: {best}")

    print(f"\ntest period 2018-01 to 2024-06, definition {best}:")
    df = out[best]; t = df[df.index > SELECT_END]
    print(summarize("long leg, raw", t.long.dropna()))
    print(summarize("equal-weighted universe, raw", t.universe))
    print(summarize("long minus universe", (t.long - t.universe).dropna()))
    print(summarize("long minus short", (t.long - t.short).dropna()))
    print("six-factor regressions on the test period (alpha per year, Newey-West t; loadings with t in brackets):")
    rf = fac["RF"]
    regress("long leg, excess of the risk-free rate", (t.long - rf.reindex(t.index)).dropna(), fac)
    regress("universe, excess of the risk-free rate", (t.universe - rf.reindex(t.index)).dropna(), fac)
    regress("long minus short", (t.long - t.short).dropna(), fac)
    print("\nthe other two definitions on the test period, for the record (long minus universe):")
    for k, df in out.items():
        if k != best:
            tt = df[df.index > SELECT_END]; print("  " + k + ": " + summarize("", (tt.long - tt.universe).dropna()))
    print("\nby year, chosen definition, long minus universe:")
    for yr, gy in t.groupby(t.index.year):
        print(f"  {yr}: {((gy.long - gy.universe).mean() * 12):>+6.1%}/yr annualised over {len(gy)} months; avg names {gy.nL.mean():.0f}")
    out[best].to_csv(ROOT / "data/out/insider_signal_monthly.csv")
    print("INSIDER DONE")


if __name__ == "__main__":
    main()
