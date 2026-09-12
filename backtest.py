#!/usr/bin/env python3
"""Is the restatement score worth money? Forward returns by score decile, out of sample.

Every test filing (2019-2023) is scored by the headline model, which never saw those years.
The stock is bought at the close of the first trading day after the 10-K is filed and held six
and twelve months; returns are measured against SPY over the same window. Deciles are formed
within each filing year. Filings whose ticker's price history ends before the exit are closed
at the last available price and counted as "went dark" - Yahoo drops delisted tickers, so the
short side's best outcomes are the ones most likely to be missing.

  python backtest.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from relabel import submissions

MARKET = Path("data/raw/market")
RNG = np.random.default_rng(7)


def bootstrap_diff(a, b, n=2000):
    d = [RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def main():
    z = np.load("data/out/scores_headline_final.npz", allow_pickle=True)
    adsh, y, filed, score = z["adsh"], z["y"], pd.to_datetime(z["filed"]), z["s4"]
    year = filed.year
    subs = submissions()
    tick = json.load(open(MARKET / "company_tickers.json"))
    by_cik = {}
    for v in tick.values():
        by_cik.setdefault(str(int(v["cik_str"])), v["ticker"])
    ticker = np.array([by_cik.get(subs.get(a, ("?",))[0] or "?", "") for a in adsh.tolist()])
    prices = pd.read_parquet(MARKET / "prices.parquet", columns=["date", "ticker", "adj"])
    prices["date"] = pd.to_datetime(prices["date"])
    spy = prices[prices.ticker == "SPY"].set_index("date")["adj"].sort_index()
    import sys; sys.path.insert(0, "signals"); from panel import month_panel
    _, _, broken = month_panel(screen="ticker")                                   # the same broken-history screen as the portfolio test
    series = {t: g.set_index("date")["adj"].sort_index() for t, g in prices[(prices.ticker != "SPY") & ~prices.ticker.isin(broken)].groupby("ticker")}
    print(f"test filings {len(adsh):,}; with a ticker {int((ticker != '').sum()):,}; with prices {sum(1 for t in ticker if t in series):,}\n")

    rows = []
    for i in range(len(adsh)):
        s = series.get(ticker[i])
        if s is None:
            continue
        start = filed[i] + pd.Timedelta(days=1)
        after = s.loc[start:]
        if len(after) < 5:
            continue
        p0, d0 = after.iloc[0], after.index[0]
        rec = {"i": i, "year": year[i], "score": score[i], "y": y[i], "decile": np.nan, "p0": p0}
        for name, days in (("r6", 182), ("r12", 365)):
            end = d0 + pd.Timedelta(days=days)
            win = s.loc[d0:end]
            p1 = win.iloc[-1]
            dark = win.index[-1] < end - pd.Timedelta(days=10) and s.index[-1] < end - pd.Timedelta(days=10)
            m0 = spy.loc[d0:].iloc[0]; m1 = spy.loc[d0:end].iloc[-1]
            rec[name] = p1 / p0 - 1
            rec[name + "_adj"] = (p1 / p0 - 1) - (m1 / m0 - 1)
            rec[name + "_dark"] = bool(dark)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df = df[df.year <= 2023]
    df["decile"] = df.groupby("year")["score"].transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False) + 1)
    print(f"{len(df):,} filings with a price path after filing ({int(df.y.sum())} later restated)\n")

    print("did the restatements themselves hurt? market-adjusted 12-month return after the 10-K:")
    for lab, m in (("later restated", df.y == 1), ("not restated", df.y == 0)):
        print(f"  {lab:<16} n {int(m.sum()):>6,}  mean {df.loc[m, 'r12_adj'].mean():+.1%}  median {df.loc[m, 'r12_adj'].median():+.1%}  went dark within 12m {df.loc[m, 'r12_dark'].mean():.1%}")

    print("\nby score decile (10 = most likely to restate), formed within each year:")
    print(f"{'decile':>7}{'n':>7}{'restated':>10}{'mean 6m adj':>13}{'mean 12m adj':>14}{'median 12m adj':>16}{'went dark 12m':>15}{'no price data*':>16}")
    for d in range(1, 11):
        m = df.decile == d
        print(f"{d:>7}{int(m.sum()):>7,}{df.loc[m, 'y'].mean():>10.1%}{df.loc[m, 'r6_adj'].mean():>+13.1%}{df.loc[m, 'r12_adj'].mean():>+14.1%}{df.loc[m, 'r12_adj'].median():>+16.1%}{df.loc[m, 'r12_dark'].mean():>15.1%}{'':>16}")
    # survivorship: how many filings per decile of the FULL test set have no prices at all
    full = pd.DataFrame({"year": year, "score": score, "has": [t in series for t in ticker]})
    full["decile"] = full.groupby("year")["score"].transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False) + 1)
    print("* share of ALL test filings in that decile with no usable price history (Yahoo drops delisted tickers):")
    print("  " + "  ".join(f"d{d}: {1 - full.loc[full.decile == d, 'has'].mean():.0%}" for d in range(1, 11)))

    print("\nthe same deciles with the tails tamed (12-month market-adjusted):")
    print(f"{'decile':>7}{'winsorised mean (cap +100%)':>29}{'trimmed mean (5%)':>19}{'lost > 50%':>12}{'gained > 100%':>15}{'restated':>10}")
    for d in range(1, 11):
        r = df.loc[df.decile == d, "r12_adj"].values
        w = np.clip(r, -1.0, 1.0); lo_, hi_ = np.percentile(r, [5, 95]); tr = r[(r >= lo_) & (r <= hi_)]
        print(f"{d:>7}{w.mean():>+29.1%}{tr.mean():>+19.1%}{(r < -0.5).mean():>12.1%}{(r > 1.0).mean():>15.1%}{df.loc[df.decile == d, 'y'].mean():>10.1%}")
    top, bot, rest = df[df.decile == 10], df[df.decile == 1], df[df.decile < 10]
    for name, a, b in (("decile 10 minus decile 1", top, bot), ("decile 10 minus deciles 1-9", top, rest)):
        wa, wb = np.clip(a.r12_adj.values, -1, 1), np.clip(b.r12_adj.values, -1, 1)
        lo, hi = bootstrap_diff(wa, wb)
        print(f"{name}, winsorised 12-month market-adjusted: {wa.mean() - wb.mean():+.1%} [{lo:+.1%}, {hi:+.1%}]")
    print("\nper year, decile 10 winsorised mean and share losing more than half:")
    for yr in sorted(df.year.unique()):
        r = df[(df.year == yr) & (df.decile == 10)].r12_adj.values
        print(f"  {yr}: n {len(r):>4}  winsorised mean {np.clip(r, -1, 1).mean():+.1%}   lost > 50%: {(r < -0.5).mean():.0%}   gained > 100%: {(r > 1).mean():.0%}")
    for name, a, b in (("decile 10 minus decile 1", top, bot), ("decile 10 minus deciles 1-9", top, rest)):
        lo, hi = bootstrap_diff(top.r12_adj.values, b.r12_adj.values)
        print(f"\n{name}, 12-month market-adjusted: {a.r12_adj.mean() - b.r12_adj.mean():+.1%} [{lo:+.1%}, {hi:+.1%}]")
    print("\nper year, decile 10 mean 12-month market-adjusted return (a short position earns the negative of this):")
    for yr in sorted(df.year.unique()):
        t = df[(df.year == yr) & (df.decile == 10)]
        print(f"  {yr}: n {len(t):>4}  {t.r12_adj.mean():+.1%}  (median {t.r12_adj.median():+.1%}; {t.r12_dark.mean():.0%} went dark)")
    for floor in (5.0, 10.0):
        big = df[df.p0 >= floor].copy()
        big["decile"] = big.groupby("year")["score"].transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False) + 1)
        print(f"\nstocks priced at least ${floor:.0f} at entry ({len(big):,} filings, {int(big.y.sum())} later restated): 12-month market-adjusted")
        print(f"{'decile':>7}{'n':>7}{'restated':>10}{'raw mean':>10}{'winsorised':>12}{'median':>9}{'lost > 50%':>12}{'gained > 100%':>15}")
        for d in range(1, 11):
            r = big.loc[big.decile == d, "r12_adj"].values
            print(f"{d:>7}{len(r):>7,}{big.loc[big.decile == d, 'y'].mean():>10.1%}{r.mean():>+10.1%}{np.clip(r, -1, 1).mean():>+12.1%}{np.median(r):>+9.1%}{(r < -0.5).mean():>12.1%}{(r > 1).mean():>15.1%}")
        t, o = big[big.decile == 10].r12_adj.values, big[big.decile < 10].r12_adj.values
        lo, hi = bootstrap_diff(t, o); lo2, hi2 = bootstrap_diff(np.clip(t, -1, 1), np.clip(o, -1, 1))
        print(f"  decile 10 minus 1-9: raw {t.mean() - o.mean():+.1%} [{lo:+.1%}, {hi:+.1%}]; winsorised {np.clip(t, -1, 1).mean() - np.clip(o, -1, 1).mean():+.1%} [{lo2:+.1%}, {hi2:+.1%}]")
    print("BACKTEST DONE")


if __name__ == "__main__":
    main()
