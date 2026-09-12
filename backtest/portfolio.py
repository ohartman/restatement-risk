#!/usr/bin/env python3
"""Is the restatement score tradable? A monthly long-short backtest with frictions and a
factor regression, on the out-of-sample scores (10-Ks filed 2019-2023, scored by a model
trained on filings through 2018).

Each month-end, every stock's signal is the score of its latest 10-K filed in the past
twelve months. Stocks below a price floor are excluded. Deciles are formed on the signal;
the portfolio is long decile 1 (least likely to restate) and short decile 10, equal-weighted,
rebalanced monthly. A stock whose price history ends mid-month is closed at its last price
and charged a delisting return (Shumway 1997: -30% for performance delistings; 0% is also
reported). Returns are regressed on the Fama-French five factors plus momentum (Ken
French's library, free). Costs: a one-way transaction cost on turnover and an annual
borrow fee on the short leg, both at published ranges.

  python backtest/portfolio.py [--floor 10] [--delist -0.30]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from relabel import submissions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MARKET = ROOT / "data/raw/market"; FACT = ROOT / "data/raw/factors"; OUT = ROOT / "data/out"


def factors():
    ff = pd.read_csv(FACT / "F-F_Research_Data_5_Factors_2x3_daily.csv", skiprows=3, index_col=0)
    ff = ff[ff.index.astype(str).str.match(r"^\d{8}$")]; ff.index = pd.to_datetime(ff.index.astype(str)); ff = ff.astype(float) / 100
    mom = pd.read_csv(FACT / "F-F_Momentum_Factor_daily.csv", skiprows=13, index_col=0)
    mom = mom[mom.index.astype(str).str.strip().str.match(r"^\d{8}$")]; mom.index = pd.to_datetime(mom.index.astype(str).str.strip())
    mom.columns = ["MOM"]; mom = mom.astype(float) / 100
    d = ff.join(mom, how="inner")
    m = (1 + d).groupby(d.index.to_period("M")).prod() - 1        # compound daily to monthly
    return m


def ols(y, X):
    X1 = np.column_stack([np.ones(len(y)), X]); beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta; n, k = X1.shape
    # Newey-West (lag 3) standard errors
    S = np.zeros((k, k)); u = resid[:, None] * X1
    S += u.T @ u
    for L in range(1, 4):
        w = 1 - L / 4; G = u[L:].T @ u[:-L]; S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X1.T @ X1); V = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.diag(V))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--floor", type=float, default=10.0); ap.add_argument("--delist", type=float, default=-0.30)
    ap.add_argument("--tcost", type=float, default=0.0025, help="one-way transaction cost per dollar traded")
    args = ap.parse_args()
    z = np.load(OUT / "scores_headline_final.npz", allow_pickle=True)
    adsh, filed, score = z["adsh"], pd.to_datetime(z["filed"]), z["s4"]
    subs = submissions(); tick = json.load(open(MARKET / "company_tickers.json")); by_cik = {}
    for v in tick.values():
        by_cik.setdefault(str(int(v["cik_str"])), v["ticker"])
    sig = pd.DataFrame({"ticker": [by_cik.get(subs.get(a, ("?",))[0] or "?", "") for a in adsh.tolist()], "filed": filed, "score": score})
    sig = sig[sig.ticker != ""].sort_values("filed")
    prices = pd.read_parquet(MARKET / "prices.parquet", columns=["date", "ticker", "close", "adj"]); prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices.ticker != "SPY"]
    last_day = prices.date.max()
    # month-end panel: last adj/close in each month, plus the ticker's final trading day
    prices["month"] = prices.date.dt.to_period("M")
    me = prices.sort_values("date").groupby(["ticker", "month"]).agg(adj=("adj", "last"), close=("close", "last"), last_date=("date", "max")).reset_index()
    end_of = {t: d for t, d in prices.groupby("ticker").date.max().items()}
    adj = me.pivot(index="month", columns="ticker", values="adj").sort_index()
    close = me.pivot(index="month", columns="ticker", values="close").sort_index()
    ret = adj / adj.shift(1) - 1
    # delisting: the month in which a ticker's history ends (before the panel's last day) gets the delisting return on top
    months = ret.index
    dark = pd.DataFrame(False, index=months, columns=ret.columns)
    for t, d in end_of.items():
        if d < last_day - pd.Timedelta(days=7) and t in dark.columns:
            dark.loc[d.to_period("M"), t] = True
    ret = ret.where(~dark, (1 + ret) * (1 + args.delist) - 1)
    fac = factors()

    rows, held = [], {}
    prev_long, prev_short = set(), set()
    for k in range(len(months) - 1):
        m, nxt = months[k], months[k + 1]
        asof = m.to_timestamp(how="end")
        if asof < pd.Timestamp("2019-03-31") or nxt not in ret.index:
            continue
        s = sig[(sig.filed <= asof) & (sig.filed > asof - pd.Timedelta(days=365))].groupby("ticker").score.last()
        px = close.loc[m].reindex(s.index)
        ok = s.index[(px >= args.floor).fillna(False).values & adj.loc[m].reindex(s.index).notna().values]
        s = s.loc[ok]
        if len(s) < 100:
            continue
        dec = pd.qcut(s.rank(method="first"), 10, labels=False) + 1
        longs, shorts = set(dec[dec == 1].index), set(dec[dec == 10].index)
        r_next = ret.loc[nxt]
        rl, rs = r_next.reindex(list(longs)).fillna(0).mean(), r_next.reindex(list(shorts)).fillna(0).mean()
        r_all = r_next.reindex(s.index).fillna(0).mean()
        turn = (len(longs - prev_long) + len(prev_long - longs)) / max(1, 2 * len(longs)) + (len(shorts - prev_short) + len(prev_short - shorts)) / max(1, 2 * len(shorts))
        r_ex10 = r_next.reindex(list(set(s.index) - shorts)).fillna(0).mean()
        rows.append({"month": nxt, "n": len(s), "long_d1": rl, "short_d10": rs, "universe": r_all, "ex10": r_ex10, "ls": rl - rs, "turnover": turn,
                     "mkt": fac.loc[nxt, "Mkt-RF"] if nxt in fac.index else np.nan, "rf": fac.loc[nxt, "RF"] if nxt in fac.index else np.nan})
        prev_long, prev_short = longs, shorts
    df = pd.DataFrame(rows).set_index("month")
    df = df.join(fac[["SMB", "HML", "RMW", "CMA", "MOM"]], how="left")
    df.to_csv(OUT / f"backtest_monthly_floor{int(args.floor)}.csv")
    print(f"floor ${args.floor:.0f}, delisting return {args.delist:+.0%}, one-way cost {args.tcost:.2%}: {len(df)} months {df.index[0]} to {df.index[-1]}, "
          f"median universe {int(df.n.median()):,} stocks, ~{int(df.n.median() / 10)} per leg\n")

    def summarize(name, r):
        ann = r.mean() * 12; vol = r.std() * np.sqrt(12); sharpe = ann / vol if vol > 0 else np.nan
        return f"{name:<44} {ann:>+7.1%}/yr  vol {vol:>5.1%}  Sharpe {sharpe:>5.2f}  worst month {r.min():>+6.1%}  months up {(r > 0).mean():>4.0%}"
    print(summarize("short decile 10 alone (minus market)", -(df.short_d10 - df.mkt - df.rf)))
    print(summarize("long decile 1 alone (minus market)", df.long_d1 - df.mkt - df.rf))
    print(summarize("long decile 1 / short decile 10, gross", df.ls))
    cost = df.turnover * args.tcost * 2                     # both legs traded
    print(summarize("  net of transaction costs", df.ls - cost))
    for fee in (0.02, 0.10, 0.30):
        print(summarize(f"  net of costs and {fee:.0%}/yr borrow fee", df.ls - cost - fee / 12))
    print("\nlong only: the equal-weighted universe with the riskiest decile removed, versus the universe itself")
    print(summarize("universe, equal-weighted", df.universe))
    print(summarize("universe minus decile 10", df.ex10))
    d = df.ex10 - df.universe
    print(f"{'  difference':<44} {d.mean() * 12:>+7.1%}/yr   months better {(d > 0).mean():.0%}   worst month {d.min():+.1%}   compounded {((1 + df.ex10).prod() / (1 + df.universe).prod() - 1):+.1%} over the period")
    print("\nfactor regression of the gross long-short return on Mkt-RF, SMB, HML, RMW, CMA, MOM (Newey-West t-stats):")
    X = df[["mkt", "SMB", "HML", "RMW", "CMA", "MOM"]].values; y = df.ls.values
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    beta, se = ols(y[ok], X[ok])
    names = ["alpha (monthly)", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    for nm, b, s_ in zip(names, beta, se):
        print(f"  {nm:<16} {b:>+8.4f}   t {b / s_:>+6.2f}" + (f"   = {b * 12:+.1%}/yr" if nm.startswith("alpha") else ""))
    print("\nby year (gross long-short, and the short leg's raw return):")
    for yr, g in df.groupby(df.index.year):
        print(f"  {yr}: {len(g):>2} months  long-short {(1 + g.ls).prod() - 1:>+7.1%}   short leg raw {(1 + g.short_d10).prod() - 1:>+7.1%}   universe {(1 + g.universe).prod() - 1:>+7.1%}   market {(1 + g.mkt + g.rf).prod() - 1:>+7.1%}")
    print("PORTFOLIO DONE")


if __name__ == "__main__":
    main()
