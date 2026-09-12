#!/usr/bin/env python3
"""Cloning 13F filers: does following disclosed holdings, with the mandatory lag, beat the market?

Every institution managing over $100M discloses its long positions each quarter, within 45
days of quarter end. This clones those disclosures with the lag a real copier faces: for
report quarter Q, only filings made by the deadline are used, and the portfolio is formed
at the end of the month after the deadline (Q = 31 March -> filings by 15 May -> buy at
31 May, hold to the next formation).

Two definitions, declared in advance, chosen on 2013-2017 and tested on 2018-2024:
  C1  best ideas of the best managers: managers ranked by the trailing eight-quarter return
      of their disclosed holdings (equal-weighted, in our price universe); the top decile's
      ten largest positions each, equal-weighted.
  C2  consensus: the fifty stocks held by the most distinct managers, equal-weighted.

CUSIPs map to tickers through the SEC's fails-to-deliver files. The universe is the price
panel (non-financial 10-K filers with Yahoo prices), month-end close of at least $5.

  python signals/clone13f.py
"""
import glob
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "backtest")); sys.path.insert(0, str(ROOT / "signals"))
from insider import month_panel, regress, summarize  # noqa: E402
from portfolio import factors  # noqa: E402

F13 = ROOT / "data/raw/form13f"; FTD = ROOT / "data/raw/ftd"
SELECT_END = pd.Period("2017-12", "M"); FLOOR = 5.0


def cusip_map():
    m = {}
    for p in sorted(glob.glob(str(FTD / "*.zip"))):
        z = zipfile.ZipFile(p)
        for n in z.namelist():
            txt = z.read(n).decode("latin-1")
            for line in txt.splitlines()[1:]:
                parts = line.split("|")
                if len(parts) >= 3 and len(parts[1]) == 9:
                    m.setdefault(parts[1], parts[2].strip())
    return m


def load_13f(tickers_wanted, cmap):
    """Holdings by (manager CIK, report period) for stocks in our universe, filed by the deadline."""
    hold = []
    for p in sorted(glob.glob(str(F13 / "*_form13f.zip"))):
        z = zipfile.ZipFile(p)
        sub = pd.read_csv(io.BytesIO(z.read("SUBMISSION.tsv")), sep="\t", dtype=str, usecols=["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
        sub = sub[sub.SUBMISSIONTYPE == "13F-HR"].copy()
        sub["filed"] = pd.to_datetime(sub.FILING_DATE, format="%d-%b-%Y", errors="coerce"); sub["period"] = pd.to_datetime(sub.PERIODOFREPORT, format="%d-%b-%Y", errors="coerce")
        sub = sub[(sub.filed - sub.period).dt.days.between(0, 46)]                    # by the deadline
        info = pd.read_csv(io.BytesIO(z.read("INFOTABLE.tsv")), sep="\t", dtype=str, usecols=["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "PUTCALL"])
        info = info[info.PUTCALL.isna() & info.ACCESSION_NUMBER.isin(sub.ACCESSION_NUMBER)]
        info["ticker"] = info.CUSIP.str.upper().map(cmap); info = info.dropna(subset=["ticker"]); info = info[info.ticker.isin(tickers_wanted)]
        info["value"] = pd.to_numeric(info.VALUE, errors="coerce")
        df = info.merge(sub[["ACCESSION_NUMBER", "CIK", "period", "filed"]], on="ACCESSION_NUMBER")
        # one filing per manager-period (the latest by the deadline); values in $ thousands before 2023, dollars after
        df["value"] = np.where(df.period < pd.Timestamp("2023-01-01"), df.value * 1000, df.value)
        g = df.groupby(["CIK", "period", "ticker"], as_index=False).value.sum()
        hold.append(g); print(f"  {Path(p).stem}: {len(sub):,} filings by the deadline, {len(g):,} in-universe positions", flush=True)
    return pd.concat(hold, ignore_index=True)


def main():
    ret, close = month_panel(); fac = factors()
    cmap = cusip_map(); print(f"CUSIP map: {len(cmap):,} securities")
    tickers = set(ret.columns)
    H = load_13f(tickers, cmap)
    H = H.groupby(["CIK", "period", "ticker"], as_index=False).value.last()
    periods = sorted(H.period.unique())
    print(f"\n{len(H):,} manager-period-stock rows, {H.CIK.nunique():,} managers, {len(periods)} report periods {periods[0].date()} to {periods[-1].date()}")

    # manager trailing performance: equal-weighted return of disclosed holdings over the quarter after the deadline
    quarter_ret = {}
    for q in periods:
        form = (q + pd.DateOffset(months=2)).to_period("M")                       # deadline month end
        hold_m = [form + 1, form + 2, form + 3]
        if not all(m in ret.index for m in hold_m):
            continue
        r3 = (1 + ret.loc[hold_m]).prod() - 1
        quarter_ret[q] = r3
    perf = defaultdict(dict)
    for q, g in H.groupby("period"):
        if q not in quarter_ret:
            continue
        r3 = quarter_ret[q]
        for cik, gg in g.groupby("CIK"):
            if len(gg) >= 20 and gg.value.sum() >= 1e8:
                perf[cik][q] = r3.reindex(gg.ticker).mean()

    rows = {"C1": [], "C2": []}
    for q in periods:
        form = (q + pd.DateOffset(months=2)).to_period("M"); nxt3 = [form + 1, form + 2, form + 3]
        if not all(m in ret.index for m in nxt3) or form not in close.index:
            continue
        universe = close.loc[form][close.loc[form] >= FLOOR].dropna().index
        g = H[H.period == q]
        # C1: managers with eight prior quarters of performance, top decile by trailing mean
        prior = [p for p in periods if p < q][-8:]
        tr = {c: np.mean([perf[c][p] for p in prior if p in perf[c]]) for c in perf if sum(p in perf[c] for p in prior) >= 6}
        picks1 = set()
        if len(tr) >= 30:
            cut = np.quantile(list(tr.values()), 0.9); best = [c for c, v in tr.items() if v >= cut]
            for c in best:
                top = g[(g.CIK == c) & g.ticker.isin(universe)].nlargest(10, "value").ticker
                picks1.update(top)
        # C2: consensus by number of holders
        counts = g[g.ticker.isin(universe)].groupby("ticker").CIK.nunique()
        picks2 = set(counts.nlargest(50).index)
        r_u = ((1 + ret.loc[nxt3].reindex(columns=universe)).prod() - 1).mean()
        for k, picks in (("C1", picks1), ("C2", picks2)):
            if len(picks) < 10:
                continue
            r_p = ((1 + ret.loc[nxt3].reindex(columns=list(picks)).fillna(0)).prod() - 1).mean()
            rows[k].append({"quarter": form, "port": r_p, "universe": r_u, "n": len(picks), "mkt": (1 + fac.loc[nxt3, "Mkt-RF"] + fac.loc[nxt3, "RF"]).prod() - 1 if all(m in fac.index for m in nxt3) else np.nan})
    out = {k: pd.DataFrame(v).set_index("quarter") for k, v in rows.items()}

    def q_summary(name, r):
        ann = r.mean() * 4; vol = r.std() * 2
        return f"{name:<52} {ann:>+7.1%}/yr  vol {vol:>5.1%}  Sharpe {ann / vol if vol > 0 else float('nan'):>5.2f}  quarters up {(r > 0).mean():>4.0%}  n {len(r)}"
    print("\nselection period (formation quarters through 2017), portfolio minus the equal-weighted universe:")
    best, best_s = None, -9
    for k, df in out.items():
        d = df[df.index <= SELECT_END]; r = d.port - d.universe
        sh = r.mean() / r.std() * 2 if len(r) > 3 and r.std() > 0 else float("nan")
        print(f"  {k}: {q_summary('', r)}   avg names {d.n.mean():.0f}")
        if sh > best_s:
            best, best_s = k, sh
    print(f"chosen: {best}")
    df = out[best]; t = df[df.index > SELECT_END]
    print(f"\ntest period 2018 on, definition {best}, quarterly:")
    print(q_summary("clone portfolio, raw", t.port)); print(q_summary("equal-weighted universe, raw", t.universe)); print(q_summary("market (Mkt-RF + RF), raw", t.mkt.dropna()))
    print(q_summary("clone minus universe", t.port - t.universe)); print(q_summary("clone minus market", (t.port - t.mkt).dropna()))
    # monthly series for the factor regression: spread the quarterly holding across its three months
    mrows = []
    for form, row in t.iterrows():
        for m in (form + 1, form + 2, form + 3):
            if m in ret.index:
                mrows.append({"month": m, "form": form})
    # rebuild monthly portfolio returns for the chosen definition
    print("\nsix-factor regression on the test period (monthly, alpha per year, Newey-West t):")
    picks_by_form = {}
    for q in periods:
        form = (q + pd.DateOffset(months=2)).to_period("M")
        if form not in t.index:
            continue
        g = H[H.period == q]; universe = close.loc[form][close.loc[form] >= FLOOR].dropna().index
        if best == "C2":
            picks_by_form[form] = list(g[g.ticker.isin(universe)].groupby("ticker").CIK.nunique().nlargest(50).index)
        else:
            prior = [p for p in periods if p < q][-8:]
            tr = {c: np.mean([perf[c][p] for p in prior if p in perf[c]]) for c in perf if sum(p in perf[c] for p in prior) >= 6}
            cut = np.quantile(list(tr.values()), 0.9); s = set()
            for c in [c for c, v in tr.items() if v >= cut]:
                s.update(g[(g.CIK == c) & g.ticker.isin(universe)].nlargest(10, "value").ticker)
            picks_by_form[form] = list(s)
    mser = {}
    for form, picks in picks_by_form.items():
        for m in (form + 1, form + 2, form + 3):
            if m in ret.index:
                mser[m] = ret.loc[m].reindex(picks).fillna(0).mean()
    mser = pd.Series(mser).sort_index()
    regress("clone portfolio, excess of the risk-free rate", (mser - fac["RF"].reindex(mser.index)).dropna(), fac)
    print("\nby year, clone minus universe (quarterly, annualised):")
    for yr, gy in t.groupby(t.index.year):
        print(f"  {yr}: {((gy.port - gy.universe).mean() * 4):>+6.1%}/yr over {len(gy)} quarters; avg names {gy.n.mean():.0f}")
    out[best].to_csv(ROOT / "data/out/clone13f_quarterly.csv")
    print("CLONE DONE")


if __name__ == "__main__":
    main()
