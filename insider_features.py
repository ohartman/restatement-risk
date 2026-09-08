#!/usr/bin/env python3
"""Insider trading before each 10-K, from the SEC's Forms 3/4/5 bulk data sets.

Officers and directors must report their trades within two business days, and
the SEC publishes the lot as quarterly tab-separated files. Unlike stock prices
from a free quote service, these cover every filer, delisted or not, so there is
no survivorship hole.

For each 10-K, over the twelve months (and the last 90 days) before the filing:
  n_sales, n_buys           open-market sales (code S) and purchases (code P)
  sold_value, bought_value  dollars, log10
  net_sold_frac             (sold - bought) / (sold + bought), -1..1
  n_sellers                 distinct insiders who sold
  officer_sales             sales by people flagged Officer
  n_filings_90d             any Form 4 activity in the last quarter before filing
  disp_share_frac           shares disposed / shares held after, a "dumping" measure

  python insider_features.py   -> data/out/features_insider.npz (X, adsh, names)
"""

import io
import zipfile
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from relabel import submissions

RAW = Path("data/raw/insider")
NAMES = ["n_sales_12m", "n_buys_12m", "log_sold_12m", "log_bought_12m", "net_sold_frac",
         "n_sellers_12m", "officer_sales_12m", "n_sales_90d", "n_filings_90d", "disp_share_frac"]


def load_trades():
    parts = []
    for zp in sorted(RAW.glob("*_form345.zip")):
        z = zipfile.ZipFile(zp)
        sub = pd.read_csv(io.BytesIO(z.read("SUBMISSION.tsv")), sep="\t",
                          usecols=["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK"], dtype=str)
        own = pd.read_csv(io.BytesIO(z.read("REPORTINGOWNER.tsv")), sep="\t",
                          usecols=["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"], dtype=str)
        tr = pd.read_csv(io.BytesIO(z.read("NONDERIV_TRANS.tsv")), sep="\t",
                         usecols=["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                  "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS"],
                         dtype={"ACCESSION_NUMBER": str, "TRANS_CODE": str, "TRANS_ACQUIRED_DISP_CD": str},
                         low_memory=False)
        tr = tr[tr.TRANS_CODE.isin(["S", "P"])]
        own = own.drop_duplicates("ACCESSION_NUMBER")
        own["officer"] = own.RPTOWNER_RELATIONSHIP.fillna("").str.contains("Officer", case=False)
        df = tr.merge(sub, on="ACCESSION_NUMBER", how="inner").merge(
            own[["ACCESSION_NUMBER", "RPTOWNERCIK", "officer"]], on="ACCESSION_NUMBER", how="left")
        df["filed"] = pd.to_datetime(df.FILING_DATE, format="%d-%b-%Y", errors="coerce")
        df["cik"] = df.ISSUERCIK.str.lstrip("0")
        df["shares"] = pd.to_numeric(df.TRANS_SHARES, errors="coerce").fillna(0).abs()
        df["price"] = pd.to_numeric(df.TRANS_PRICEPERSHARE, errors="coerce").fillna(0)
        df["held"] = pd.to_numeric(df.SHRS_OWND_FOLWNG_TRANS, errors="coerce")
        df["value"] = df.shares * df.price
        parts.append(df[["cik", "filed", "TRANS_CODE", "shares", "value", "held", "RPTOWNERCIK", "officer"]]
                     .dropna(subset=["filed"]))
        print(f"  {zp.stem}: {len(df):,} open-market trades", flush=True)
    return pd.concat(parts, ignore_index=True).sort_values(["cik", "filed"])


def main():
    trades = load_trades()
    print(f"\n{len(trades):,} open-market insider trades across {trades.cik.nunique():,} issuers")
    by = {c: g.reset_index(drop=True) for c, g in trades.groupby("cik")}

    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y = z["adsh"], z["y"]
    subs = submissions()
    X = np.full((len(adsh), len(NAMES)), np.nan)
    for i, a in enumerate(adsh.tolist()):
        cik, filed, _ = subs.get(a, (None, None, None))
        if cik is None:
            continue
        g = by.get(cik)
        end = pd.Timestamp(filed) - pd.Timedelta(days=1)
        if g is None:
            X[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, np.nan]
            continue
        dates = g.filed.values.astype("datetime64[ns]")
        lo = bisect_left(dates, np.datetime64(end - pd.Timedelta(days=365)))
        hi = bisect_right(dates, np.datetime64(end))
        w = g.iloc[lo:hi]
        lo90 = bisect_left(dates, np.datetime64(end - pd.Timedelta(days=90)))
        w90 = g.iloc[lo90:hi]
        s, b = w[w.TRANS_CODE == "S"], w[w.TRANS_CODE == "P"]
        sold, bought = s.value.sum(), b.value.sum()
        held = s.held.sum()
        X[i] = [len(s), len(b), np.log10(1 + sold), np.log10(1 + bought),
                (sold - bought) / (sold + bought) if (sold + bought) > 0 else 0.0,
                s.RPTOWNERCIK.nunique(), int(s.officer.fillna(False).sum()),
                int((w90.TRANS_CODE == "S").sum()), len(w90),
                s.shares.sum() / (s.shares.sum() + held) if (s.shares.sum() + held) > 0 else np.nan]
    Path("data/out").mkdir(exist_ok=True)
    np.savez("data/out/features_insider.npz", X=X, adsh=adsh, names=np.array(NAMES))
    any_sale = X[:, 0] > 0
    print(f"\n{len(adsh):,} filings; {any_sale.sum():,} ({100*any_sale.mean():.0f}%) had insider sales in the prior year")
    print(f"restatement rate with insider sales {y[any_sale].mean():.2%}   without {y[~any_sale].mean():.2%}")
    for k, n in enumerate(NAMES):
        v = X[:, k]; ok = np.isfinite(v)
        from sklearn.metrics import roc_auc_score
        print(f"  {n:<20} coverage {100*ok.mean():3.0f}%  AUC alone {roc_auc_score(y[ok], v[ok]):.3f}")


if __name__ == "__main__":
    main()
