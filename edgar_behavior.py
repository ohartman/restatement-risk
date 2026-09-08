#!/usr/bin/env python3
"""Filing behaviour for Bao et al.'s firm-years, from EDGAR's free quarterly indices.

Our single most important feature was how a company files, not what it files:
days from year-end to the 10-K, filer class, amendments. Bao et al.'s Compustat
panel has none of that. EDGAR's full-index files (1993 on) list every filing
with form type, CIK and date, so the same behaviour can be reconstructed for
their firm-years -- linked gvkey -> CIK through the farr package's table.

Compustat's fyear has no fiscal-year-end date, so the lag is measured against
the firm itself: the 10-K's day-of-year relative to the firm's own median across
its years. Everything is from filings dated on or before that 10-K.

  doy_10k          day of year the fiscal-year 10-K was filed (Jan-May shifted +365 so late Dec filers and Feb filers sort sensibly)
  lag_vs_median    days later than the firm's own median 10-K timing
  nt_10k           an NT 10-K (formal late-filing notice) within 120 days before the 10-K
  n_nt_3y          NT 10-K / NT 10-Q notices in the prior three years
  n_10ka_3y        10-K/A amendments in the prior three years
  n_10qa_3y        10-Q/A amendments in the prior three years
  n_8k_1y          8-Ks in the prior year
  n_filings_1y     everything filed in the prior year
  firm_age         years since the firm's first EDGAR filing
  no_10k_found     1 if no 10-K could be matched to the fiscal year (kept, with the rest NaN)

  python edgar_behavior.py    -> data/raw/bao/edgar_behavior.csv  (gvkey, fyear, cik, features...)
"""

import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

IDX = Path("data/raw/edgar_index")
BAO = Path("data/raw/bao")
# Annual reports: 10-K and its variants, EDGAR's unhyphenated small-business codes (10KSB),
# and the 20-F / 40-F that foreign private issuers file instead of a 10-K.
TENK_FORMS = ("10-K", "10-K405", "10-KSB", "10-KSB40", "10KSB", "10KSB40", "10-KT", "10-KT405",
              "10KT405", "10-KSB405", "20-F", "40-F", "10-K405/A"[:0] or "20-F/A"[:0] or "10-K")
NAMES = ["doy_10k", "lag_vs_median", "nt_10k", "n_nt_3y", "n_10ka_3y", "n_10qa_3y", "n_8k_1y",
         "n_filings_1y", "firm_age", "no_10k_found"]


LINE = re.compile(r"^(\S.*?)\s{2,}(\S.*?)\s{2,}(\d+)\s+(\d{4}-\d{2}-\d{2})\s+\S+\s*$")


def parse_index(path):
    """Yield (form, cik, date) from a form.idx. The header and the rows are not
    aligned to the same columns, so rows are parsed by their own structure:
    form type, two-plus spaces, company name, two-plus spaces, CIK, date, file."""
    with path.open(encoding="latin-1") as f:
        for l in f:
            m = LINE.match(l.rstrip("\r\n"))
            if m:
                yield m.group(1).strip(), int(m.group(3)), date.fromisoformat(m.group(4))


def main():
    filings = defaultdict(list)
    for p in sorted(IDX.glob("form_*.idx")):
        n = 0
        for form, cik, d in parse_index(p):
            filings[cik].append((d, form)); n += 1
        print(f"  {p.name}: {n:,} filings", flush=True)
    for v in filings.values():
        v.sort()
    print(f"{sum(len(v) for v in filings.values()):,} filings for {len(filings):,} CIKs")

    link = pd.read_csv(BAO / "gvkey_ciks.csv")
    link["gvkey"] = pd.to_numeric(link.gvkey, errors="coerce")
    link = link.dropna(subset=["gvkey"])
    link["gvkey"] = link.gvkey.astype(int)
    link["first"] = pd.to_datetime(link.first_date, errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    link["last"] = pd.to_datetime(link.last_date, errors="coerce").fillna(pd.Timestamp("2100-01-01"))
    by_gvkey = {g: grp for g, grp in link.groupby("gvkey")}

    bao = pd.read_csv(BAO / "data_FraudDetection_JAR2020.csv", usecols=["gvkey", "fyear"])
    bao = bao[bao.fyear.between(1990, 2014)]          # the panel carries a 9999 sentinel in a few rows
    rows = []
    # usecols keeps the file's column order (fyear, gvkey), so select explicitly
    for gvkey, fyear in bao[["gvkey", "fyear"]].astype(int).itertuples(index=False):
        grp = by_gvkey.get(gvkey)
        if grp is None:
            continue
        # A gvkey can link to several CIKs (parent, subsidiaries, debt issuers). Try the ones
        # valid around the fiscal year first, then the rest, and keep whichever actually filed
        # a 10-K for that year -- Jun 1 Y to Dec 31 Y+1 under Compustat's fyear convention.
        mid = pd.Timestamp(int(fyear), 12, 31)
        near = grp[(grp["first"] <= mid + pd.Timedelta(days=365)) & (grp["last"] >= mid - pd.Timedelta(days=365))]
        order = list(dict.fromkeys(list(near.cik.astype(int)) + list(grp.cik.astype(int))))
        cik, fl, tenk, dates = order[0], None, [], []
        for c in order:
            f_ = filings.get(c)
            if not f_:
                continue
            d_ = [d for d, _ in f_]
            lo, hi = bisect_left(d_, date(fyear, 6, 1)), bisect_right(d_, date(fyear + 1, 12, 31))
            t_ = [(d, f) for d, f in f_[lo:hi] if f in TENK_FORMS]
            if t_:
                cik, fl, tenk, dates = c, f_, t_, d_
                break
            if fl is None:
                cik, fl, dates = c, f_, d_
        if not tenk:
            rows.append((gvkey, fyear, cik) + (np.nan,) * 9 + (1.0,))
            continue
        d10, _ = tenk[0]
        doy = d10.timetuple().tm_yday + (365 if d10.month <= 5 else 0)
        # the firm's own median 10-K timing over all its years
        all10 = [dd for dd, f in fl if f in TENK_FORMS]
        med = float(np.median([dd.timetuple().tm_yday + (365 if dd.month <= 5 else 0) for dd in all10]))
        before = fl[:bisect_right(dates, d10)]
        w1 = [x for x in before if x[0] >= d10 - timedelta(days=365)]
        w3 = [x for x in before if x[0] >= d10 - timedelta(days=1095)]
        w120 = [x for x in before if x[0] >= d10 - timedelta(days=120)]
        rows.append((gvkey, fyear, cik, float(doy), doy - med,
                     float(any(f.startswith("NT 10-K") for _, f in w120)),
                     float(sum(1 for _, f in w3 if f.startswith("NT 10"))),
                     float(sum(1 for _, f in w3 if f in ("10-K/A", "10-K405/A", "10-KSB/A", "10KSB/A", "20-F/A", "40-F/A"))),
                     float(sum(1 for _, f in w3 if f in ("10-Q/A", "10QSB/A"))),
                     float(sum(1 for _, f in w1 if f.startswith("8-K"))),
                     float(len(w1)), (d10 - dates[0]).days / 365.25, 0.0))
    out = pd.DataFrame(rows, columns=["gvkey", "fyear", "cik"] + NAMES)
    out.to_csv(BAO / "edgar_behavior.csv", index=False)
    print(f"\n{len(out):,} of {len(bao):,} firm-years linked; 10-K found for {(out.no_10k_found == 0).mean():.1%} of them")
    print(out[NAMES].describe().T[["count", "mean", "50%"]].round(2))


if __name__ == "__main__":
    main()
