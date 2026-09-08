#!/usr/bin/env python3
"""Two feature blocks the single-filing view cannot see.

**History.** A filing is currently scored in isolation, but companies file every
year and the sequence carries what a snapshot cannot: whether margins are
drifting, whether growth is erratic, whether assets have outrun revenue for
three years running -- and, most directly, whether this company has restated
before. Everything here is built strictly from filings dated before the one
being scored, so nothing leaks backwards from the future.

**Industry.** Every ratio so far is absolute, and absolutes mean different
things in different industries: a 40% gross margin is unremarkable in software
and alarming in grocery retail. Each feature is re-expressed as a percentile
within its two-digit SIC group and filing year, so the model sees "unusual for
this industry, this year" rather than a raw level. The percentile is computed
from the training period only -- letting the test period define its own norms
would leak information about it.
"""

from collections import defaultdict

import numpy as np

HIST_VARS = [
    "prior_restatement", "prior_restatement_count", "years_filed",
    "roa_trend", "roa_vol", "margin_vol", "rev_growth_vol",
    "assets_vs_rev_3y", "accrual_persistence", "leverage_trend",
]


def build_history(rows, label_dates):
    """Attach per-filing history, using only what preceded each filing.

    rows must carry cik, filed, and the feature dict; label_dates is
    cik -> Item 4.02 announcement dates, used for prior-restatement only when
    the announcement predates the filing being scored.
    """
    by_cik = defaultdict(list)
    for r in rows:
        by_cik[r["cik"]].append(r)
    for cik in by_cik:
        by_cik[cik].sort(key=lambda r: r["filed"])

    for cik, seq in by_cik.items():
        past_dates = sorted(label_dates.get(cik, []))
        for i, r in enumerate(seq):
            prev = seq[:i]                      # strictly earlier filings only
            h = {k: None for k in HIST_VARS}
            earlier = [d for d in past_dates if d < r["filed"]]
            h["prior_restatement"] = 1.0 if earlier else 0.0
            h["prior_restatement_count"] = float(len(earlier))
            h["years_filed"] = float(len(prev))

            def series(key):
                out = []
                for p in prev[-4:]:
                    v = p["feat"].get(key)
                    if v is not None and np.isfinite(v):
                        out.append(v)
                return out

            roa = series("roa") + ([r["feat"]["roa"]]
                                   if r["feat"].get("roa") is not None else [])
            if len(roa) >= 3:
                h["roa_trend"] = float(np.polyfit(range(len(roa)), roa, 1)[0])
                h["roa_vol"] = float(np.std(roa))
            for src, dst in (("net_margin", "margin_vol"),
                             ("rev_growth", "rev_growth_vol"),
                             ("accruals_ta", "accrual_persistence")):
                s = series(src)
                if len(s) >= 3:
                    h[dst] = float(np.std(s)) if dst != "accrual_persistence" \
                        else float(np.mean(s))
            lev = series("leverage")
            if len(lev) >= 3:
                h["leverage_trend"] = float(np.polyfit(range(len(lev)), lev, 1)[0])

            ag = series("asset_growth")
            rg = series("rev_growth")
            if len(ag) >= 2 and len(rg) >= 2:
                n = min(len(ag), len(rg))
                # Assets outrunning revenue for years is the classic shape of
                # capitalising what should have been expensed.
                h["assets_vs_rev_3y"] = float(np.mean(ag[-n:]) - np.mean(rg[-n:]))
            r["hist"] = h
    for r in rows:
        r.setdefault("hist", {k: None for k in HIST_VARS})
    return rows


def industry_percentiles(rows, train_rows, keys):
    """Re-express each feature as its percentile within SIC-2 and filing year.

    Reference distributions come from the training rows only.
    """
    ref = defaultdict(lambda: defaultdict(list))
    for r in train_rows:
        g = (r.get("sic2"), r["filed"].year)
        for k in keys:
            v = r["feat"].get(k)
            if v is not None and np.isfinite(v):
                ref[g][k].append(v)
    # A thin industry-year falls back to the whole year.
    year_ref = defaultdict(lambda: defaultdict(list))
    for r in train_rows:
        for k in keys:
            v = r["feat"].get(k)
            if v is not None and np.isfinite(v):
                year_ref[r["filed"].year][k].append(v)
    for g in ref:
        for k in ref[g]:
            ref[g][k].sort()
    for y in year_ref:
        for k in year_ref[y]:
            year_ref[y][k].sort()

    for r in rows:
        g = (r.get("sic2"), r["filed"].year)
        pct = {}
        for k in keys:
            v = r["feat"].get(k)
            pool = ref[g].get(k) or year_ref[r["filed"].year].get(k)
            if v is None or not np.isfinite(v) or not pool or len(pool) < 20:
                pct[k] = None
            else:
                pct[k] = float(np.searchsorted(pool, v) / len(pool))
        r["pct"] = pct
    return rows
