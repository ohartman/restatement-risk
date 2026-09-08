#!/usr/bin/env python3
"""Close the two look-ahead channels the code audit found in the headline stack, and
report firm-clustered confidence intervals.

  1. Market block, missingness. Yahoo drops delisted tickers, so "no prices" for an old
     filing means "delisted by the download date" - a future fact - and the imputer's
     missing indicators hand it to the forest. Fix: the market columns are median-filled
     WITHOUT missing indicators.
  2. Market block, split adjustment. Yahoo closes are adjusted for splits after the window,
     so price level, market cap, book-to-market and turnover of a filing can be rewritten by
     a later reverse split. Fix: those four columns are dropped; the within-window ratios
     (returns, volatility, drawdown, negative-return days) stay.
  3. History column industry_wave_rate divides by a whole-panel firm count. Dropped.
  4. The bootstrap resampled filings; a firm's several 10-Ks move together. Firm-clustered
     intervals are reported beside the filing-level ones.

  BRF_STRATEGY=0.1 BRF_MAX_FEATURES=15 SCRUTINY_NPZ=... FLAGS2_NPZ=... python leakfix_test.py
"""
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from edge import impute
from events_test import brf, joined
from final import TEST_START, ci, ci_cluster, paired, paired_cluster
from relabel import submissions

LEVEL = {"turnover", "log_mcap", "btm", "penny", "price_cov"}


def impute_split(Xa, Xm, trn, tst):
    """Xa: columns that get missing indicators; Xm: columns median-filled without them."""
    A_tr, A_te = impute(Xa[trn], Xa[tst])
    med = np.nanmedian(Xm[trn], axis=0); med = np.where(np.isnan(med), 0.0, med)
    M_tr = np.where(np.isnan(Xm[trn]), med, Xm[trn]); M_te = np.where(np.isnan(Xm[tst]), med, Xm[tst])
    return np.hstack([A_tr, M_tr]), np.hstack([A_te, M_te])


def fit(Xa, Xm, y, trn, tst, seeds=5):
    Xtr, Xte = impute_split(Xa, Xm, trn, tst) if Xm is not None else impute(Xa[trn], Xa[tst])
    return np.mean([brf(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    subs = submissions()
    cik = np.array([subs.get(a, ("?",))[0] or "?" for a in adsh.tolist()])
    R, _ = joined(adsh, "data/out/features_raw.npz")
    R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    E, _ = joined(adsh, "data/out/features_events.npz")
    I, _ = joined(adsh, "data/out/features_insider.npz")
    M, mn = joined(adsh, "data/out/features_market.npz")
    L, _ = joined(adsh, "data/out/features_letters.npz")
    S, sn = joined(adsh, os.environ["SCRUTINY_NPZ"])
    F, _ = joined(adsh, os.environ["FLAGS2_NPZ"])
    M_old = M[:, [i for i, n in enumerate(mn) if n != "price_cov"]]
    M_safe = M[:, [i for i, n in enumerate(mn) if n not in LEVEL]]
    S_safe = S[:, [i for i, n in enumerate(sn) if n != "industry_wave_rate"]]
    trn, tst = filed < TEST_START, filed >= TEST_START
    yt, ty, ct = y[tst], years[tst], cik[tst]
    print(f"test firms {len(set(ct)):,} for {int(tst.sum()):,} filings; market rows with prices: {np.isfinite(M[:, 0]).mean():.0%}\n", flush=True)

    variants = [
        ("current headline stack (330), filing-level CI", np.hstack([fin, E, I, M_old, L, S, F]), None),
        ("market block median-filled, no missing indicators", np.hstack([fin, E, I, L, S, F]), M_old),
        ("+ price-level columns dropped (turnover, mcap, btm, penny)", np.hstack([fin, E, I, L, S, F]), M_safe),
        ("+ industry_wave_rate dropped  (the clean stack)", np.hstack([fin, E, I, L, S_safe, F]), M_safe),
        ("no market block at all", np.hstack([fin, E, I, L, S_safe, F]), None),
        ("statements only (242), for reference", fin, None),
    ]
    print(f"{'variant':<62}{'test AUC':>9}{'filing CI':>17}{'firm-clustered CI':>20}{'   vs current (paired, clustered)':>34}")
    base = None
    for name, Xa, Xm in variants:
        st = fit(Xa, Xm, y, trn, tst)
        lo, hi = ci(yt, st); clo, chi = ci_cluster(yt, st, ct)
        line = f"{name:<62}{roc_auc_score(yt, st):>9.3f}   [{lo:.3f}-{hi:.3f}]     [{clo:.3f}-{chi:.3f}]"
        if base is None:
            base = st
        else:
            m, (dlo, dhi), p = paired_cluster(yt, base, st, ct)
            line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]  P {p:.3f}"
        print(line + "   " + " ".join(f"{yr}:{roc_auc_score(yt[ty == yr], st[ty == yr]):.3f}" for yr in sorted(set(ty))), flush=True)
    print("LEAKFIX DONE")


if __name__ == "__main__":
    main()
