#!/usr/bin/env python3
"""Head-to-head with Bertomeu, Cheynel, Floyd & Pan (2021) on the same firm-years and the same label.

They predict Item 4.02 non-reliance filings -- our label -- with paid data
(Compustat, CRSP, Audit Analytics) and a gradient booster, and publish their
out-of-sample scores for 2011-2019 by gvkey and fiscal year. Linking gvkey to
CIK (the farr table) puts their score and ours on the same 10-Ks.

The overlap with our test period is the 10-Ks for fiscal 2018 and 2019 (filed
2019-2020). On those, both models are out-of-sample: theirs trained on
2001-2010, ours on filings through 2018. Same rows, same label, paired.

  python bertomeu_compare.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from edge import impute
from events_test import joined
from asof_test import brf   # n_jobs capped by ASOF_JOBS so the extraction keeps its cores
from final import TEST_START, ci, paired
from relabel import submissions


def rank(v):
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    subs = submissions()
    link = pd.read_csv("data/raw/bao/gvkey_ciks.csv")
    link["gvkey"] = pd.to_numeric(link.gvkey, errors="coerce")
    link = link.dropna(subset=["gvkey"])
    g2c = {}
    for gv, c in zip(link.gvkey.astype(int), link.cik.astype(int)):
        g2c.setdefault(gv, str(c))
    theirs = pd.read_csv("data/raw/bertomeu/oos_2011_2019.csv")
    theirs["cik"] = theirs.gvkey.map(g2c)
    theirs = theirs.dropna(subset=["cik"])
    key = {(r.cik, int(r.fyear)): float(r.pred1) for r in theirs.itertuples()}
    print(f"their out-of-sample scores: {len(theirs):,} firm-years with a CIK link")

    # our filings keyed by (cik, fiscal period year)
    import csv, io, zipfile
    from pathlib import Path as _P
    pend_month = {}
    for zp in sorted(_P("data/raw").glob("*q?.zip")):
        with zipfile.ZipFile(zp).open("sub.txt") as fh:
            for r in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="	"):
                if r["period"]:
                    pend_month[r["adsh"]] = int(r["period"][4:6])
    ours_idx, their_score = [], []
    for i, a in enumerate(adsh.tolist()):
        cik, f, pyear = subs.get(a, (None, None, None))
        if cik is None or pyear is None:
            continue
        fyear = int(pyear) - 1 if pend_month.get(a, 12) <= 5 else int(pyear)     # Compustat convention
        s = key.get((cik, fyear))
        if s is not None:
            ours_idx.append(i); their_score.append(s)
    ours_idx = np.array(ours_idx); their_score = np.array(their_score)
    years = filed[ours_idx].astype("datetime64[Y]").astype(int) + 1970
    print(f"matched to {len(ours_idx):,} of our 10-Ks (fiscal 2011-2019); {int(y[ours_idx].sum())} restated")
    print("their AUC on our label, by our filing year:")
    for yr in sorted(set(years)):
        m = years == yr
        if 0 < y[ours_idx][m].sum() < m.sum():
            print(f"  {yr}: n {m.sum():>5,}  restated {int(y[ours_idx][m].sum()):>4}  their AUC {roc_auc_score(y[ours_idx][m], their_score[m]):.3f}")

    # our full stack on the overlap inside our test period
    R, _ = joined(adsh, "data/out/features_raw.npz"); R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    E, _ = joined(adsh, "data/out/features_events.npz"); I, _ = joined(adsh, "data/out/features_insider.npz")
    from events_test import market_block, scrutiny_block
    M = market_block(adsh)                                              # audited block, imputed without indicators, kept last
    L, _ = joined(adsh, "data/out/features_letters.npz"); S = scrutiny_block(adsh, os.environ.get("SCRUTINY_NPZ", "data/out/features_scrutiny.npz"))
    y_tight = np.load("data/out/features_tight.npz", allow_pickle=True)["y"]
    from asof_test import asof_labels
    y_asof = asof_labels(adsh, filed)(TEST_START)     # training labels as known on the cutoff
    full = np.hstack([fin, E, I, L, S, M])
    for label_name, yy in (("loose label: any Item 4.02 within 3 years", y), ("tight label: the 10-K's own year is later condemned", y_tight)):
      print(f"\n################ {label_name} ################")
      for name, X, train_to, ytr in (("statements only (242)", fin, TEST_START, yy), ("full free stack (295)", full, TEST_START, yy),
                                     ("full stack, trained on 2014-16 only", full, np.datetime64("2017-01-01"), yy),
                                     ("full stack, labels as known 2019-01-01", full, TEST_START, y_asof)):
        y = yy
        trn, tst = filed < train_to, filed >= TEST_START
        Xtr, Xte = impute(X[trn], X[tst], plain_last=M.shape[1] if X.shape[1] > 242 else 0)
        ours = np.mean([brf(s).fit(Xtr, ytr[trn]).predict_proba(Xte)[:, 1] for s in range(5)], axis=0)
        test_pos = {i: k for k, i in enumerate(np.flatnonzero(tst))}
        both = [(test_pos[i], t) for i, t in zip(ours_idx, their_score) if i in test_pos]
        oi = np.array([b[0] for b in both]); ts = np.array([b[1] for b in both])
        yt = y[tst][oi]; os_ = ours[oi]
        print(f"\n=== overlap inside our test period: {len(oi):,} 10-Ks (fiscal 2018-2019, filed 2019-2020), {int(yt.sum())} restated ===")
        print(f"  {name:<24} AUC {roc_auc_score(yt, os_):.3f}  [{ci(yt, os_)[0]:.3f}-{ci(yt, os_)[1]:.3f}]")
        print(f"  {'Bertomeu et al. (paid)':<24} AUC {roc_auc_score(yt, ts):.3f}  [{ci(yt, ts)[0]:.3f}-{ci(yt, ts)[1]:.3f}]")
        m, (lo, hi), p = paired(yt, ts, os_)
        print(f"  ours minus theirs: {m:+.3f} [{lo:+.3f},{hi:+.3f}]  P(<=0) {p:.3f}")
        blend = 0.5 * rank(os_) + 0.5 * rank(ts)
        m2, (lo2, hi2), p2 = paired(yt, os_, blend)
        print(f"  blend of both  AUC {roc_auc_score(yt, blend):.3f}   vs ours {m2:+.3f} [{lo2:+.3f},{hi2:+.3f}]")


if __name__ == "__main__":
    main()
