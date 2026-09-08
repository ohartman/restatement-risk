#!/usr/bin/env python3
"""The morning headline: every block, every test filing, once.

After fetch_9a.py has filled the Item 9A / 8 / 3 flags for the 2021-2023 test
filings, this merges them into the scrutiny block and runs the full stack on
the whole 17,755-filing test set -- the same rows as every headline before it.
Stages are added in the order they were discovered; each is paired against the
previous. Precision at 100, 500 and 1,000 are all shown, because 100 is a noisy
count.

  python headline_test.py
"""

import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from edge import impute
from events_test import brf, joined
from final import TEST_START, ci, paired

VAL_YEARS = (2017, 2018)


PLAIN = [0]          # columns at the right edge of every stage that are imputed without indicators


def fit_score(X, y, trn, tst, seeds=5):
    Xtr, Xte = impute(X[trn], X[tst], plain_last=PLAIN[0] if X.shape[1] > 242 else 0)
    return np.mean([brf(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)


def merge_flags():
    z = np.load("data/out/features_scrutiny.npz", allow_pickle=True)
    X, adsh, names = z["X"], z["adsh"], z["names"]
    pos = {a: i for i, a in enumerate(adsh.tolist())}
    p = Path("data/raw/mdna/flags_9a.jsonl")
    n = 0
    if p.exists():
        for l in p.open(encoding="utf-8"):
            r = json.loads(l)
            i = pos.get(r["adsh"])
            if i is not None and r.get("flags") and not np.isfinite(X[i, 0]):
                X[i, :10] = r["flags"]; n += 1
        np.savez("data/out/features_scrutiny.npz", X=X, adsh=adsh, names=names)
    print(f"merged {n:,} fetched flag rows; text flags now on {int(np.isfinite(X[:, 0]).sum()):,} filings")
    return X


def main():
    S_all = merge_flags() if not os.environ.get("SCRUTINY_NPZ") else None
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    R, _ = joined(adsh, "data/out/features_raw.npz")
    R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    E, _ = joined(adsh, "data/out/features_events.npz")
    I, _ = joined(adsh, "data/out/features_insider.npz")
    from events_test import market_block, scrutiny_block
    M = market_block(adsh); PLAIN[0] = M.shape[1]                      # audited block, imputed without indicators, kept last
    L, _ = joined(adsh, "data/out/features_letters.npz")
    S = scrutiny_block(adsh, os.environ.get("SCRUTINY_NPZ", "data/out/features_scrutiny.npz"))
    # thirty more disclosure flags, if built; fill test rows from saved documents first
    F = None
    f2_path = os.environ.get("FLAGS2_NPZ", "data/out/features_flags2.npz")
    if Path(f2_path).exists():
        if not os.environ.get("FLAGS2_NPZ"):
            import flags2_features
            flags2_features.extend()
        F, _ = joined(adsh, f2_path)
    trn, tst = filed < TEST_START, filed >= TEST_START
    yt = y[tst]; ty = years[tst]
    has = np.isfinite(S[:, 0])
    print(f"test filings with text flags: {int((tst & has).sum()):,} of {int(tst.sum()):,}  "
          f"(restated among those without: {y[tst & ~has].mean():.2%} vs with: {y[tst & has].mean():.2%})\n")

    stages = [("financial statements only (242)", fin),
              ("+ 8-K events, insider, market (270)", np.hstack([fin, E, I, M])),
              ("+ SEC letters and late notices (280)", np.hstack([fin, E, I, L, M])),
              ("+ 10-K controls / auditor / legal flags (295)", np.hstack([fin, E, I, L, S, M]))]
    if F is not None:
        stages.append(("+ thirty more disclosure flags", np.hstack([fin, E, I, L, S, F, M])))
    c_path = os.environ.get("CHANGE_NPZ", "data/out/features_change.npz")
    if Path(c_path).exists():
        C, _ = joined(adsh, c_path)
        stages.append(("+ year-over-year change block", np.hstack([fin, E, I, L, S, F, C, M]) if F is not None else np.hstack([fin, E, I, L, S, C, M])))
    print(f"{'stage':<50}{'val 2017':>10}{'val 2018':>10}{'test AUC':>10}{'95% CI':>17}{'p@100':>7}{'p@500':>7}{'p@1000':>8}"
          f"{'  vs previous (paired)':>26}{'P(<=0)':>8}")
    prev = None
    out = {}
    for name, X in stages:
        vals = [roc_auc_score(y[years == v], fit_score(X, y, years < v, years == v, seeds=3)) for v in VAL_YEARS]
        st = fit_score(X, y, trn, tst)
        out[name] = st
        lo, hi = ci(yt, st)
        order = np.argsort(-st)
        line = (f"{name:<50}{vals[0]:>10.3f}{vals[1]:>10.3f}{roc_auc_score(yt, st):>10.3f}   [{lo:.3f}-{hi:.3f}]"
                f"{yt[order[:100]].mean():>7.0%}{yt[order[:500]].mean():>7.1%}{yt[order[:1000]].mean():>8.1%}")
        if prev is not None:
            m, (dlo, dhi), p = paired(yt, prev, st)
            line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>10.3f}"
        print(line, flush=True)
        prev = st
    first, last = out[stages[0][0]], out[stages[-1][0]]
    m, (dlo, dhi), p = paired(yt, first, last)
    print(f"\nfinancial-only -> full stack: {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]  P(<=0) {p:.3f}")
    print("per year: " + "  ".join(f"{yr}: {roc_auc_score(yt[ty==yr], first[ty==yr]):.3f}->{roc_auc_score(yt[ty==yr], last[ty==yr]):.3f}" for yr in sorted(set(ty))))
    np.savez("data/out/scores_headline_final.npz", adsh=adsh[tst], y=yt, filed=z["filed"][tst],
             **{f"s{k}": v for k, v in enumerate(out.values())})


if __name__ == "__main__":
    main()
