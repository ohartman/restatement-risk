#!/usr/bin/env python3
"""Test one feature block on top of the full stack: python block_test.py data/out/features_X.npz"""
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from edge import impute
from events_test import brf, joined
from final import TEST_START, ci, paired
VAL_YEARS = (2017, 2018)
PLAIN = 0


def fit_score(X, y, trn, tst, seeds=5):
    Xtr, Xte = impute(X[trn], X[tst], plain_last=PLAIN)
    return np.mean([brf(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)
z = np.load("data/out/features.npz", allow_pickle=True)
adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
years = filed.astype("datetime64[Y]").astype(int) + 1970
R, _ = joined(adsh, "data/out/features_raw.npz"); R2, n2 = joined(adsh, "data/out/features_raw2.npz")
fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
E, _ = joined(adsh, "data/out/features_events.npz"); I, _ = joined(adsh, "data/out/features_insider.npz")
from events_test import market_block, scrutiny_block
M = market_block(adsh)                                                   # audited: ratios only, imputed without indicators, kept last
L, _ = joined(adsh, "data/out/features_letters.npz"); S = scrutiny_block(adsh, os.environ.get("SCRUTINY_NPZ", "data/out/features_scrutiny.npz"))
F, _ = joined(adsh, os.environ.get("FLAGS2_NPZ", "data/out/features_flags2_ec.npz"))
B, bn = joined(adsh, sys.argv[1])
stack = np.hstack([fin, E, I, L, S, F]); PLAIN = M.shape[1]      # the headline stack; M appended last at fit time (plain-imputed)
# --all-rows: keep every flagged row and let the block be NaN where it does not exist (for blocks whose
# coverage starts in a later year, e.g. Form AP from 2017); otherwise only rows that have the block
rows = np.isfinite(S[:, 0]) if "--all-rows" in sys.argv else (np.isfinite(S[:, 0]) & np.isfinite(B).any(axis=1))
trn, tst = rows & (filed < TEST_START), rows & (filed >= TEST_START)
yt = y[tst]; ty = years[tst]
print(f"rows with flags and this block: train {trn.sum():,} ({int(y[trn].sum())})  test {tst.sum():,} ({int(yt.sum())})  years {sorted(set(ty))}")
print(f"{'stage':<26}{'n':>5}{'val 2017':>10}{'val 2018':>10}{'test AUC':>10}{'95% CI':>17}{'  vs previous (paired)':>26}{'P(<=0)':>8}")
prev = None
for name, X in (("full stack", np.hstack([stack, M])), ("+ " + sys.argv[1].split("features_")[-1].replace(".npz", ""), np.hstack([stack, B, M]))):
    vals = [roc_auc_score(y[rows & (years == v)], fit_score(X, y, rows & (years < v), rows & (years == v), seeds=3)) for v in VAL_YEARS]
    st = fit_score(X, y, trn, tst); lo, hi = ci(yt, st)
    line = f"{name:<26}{X.shape[1]:>5}{vals[0]:>10.3f}{vals[1]:>10.3f}{roc_auc_score(yt, st):>10.3f}   [{lo:.3f}-{hi:.3f}]"
    if prev is not None:
        m, (dlo, dhi), p = paired(yt, prev, st); line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>12.3f}"
        print(line); print("per year: " + "  ".join(f"{yr}: {roc_auc_score(yt[ty==yr], prev[ty==yr]):.3f}->{roc_auc_score(yt[ty==yr], st[ty==yr]):.3f}" for yr in sorted(set(ty))))
    else:
        print(line, flush=True)
    prev = st
Xtr, _ = impute(np.hstack([stack, B])[trn], np.hstack([stack, B])[tst])
imp = brf(0).fit(Xtr, y[trn]).feature_importances_[stack.shape[1]:stack.shape[1] + B.shape[1]]
print("block features by importance: " + ", ".join(f"{bn[i]} ({imp[i]:.4f})" for i in np.argsort(-imp)))
