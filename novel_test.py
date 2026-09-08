#!/usr/bin/env python3
"""Six ideas borrowed from how other models get trained, applied to the restatement forest.

Each targets a structural fact about this problem rather than swapping the learner:

  dedup      one Item 4.02 labels three consecutive 10-Ks; weight each positive by
             1/(10-Ks its announcement labels) so each EVENT counts once   (LLM data dedup)
  proximity  weight positives by exp(-days to announcement / 365): the filing just
             before the announcement is the misstated one                (label smoothing)
  pu         negatives the out-of-fold model finds suspicious get weight
             1 - 0.5*score: the clean class is not clean                  (positive-unlabeled learning)
  cascade    a second forest trained only on the top 30% of stage-1 scores
             re-ranks that region                                          (Viola-Jones cascades, hard-negative mining)
  calibrate  per-industry (2-digit SIC) isotonic calibration of out-of-fold
             scores, so industries interleave correctly in the global rank  (group calibration)
  windows    average of forests trained on 2014-18, 2015-18, 2016-18        (snapshot ensembles / model soups)

Validation: train on 2014-16, score 2017-18 (one AUC over both years). Test: train
on 2014-18, score 2019-23 once, paired against the plain forest. Same 242 features.

  python novel_test.py
"""

import csv
import io
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from edge import impute
from events_test import joined
from final import TEST_START, ci, paired
from relabel import submissions
from train import load_labels

VAL_START = np.datetime64("2017-01-01")
WINDOW = 1095


def brf(seed, n=400):
    return BalancedRandomForestClassifier(n_estimators=n, min_samples_leaf=5, sampling_strategy="all",
                                          replacement=True, bootstrap=False, random_state=seed, n_jobs=-1)


def fit_score(X, y, trn, tst, w=None, seeds=2):
    Xtr, Xte = impute(X[trn], X[tst])
    out = []
    for s in range(seeds):
        m = brf(s)
        m.fit(Xtr, y[trn], sample_weight=None if w is None else w[trn])
        out.append(m.predict_proba(Xte)[:, 1])
    return np.mean(out, axis=0)


def oof_scores(X, y, rows, groups, w=None, seeds=1):
    """Out-of-fold scores for the given rows, five folds by company."""
    Xi, _ = impute(X[rows], X[rows][:1])
    yy, gg = y[rows], groups[rows]
    oof = np.zeros(rows.sum())
    for f_tr, f_va in GroupKFold(5).split(Xi, yy, gg):
        preds = []
        for s in range(seeds):
            m = brf(s, 300)
            m.fit(Xi[f_tr], yy[f_tr], sample_weight=None if w is None else w[rows][f_tr])
            preds.append(m.predict_proba(Xi[f_va])[:, 1])
        oof[f_va] = np.mean(preds, axis=0)
    return oof


def rank(v):
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


def sic_by_adsh():
    out = {}
    for zp in sorted(Path("data/raw").glob("*q?.zip")):
        with zipfile.ZipFile(zp).open("sub.txt") as f:
            for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"), delimiter="\t"):
                out[r["adsh"]] = (r["sic"] or "0000")[:2]
    return out


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    R, _ = joined(adsh, "data/out/features_raw.npz")
    R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    X = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    subs = submissions()
    ciks = np.array([subs.get(a, ("",))[0] for a in adsh.tolist()])
    fdates = [subs[a][1] for a in adsh.tolist()]

    # --- event structure of the positives: which announcement labels each 10-K, and how far away it is ---
    labels = load_labels()
    days_to = np.full(len(adsh), np.nan)
    event_id = np.full(len(adsh), -1)
    events = {}
    for i, a in enumerate(adsh.tolist()):
        c = ciks[i]
        hits = [x for x in labels.get(c, []) if 0 <= (x - fdates[i]).days <= WINDOW]
        if hits:
            first = min(hits)
            days_to[i] = (first - fdates[i]).days
            event_id[i] = events.setdefault((c, first), len(events))
    n_per_event = np.bincount(event_id[event_id >= 0])
    print(f"{int(y.sum()):,} positive filings belong to {len(events):,} announcement events "
          f"(mean {y.sum()/len(events):.2f} 10-Ks per event); median days to announcement {np.nanmedian(days_to):.0f}")

    w_dedup = np.ones(len(adsh))
    w_dedup[event_id >= 0] = 1.0 / n_per_event[event_id[event_id >= 0]]
    w_prox = np.ones(len(adsh))
    w_prox[y == 1] = 0.25 + np.exp(-days_to[y == 1] / 365.0)          # floor keeps far filings in
    w_both = w_dedup * w_prox

    fm = filed < VAL_START; vm = (filed >= VAL_START) & (filed < TEST_START)
    trn, tst = filed < TEST_START, filed >= TEST_START
    yt = y[tst]
    print(f"fit {fm.sum():,} ({y[fm].sum()})  val {vm.sum():,} ({y[vm].sum()})  test {tst.sum():,} ({yt.sum()})\n")

    results = {}

    def report(name, sv, st):
        results[name] = st
        base = results["plain forest"]
        lo, hi = ci(yt, st)
        line = f"{name:<24}{roc_auc_score(y[vm], sv):>9.3f}{roc_auc_score(yt, st):>10.3f}   [{lo:.3f}-{hi:.3f}]{yt[np.argsort(-st)[:100]].mean():>8.1%}"
        if st is not base:
            m, (dlo, dhi), p = paired(yt, base, st)
            line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>10.3f}"
        print(line, flush=True)

    print(f"{'idea':<24}{'val AUC':>9}{'test AUC':>10}{'95% CI':>17}{'p@100':>8}{'  vs plain (paired)':>22}{'P(<=0)':>10}")
    sv0, st0 = fit_score(X, y, fm, vm), fit_score(X, y, trn, tst)
    report("plain forest", sv0, st0)

    # 1-3: reweighting
    for name, w in (("dedup events", w_dedup), ("proximity weights", w_prox), ("dedup + proximity", w_both)):
        report(name, fit_score(X, y, fm, vm, w), fit_score(X, y, trn, tst, w))

    # 3: positive-unlabeled -- down-weight suspicious negatives using out-of-fold scores
    oof_fit = oof_scores(X, y, fm, ciks); oof_trn = oof_scores(X, y, trn, ciks)
    w_pu_fit = np.ones(len(adsh)); w_pu_fit[np.flatnonzero(fm)[y[fm] == 0]] = 1 - 0.5 * rank(oof_fit)[y[fm] == 0]
    w_pu_trn = np.ones(len(adsh)); w_pu_trn[np.flatnonzero(trn)[y[trn] == 0]] = 1 - 0.5 * rank(oof_trn)[y[trn] == 0]
    report("PU: soft negatives", fit_score(X, y, fm, vm, w_pu_fit), fit_score(X, y, trn, tst, w_pu_trn))

    # 4: cascade -- stage 2 on the top 30% of stage-1 scores
    def cascade(rows_tr, oof, s1_te, rows_te):
        top_tr = np.flatnonzero(rows_tr)[rank(oof) >= 0.7]
        mask_tr = np.zeros(len(adsh), bool); mask_tr[top_tr] = True
        cut = np.quantile(s1_te, 0.7)
        top_te = s1_te >= cut
        mask_te = np.zeros(len(adsh), bool); mask_te[np.flatnonzero(rows_te)[top_te]] = True
        s2 = fit_score(X, y, mask_tr, mask_te)
        out = rank(s1_te) * 0.7
        out[top_te] = 0.7 + 0.3 * rank(s2)
        return out
    report("cascade top 30%", cascade(fm, oof_fit, sv0, vm), cascade(trn, oof_trn, st0, tst))

    # 5: per-industry isotonic calibration of scores
    sic = sic_by_adsh()
    grp = np.array([sic.get(a, "00") for a in adsh.tolist()])
    def calibrate(rows_tr, oof, s_te, rows_te):
        out = s_te.copy()
        te_idx = np.flatnonzero(rows_te)
        tr_idx = np.flatnonzero(rows_tr)
        glob = IsotonicRegression(out_of_bounds="clip").fit(oof, y[tr_idx])
        for g in np.unique(grp[te_idx]):
            in_tr = grp[tr_idx] == g
            in_te = grp[te_idx] == g
            if in_tr.sum() >= 300 and y[tr_idx][in_tr].sum() >= 15:
                iso = IsotonicRegression(out_of_bounds="clip").fit(oof[in_tr], y[tr_idx][in_tr])
                out[in_te] = iso.predict(s_te[in_te])
            else:
                out[in_te] = glob.predict(s_te[in_te])
        return out
    report("industry calibration", calibrate(fm, oof_fit, sv0, vm), calibrate(trn, oof_trn, st0, tst))

    # 6: training-window ensemble
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    sv = np.mean([rank(fit_score(X, y, fm & (years >= s), vm)) for s in (2014, 2015, 2016)], axis=0)
    st = np.mean([rank(fit_score(X, y, trn & (years >= s), tst)) for s in (2014, 2015, 2016)], axis=0)
    report("window ensemble", sv, st)

    np.savez("data/out/scores_novel.npz", adsh=adsh[tst], y=yt,
             **{k.replace(" ", "_").replace(":", "").replace("%", "").replace("+", "plus"): v for k, v in results.items()})


if __name__ == "__main__":
    main()
