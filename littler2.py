#!/usr/bin/env python3
"""Tightening the little-r label until it means what it says.

littler.py's first cut -- any of net income, assets, liabilities, revenue or
equity off by more than 1% between a 10-K and the next year's comparatives --
flags 11.7% of filings, three times the Item 4.02 rate, and is only weakly
predictable (0.62). Firms that later filed an Item 4.02 do show mismatches far
more often (29.8% vs 10.8%), so the label points the right way; the 10.8% among
everyone else is retrospective standard adoptions, discontinued operations,
mergers and tag changes, none of which is an error.

This tries stricter definitions and reports, for each: the base rate, how much
more often Item 4.02 filers trip it (the enrichment ratio -- the self-check),
and how predictable it is from the financial features, both by the model
trained on Item 4.02 and by a model trained on the definition itself.

  python littler2.py
"""

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score

from edge import impute
from events_test import joined
from fscore import is_financial, read_quarter
from littler import RAW

ADOPTION_YEARS = {2017, 2018}      # comparatives most affected by ASC 606 (2018) and ASC 842 (2019) adoptions


def collect():
    own, comp, meta = {}, {}, {}
    for zp in sorted(RAW.glob("*q?.zip")):
        subs, facts = read_quarter(zp)
        for adsh, s in subs.items():
            if s["form"] != "10-K" or is_financial(s["sic"]) or not s["period"]:
                continue
            d = facts.get(adsh)
            if not d:
                continue
            P = s["period"]; prev = f"{int(P[:4]) - 1}{P[4:]}"
            cik = str(int(s["cik"]))
            meta[adsh] = (cik, P)
            if P in d:
                own[adsh] = d[P]
            if prev in d:
                comp[(cik, prev)] = d[prev]
        print(f"  {zp.stem}", flush=True)
    return own, comp, meta


def rel_diff(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(a), abs(b), 1.0), abs(a - b)


def label(own, comp, meta, adsh_list, items, tol, floor, skip_adoption):
    y = np.full(len(adsh_list), -1)
    for i, a in enumerate(adsh_list):
        m = meta.get(a)
        if m is None:
            continue
        cik, P = m
        cur, later = own.get(a), comp.get((cik, P))
        if cur is None or later is None:
            continue
        if skip_adoption and int(P[:4]) in ADOPTION_YEARS:
            continue
        flags, n = [], 0
        for k in items:
            if k == "equity":
                a_ = cur.get("assets") - cur.get("liabilities") if None not in (cur.get("assets"), cur.get("liabilities")) else None
                b_ = later.get("assets") - later.get("liabilities") if None not in (later.get("assets"), later.get("liabilities")) else None
            else:
                a_, b_ = cur.get(k), later.get(k)
            r = rel_diff(a_, b_)
            if r is None:
                continue
            n += 1
            flags.append(r[0] > tol and r[1] > floor)
        if n >= 1:
            y[i] = int(all(flags) if items == ("net_income", "equity") and len(flags) == 2 else any(flags))
    return y


def main():
    own, comp, meta = collect()
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y_big, filed = z["adsh"].tolist(), z["y"], z["filed"].astype("datetime64[D]")
    R, _ = joined(z["adsh"], "data/out/features_raw.npz")
    R2, n2 = joined(z["adsh"], "data/out/features_raw2.npz")
    X = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
    ts = np.datetime64("2019-01-01")

    def brf(seed):
        return BalancedRandomForestClassifier(n_estimators=400, min_samples_leaf=5, sampling_strategy="all",
                                              replacement=True, bootstrap=False, random_state=seed, n_jobs=-1)

    defs = [
        ("any of 5 items > 1%", ("net_income", "assets", "liabilities", "revenue", "equity"), 0.01, 1e5, False),
        ("any of 5 items > 5%", ("net_income", "assets", "liabilities", "revenue", "equity"), 0.05, 1e5, False),
        ("net income > 1%", ("net_income",), 0.01, 1e5, False),
        ("net income > 5%", ("net_income",), 0.05, 1e5, False),
        ("net income > 10%", ("net_income",), 0.10, 1e6, False),
        ("net income AND equity > 1%", ("net_income", "equity"), 0.01, 1e5, False),
        ("total assets > 1%", ("assets",), 0.01, 1e5, False),
        ("net income > 5%, skip 2017-18 periods", ("net_income",), 0.05, 1e5, True),
    ]
    print(f"\n{'definition':<40}{'n':>8}{'rate':>8}{'|4.02':>8}{'|not':>8}{'enrich':>8}"
          f"{'AUC(bigR-model)':>17}{'AUC(own)':>10}{'union->bigR':>13}")
    big_scores = None
    for name, items, tol, floor, skip in defs:
        y_r = label(own, comp, meta, adsh, items, tol, floor, skip)
        has = y_r >= 0
        rate = y_r[has].mean()
        r_big = y_r[has & (y_big == 1)].mean(); r_not = y_r[has & (y_big == 0)].mean()
        trn, tst = has & (filed < ts), has & (filed >= ts)
        Xtr, Xte = impute(X[trn], X[tst])
        if big_scores is None or big_scores[0] is not trn:
            s_big = np.mean([brf(k).fit(Xtr, y_big[trn]).predict_proba(Xte)[:, 1] for k in range(2)], axis=0)
        s_own = np.mean([brf(k).fit(Xtr, y_r[trn]).predict_proba(Xte)[:, 1] for k in range(2)], axis=0)
        y_u = np.maximum(y_r, y_big)
        s_u = np.mean([brf(k).fit(Xtr, y_u[trn]).predict_proba(Xte)[:, 1] for k in range(2)], axis=0)
        print(f"{name:<40}{has.sum():>8,}{rate:>8.2%}{r_big:>8.1%}{r_not:>8.1%}{r_big/max(r_not,1e-9):>8.1f}"
              f"{roc_auc_score(y_r[tst], s_big):>17.3f}{roc_auc_score(y_r[tst], s_own):>10.3f}"
              f"{roc_auc_score(y_big[tst], s_u):>13.3f}", flush=True)
    print("\ncolumns: rate = share flagged; |4.02 = flag rate among filings that later drew an Item 4.02; "
          "|not = among the rest; enrich = ratio (the self-check);\nAUC(bigR-model) = Item-4.02-trained forest "
          "scored on this label; AUC(own) = forest trained on this label; union->bigR = forest trained on "
          "(this OR 4.02), scored on 4.02 (0.730 without it)")


if __name__ == "__main__":
    main()
