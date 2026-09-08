#!/usr/bin/env python3
"""Second leak audit: the label, firm memory across the split, and label maturity in the test years.
Uses the saved headline scores (data/out/scores_headline_final.npz, s4 = the 325-column stack).

  1. Announcement lag. A 10-K whose Item 4.02 arrives within days of its filing is not being
     predicted; the filing itself may describe the restatement. How many positives are that
     close, and what is the AUC without them?
  2. Firm memory. A 2018 (training) and a 2019 (test) 10-K of the same company can share one
     announcement. Score the test set with those firms removed, and with test positives whose
     event also labels a training row removed.
  3. Immature test labels. 2022-2023 filings have had less than three years for an
     announcement; their negatives are partly unlabelled positives.
"""
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score

from final import ci
from relabel import submissions
from train import load_labels

WINDOW = 1095


def main():
    z = np.load("data/out/scores_headline_final.npz", allow_pickle=True)
    adsh, y, s = z["adsh"], z["y"], z["s4"]
    filed = z["filed"].astype("datetime64[D]"); years = filed.astype("datetime64[Y]").astype(int) + 1970
    f = np.load("data/out/features.npz", allow_pickle=True)
    all_adsh, all_y, all_filed = f["adsh"], f["y"], f["filed"].astype("datetime64[D]")
    subs = submissions(); labels = load_labels()
    cik = np.array([subs.get(a, ("?",))[0] or "?" for a in adsh.tolist()])
    fd = [datetime.strptime(str(d), "%Y-%m-%d").date() for d in filed.astype(str)]
    print(f"test set: {len(adsh):,} filings, {int(y.sum())} positives, AUC {roc_auc_score(y, s):.3f}\n")

    # 1. announcement lag for test positives
    lag = np.full(len(adsh), np.nan)
    for i in range(len(adsh)):
        if y[i] == 1:
            d = [(x - fd[i]).days for x in labels.get(cik[i], []) if 0 <= (x - fd[i]).days <= WINDOW]
            if d: lag[i] = min(d)
    pos = y == 1
    print("1. days from the 10-K's filing to its first qualifying Item 4.02 (test positives):")
    for lo, hi in ((0, 0), (1, 30), (31, 90), (91, 365), (366, 730), (731, 1095)):
        m = pos & (lag >= lo) & (lag <= hi)
        print(f"   {lo:>4}-{hi:<4} days: {int(m.sum()):>4} positives  mean score {s[m].mean():.3f}" if m.any() else f"   {lo:>4}-{hi:<4} days: 0")
    for cut in (0, 30, 90):
        keep = ~(pos & (lag <= cut))
        lo_, hi_ = ci(y[keep], s[keep])
        print(f"   AUC with positives announced within {cut:>2} days removed: {roc_auc_score(y[keep], s[keep]):.3f} [{lo_:.3f}-{hi_:.3f}]  ({int((pos & (lag <= cut)).sum())} removed)")

    # 2. firm memory across the split
    tr = all_filed < np.datetime64("2019-01-01")
    tr_cik_pos = defaultdict(int)
    tr_events = defaultdict(set)
    tr_fd = {}
    for a, yy, d in zip(all_adsh[tr].tolist(), all_y[tr], all_filed[tr]):
        c = subs.get(a, ("?",))[0] or "?"
        if yy == 1:
            tr_cik_pos[c] += 1
            dd = datetime.strptime(str(d), "%Y-%m-%d").date()
            for x in labels.get(c, []):
                if 0 <= (x - dd).days <= WINDOW: tr_events[c].add(x)
    in_train_pos = np.array([tr_cik_pos.get(c, 0) > 0 for c in cik])
    shared = np.zeros(len(adsh), bool)
    for i in range(len(adsh)):
        if y[i] == 1:
            ev = {x for x in labels.get(cik[i], []) if 0 <= (x - fd[i]).days <= WINDOW}
            shared[i] = bool(ev & tr_events.get(cik[i], set()))
    print("\n2. firm memory across the split:")
    print(f"   test filings of firms with a positive training filing: {int(in_train_pos.sum()):,} ({int(y[in_train_pos].sum())} positives, rate {y[in_train_pos].mean():.1%})")
    print(f"   test filings of firms with no positive training filing: {int((~in_train_pos).sum()):,} ({int(y[~in_train_pos].sum())} positives, rate {y[~in_train_pos].mean():.1%})")
    for name, m in (("firms with a positive training filing", in_train_pos), ("firms without (firm-disjoint evaluation)", ~in_train_pos)):
        lo_, hi_ = ci(y[m], s[m]); print(f"   AUC on {name:<42} {roc_auc_score(y[m], s[m]):.3f} [{lo_:.3f}-{hi_:.3f}]")
    print(f"   test positives whose announcement also labels a training filing (shared event): {int(shared.sum())} of {int(y.sum())}")
    keep = ~shared
    lo_, hi_ = ci(y[keep], s[keep]); print(f"   AUC with shared-event positives removed: {roc_auc_score(y[keep], s[keep]):.3f} [{lo_:.3f}-{hi_:.3f}]")
    keep2 = ~in_train_pos | (y == 0)
    lo_, hi_ = ci(y[keep2], s[keep2]); print(f"   AUC with every positive of a firm seen positive in training removed: {roc_auc_score(y[keep2], s[keep2]):.3f} [{lo_:.3f}-{hi_:.3f}]")

    # 3. immature test labels
    print("\n3. label maturity in the test years (announcements in the panel run to its last fetch):")
    last = max(x for v in labels.values() for x in v)
    print(f"   last Item 4.02 in the panel: {last}")
    for yr in sorted(set(years)):
        m = years == yr
        room = (last - datetime(yr, 7, 1).date()).days
        print(f"   {yr}: {int(m.sum()):>5,} filings, {int(y[m].sum()):>3} positives ({y[m].mean():.1%}); a mid-year filing had {min(room, WINDOW):>4} of {WINDOW} days for an announcement; AUC {roc_auc_score(y[m], s[m]):.3f}")
    lo_, hi_ = ci(y[years <= 2020], s[years <= 2020])
    print(f"   AUC on 2019-2020 only, where every label had the full window: {roc_auc_score(y[years <= 2020], s[years <= 2020]):.3f} [{lo_:.3f}-{hi_:.3f}]")
    print("AUDIT2 DONE")


if __name__ == "__main__":
    main()
