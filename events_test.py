#!/usr/bin/env python3
"""Do auditor changes and CFO turnover add to the financial statements?

Same test as final.py, one more block of features. The headline model (balanced
random forest on 28 ratios + 51 raw items) is refit with six event features
appended -- counts of prior Item 4.01 and CFO Item 5.02 filings in the last one
and three years, and days since the most recent of each -- and the paired
bootstrap of (with events - without events) on the same test filings is the
verdict. Per-year gains are shown so a single year cannot carry the result.

  python events_test.py
"""

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from edge import impute
from final import TEST_START, ci, paired


def joined(adsh, path):
    zr = np.load(path, allow_pickle=True)
    XR, pos = zr["X"], {a: i for i, a in enumerate(zr["adsh"].tolist())}
    R = np.full((len(adsh), XR.shape[1]), np.nan)
    for i, a in enumerate(adsh.tolist()):
        j = pos.get(a)
        if j is not None:
            R[i] = XR[j]
    return R, list(zr["names"])


MARKET_LEVEL = ("turnover", "log_mcap", "btm", "penny", "price_cov")


def market_block(adsh):
    """The audited market block: within-window ratios only (levels are split-adjusted after the
    fact by Yahoo), to be imputed without missing indicators (missingness = delisted later)."""
    M, mn = joined(adsh, "data/out/features_market.npz")
    return M[:, [i for i, n in enumerate(mn) if n not in MARKET_LEVEL]]


def scrutiny_block(adsh, path):
    """The flag block without industry_wave_rate, whose denominator counted the whole panel."""
    S, sn = joined(adsh, path)
    return S[:, [i for i, n in enumerate(sn) if n != "industry_wave_rate"]]


def strategy():
    """Positives:negatives each tree sees. "all" is 1:1; 0.1 is 1:10 (Perols et al. 2017 found
    ~1:4 better than 1:1 for rare-event fraud detection). Set BRF_STRATEGY to override."""
    import os
    s = os.environ.get("BRF_STRATEGY", "all")
    return float(s) if s.replace(".", "", 1).isdigit() else s


def max_features():
    """Features tried per split. sklearn's default is sqrt (26 of 660 here); swept on the
    validation years, 12-15 is best and 15 is the least extreme setting inside the tie band.
    Set BRF_MAX_FEATURES to override (an integer, a fraction, "sqrt" or "log2")."""
    import os
    s = os.environ.get("BRF_MAX_FEATURES", "sqrt")
    return int(s) if s.isdigit() else (float(s) if s.replace(".", "", 1).isdigit() else s)


def brf(seed):
    return BalancedRandomForestClassifier(
        n_estimators=600, min_samples_leaf=5, sampling_strategy=strategy(), max_features=max_features(),
        replacement=True, bootstrap=False, random_state=seed, n_jobs=-1)


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    X28, y, filed, adsh = z["X"], z["y"], z["filed"].astype("datetime64[D]"), z["adsh"]
    R, _ = joined(adsh, "data/out/features_raw.npz")
    E, enames = joined(adsh, "data/out/features_events.npz")
    X79 = np.hstack([X28, R])
    X85 = np.hstack([X79, E])
    trn, tst = filed < TEST_START, filed >= TEST_START
    yt = y[tst]
    years = filed[tst].astype("datetime64[Y]").astype(int) + 1970
    print(f"train {trn.sum():,} ({y[trn].sum()})   test {tst.sum():,} ({yt.sum()})")
    print("event features:", ", ".join(enames))

    # How the events relate to the label on their own, before any model.
    for j, n in enumerate(enames):
        if n.endswith("_n3y"):
            e = E[:, j]
            has = e > 0
            print(f"  {n:<12} restatement rate with event {y[has].mean():.1%} "
                  f"(n={has.sum():,})  without {y[~has].mean():.1%}")
    print()

    hand = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=1.0, min_samples_leaf=40, random_state=0)
    scores = {}
    for tag, X in (("79", X79), ("85 (+events)", X85)):
        scores[f"HGB {tag}"] = HistGradientBoostingClassifier(**hand).fit(
            X[trn], y[trn]).predict_proba(X[tst])[:, 1]
        Xtr, Xte = impute(X[trn], X[tst])
        scores[f"balanced RF {tag}, 5 seeds"] = np.mean(
            [brf(s).fit(Xtr, y[trn]).predict_proba(Xte)[:, 1] for s in range(5)], axis=0)

    print(f"{'model':<34}{'AUC':>7}{'95% CI':>17}{'p@100':>8}{'  +events vs without (paired)':>32}{'P(<=0)':>8}")
    for base in ("HGB", "balanced RF"):
        a = scores[f"{base} 79" if base == "HGB" else f"{base} 79, 5 seeds"]
        b = scores[f"{base} 85 (+events)" if base == "HGB" else f"{base} 85 (+events), 5 seeds"]
        for name, s in ((f"{base}, 79 features", a), (f"{base}, 85 (+events)", b)):
            lo, hi = ci(yt, s)
            p100 = yt[np.argsort(-s)[:100]].mean()
            line = f"{name:<34}{roc_auc_score(yt, s):>7.3f}   [{lo:.3f}-{hi:.3f}]{p100:>8.1%}"
            if s is b:
                m, (dlo, dhi), p = paired(yt, a, b)
                line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>14.3f}"
            print(line, flush=True)

    a = scores["balanced RF 79, 5 seeds"]; b = scores["balanced RF 85 (+events), 5 seeds"]
    print(f"\n{'per year':<10}{'n':>7}{'pos':>6}{'without':>9}{'+events':>9}{'gain':>7}")
    for yr in sorted(set(years)):
        m = years == yr
        print(f"{yr:<10}{m.sum():>7,}{int(yt[m].sum()):>6}{roc_auc_score(yt[m], a[m]):>9.3f}"
              f"{roc_auc_score(yt[m], b[m]):>9.3f}{roc_auc_score(yt[m], b[m]) - roc_auc_score(yt[m], a[m]):>+7.3f}")
    np.save("data/out/scores_events_test.npy", b)


if __name__ == "__main__":
    main()
