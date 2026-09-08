"""Features per split, continued downward from log2, at the 1:10 ratio on the 330 stack."""
import os, numpy as np
from sklearn.metrics import roc_auc_score
from algo_wins import forest, fit_forest, VAL_YEARS
from events_test import joined
from final import TEST_START, ci, paired
z = np.load("data/out/features.npz", allow_pickle=True); adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
years = filed.astype("datetime64[Y]").astype(int) + 1970
R, _ = joined(adsh, "data/out/features_raw.npz"); R2, n2 = joined(adsh, "data/out/features_raw2.npz")
fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
E, _ = joined(adsh, "data/out/features_events.npz"); I, _ = joined(adsh, "data/out/features_insider.npz")
M, mn = joined(adsh, "data/out/features_market.npz"); M = M[:, [i for i, n in enumerate(mn) if n != "price_cov"]]
L, _ = joined(adsh, "data/out/features_letters.npz"); S, _ = joined(adsh, os.environ["SCRUTINY_NPZ"]); F, _ = joined(adsh, os.environ["FLAGS2_NPZ"])
X = np.hstack([fin, E, I, M, L, S, F]); trn, tst = filed < TEST_START, filed >= TEST_START; yt = y[tst]
base = fit_forest(forest, X, y, trn, tst, 5)
print(f"{'features per split (of 660 incl. missing flags)':<50}{'val 2017':>9}{'val 2018':>9}{'test':>9}{'95% CI':>16}{'p@100':>6}{'   vs sqrt (paired)':>22}{'P(<=0)':>8}", flush=True)
for name, mf, n in (("sqrt = 26 (current)", "sqrt", 600), ("log2 = 9", "log2", 600), ("15", 15, 600), ("12", 12, 600), ("7", 7, 600), ("5", 5, 600), ("3", 3, 600), ("log2 = 9, 1,500 trees", "log2", 1500), ("5, 1,500 trees", 5, 1500)):
    make = lambda s, mf=mf, n=n: forest(s, max_features=mf, n=n)
    vals = [roc_auc_score(y[years == v], fit_forest(make, X, y, years < v, years == v, 3)) for v in VAL_YEARS]
    st = base if mf == "sqrt" else fit_forest(make, X, y, trn, tst, 5)
    lo, hi = ci(yt, st); order = np.argsort(-st)
    line = f"{name:<50}{vals[0]:>9.3f}{vals[1]:>9.3f}{roc_auc_score(yt, st):>9.3f}  [{lo:.3f}-{hi:.3f}]{yt[order[:100]].mean():>6.0%}"
    if mf != "sqrt":
        m, (dlo, dhi), p = paired(yt, base, st); line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}]{p:>8.3f}"
    print(line, flush=True)
print("MF SWEEP DONE")
