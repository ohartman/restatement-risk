"""Where does validation peak? Extends simple_wins to 1:20 and 1:50 per tree, same protocol."""
import os, numpy as np
from sklearn.metrics import roc_auc_score
from simple_wins import forest, fit_score, VAL_YEARS
from events_test import joined
from final import TEST_START, ci, paired
z = np.load("data/out/features.npz", allow_pickle=True); adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
years = filed.astype("datetime64[Y]").astype(int) + 1970
R, _ = joined(adsh, "data/out/features_raw.npz"); R2, n2 = joined(adsh, "data/out/features_raw2.npz")
fin = np.hstack([z["X"], R, R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]])
E, _ = joined(adsh, "data/out/features_events.npz"); I, _ = joined(adsh, "data/out/features_insider.npz")
M, mn = joined(adsh, "data/out/features_market.npz"); M = M[:, [i for i, n in enumerate(mn) if n != "price_cov"]]
L, _ = joined(adsh, "data/out/features_letters.npz"); S, _ = joined(adsh, os.environ["SCRUTINY_NPZ"])
X = np.hstack([fin, E, I, M, L, S]); trn, tst = filed < TEST_START, filed >= TEST_START; yt = y[tst]
base = fit_score(lambda s: forest(s, 0.1), X, y, trn, tst, 5)
print(f"{'variant':<20}{'val 2017':>10}{'val 2018':>10}{'test AUC':>10}{'95% CI':>17}{'  vs 1:10 (paired)':>22}")
for name, strat in (("1:10", 0.1), ("1:20", 0.05), ("1:50", 0.02), ("1:100", 0.01)):
    vals = [roc_auc_score(y[years == v], fit_score(lambda s: forest(s, strat), X, y, years < v, years == v, 3)) for v in VAL_YEARS]
    st = base if strat == 0.1 else fit_score(lambda s: forest(s, strat), X, y, trn, tst, 5)
    lo, hi = ci(yt, st); line = f"{name:<20}{vals[0]:>10.3f}{vals[1]:>10.3f}{roc_auc_score(yt, st):>10.3f}   [{lo:.3f}-{hi:.3f}]"
    if strat != 0.1:
        m, (dlo, dhi), p = paired(yt, base, st); line += f"   {m:+.3f} [{dlo:+.3f},{dhi:+.3f}] P {p:.3f}"
    print(line, flush=True)
print("RATIO MORE DONE")
