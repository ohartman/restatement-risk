#!/usr/bin/env python3
"""TabPFN v2 on the restatement problem, bagged over undersampled contexts.

TabPFN is a transformer pretrained on synthetic tables; it "fits" by reading
the training set as context, so there is no training loop and no tuning. The
v2 weights are the ones under the permissive Prior Labs licence and need no
account; v2.5/2.6/3 are gated behind a login, so they are not used here.

v2 was pretrained on contexts of up to 10,000 rows and this GPU has 6 GB, so
the 36,000 training filings cannot be one context. Instead: every positive
plus a fresh draw of negatives makes one ~8,000-row context, and the scores
from several such contexts are averaged. That is random undersampling
bagging -- the same idea as RUSBoost/EasyEnsemble, with a foundation model
where the tree used to be. Rank-based AUC is unaffected by the shifted prior.

  python edge_tabpfn.py [--bags 5] [--context 8000]
"""

import argparse
import time

import numpy as np
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(9)


def boot(y, s, n=400):
    v = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def contexts(X, y, bags, size, rng):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_neg = max(size - len(pos), len(pos))
    for _ in range(bags):
        idx = np.concatenate([pos, rng.choice(neg, min(n_neg, len(neg)), replace=False)])
        rng.shuffle(idx)
        yield X[idx], y[idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/out/features.npz")
    ap.add_argument("--labels")
    ap.add_argument("--bags", type=int, default=5)
    ap.add_argument("--context", type=int, default=8000)
    ap.add_argument("--val-start", default="2017-01-01")
    ap.add_argument("--test-start", default="2019-01-01")
    args = ap.parse_args()

    import torch
    from tabpfn import TabPFNClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}  torch {torch.__version__}")

    z = np.load(args.features, allow_pickle=True)
    X, y, filed = z["X"], z["y"], z["filed"].astype("datetime64[D]")
    if args.labels:
        zl = np.load(args.labels, allow_pickle=True)
        assert (zl["adsh"] == z["adsh"]).all()
        y = zl["y"]
    vs, ts = np.datetime64(args.val_start), np.datetime64(args.test_start)
    fm, vm, tm = filed < vs, (filed >= vs) & (filed < ts), filed >= ts
    trn = fm | vm
    print(f"fit {fm.sum():,} ({y[fm].sum()})  val {vm.sum():,} ({y[vm].sum()})  "
          f"test {tm.sum():,} ({y[tm].sum()})\n")

    def score(Xc, yc, *targets):
        clf = TabPFNClassifier(model_path="tabpfn-v2-classifier.ckpt", device=device,
                               n_estimators=1, random_state=0, ignore_pretraining_limits=True)
        clf.fit(Xc, yc)
        # Predict in chunks: a 6 GB card cannot hold the attention over the
        # whole test set at once.
        return [np.concatenate([clf.predict_proba(T[i:i + 500])[:, 1]
                                for i in range(0, len(T), 500)]) for T in targets]

    rng = np.random.default_rng(0)
    t0 = time.time()
    # Selection pass: context from the fit years only, scored on val and test.
    sv = np.zeros(vm.sum()); st = np.zeros(tm.sum())
    for b, (Xc, yc) in enumerate(contexts(X[fm], y[fm], args.bags, args.context, rng), 1):
        a, c = score(Xc, yc, X[vm], X[tm])
        sv += a; st += c
        print(f"  bag {b}/{args.bags} (fit-only context {len(yc):,} rows)  "
              f"val AUC so far {roc_auc_score(y[vm], sv):.3f}   test {roc_auc_score(y[tm], st):.3f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # Reporting pass: context from fit+val, like every other model's refit.
    st2 = np.zeros(tm.sum())
    for b, (Xc, yc) in enumerate(contexts(X[trn], y[trn], args.bags, args.context, rng), 1):
        (c,) = score(Xc, yc, X[tm])
        st2 += c
        print(f"  bag {b}/{args.bags} (fit+val context)  test AUC so far "
              f"{roc_auc_score(y[tm], st2):.3f}  [{time.time()-t0:.0f}s]", flush=True)

    lo, hi = boot(y[tm], st2)
    p100 = y[tm][np.argsort(-st2)[:100]].mean()
    print(f"\nTabPFN v2, {args.bags} bags x {args.context} rows")
    print(f"  val AUC {roc_auc_score(y[vm], sv):.3f}   test AUC (fit-only) {roc_auc_score(y[tm], st):.3f}"
          f"   refit test AUC {roc_auc_score(y[tm], st2):.3f}  [{lo:.3f}-{hi:.3f}]   p@100 {p100:.1%}")
    np.save("data/out/scores_tabpfn_test.npy", st2)
    np.save("data/out/scores_tabpfn_val.npy", sv)


if __name__ == "__main__":
    main()
