#!/usr/bin/env python3
"""Does text add anything on top of the financial features?

The balanced pilot showed text alone reaches AUC 0.61. That is not the question
that matters. Word complexity and number density could simply be proxies for
company size, which the financial features already carry -- in which case text
adds nothing. This joins the pilot's text features onto the financial matrix by
accession number and compares, on the same filings, the same folds:

    financial only  vs  financial + text

Cross-validated on the 1,599 pilot filings, since that is where text exists.
The pilot is balanced, so absolute AUCs are not comparable to the 0.69 headline;
only the gap between the two rows means anything.

  python text_increment.py
"""

import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from text_features import TEXT_VARS

z = np.load("data/out/features.npz", allow_pickle=True)
fin = {a: i for i, a in enumerate(z["adsh"].tolist())}
X_fin, y_all = z["X"], z["y"]

text = [json.loads(l) for l in open("data/out/text_features.jsonl", encoding="utf-8")]
joined = [(fin[r["adsh"]], r) for r in text if r["adsh"] in fin]
print(f"pilot filings {len(text)} | joined to financial features {len(joined)}")

idx = np.array([i for i, _ in joined])
Xf = X_fin[idx]
Xt = np.array([[r["text"][k] for k in TEXT_VARS] for _, r in joined])
y = y_all[idx]
print(f"restated {y.sum()} / {len(y)}   (balanced pilot; only the gap matters)\n")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)


def run(name, X):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=30, l2_regularization=1.0,
                                       random_state=0)
    p = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
    a = roc_auc_score(y, p)
    b = []
    for _ in range(500):
        i = rng.integers(0, len(y), len(y))
        b.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"  {name:<26} AUC {a:.3f}  [{lo:.3f}-{hi:.3f}]")
    return p


p_fin = run("financial only", Xf)
p_txt = run("text only", Xt)
p_both = run("financial + text", np.hstack([Xf, Xt]))

# Paired bootstrap on the difference, which is the honest test of "adds anything".
d = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    if 0 < y[i].sum() < len(i):
        d.append(roc_auc_score(y[i], p_both[i]) - roc_auc_score(y[i], p_fin[i]))
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"\n  gain from adding text: {np.mean(d):+.3f}  [{lo:+.3f}, {hi:+.3f}]"
      f"   {'-> real' if lo > 0 else '-> not distinguishable from zero'}")
