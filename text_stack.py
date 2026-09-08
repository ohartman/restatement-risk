#!/usr/bin/env python3
"""MD&A text as a predictor, stacked onto the forest honestly.

Text arrives from two places: EDGAR-CORPUS (Loukas et al. 2021, Apache-2.0), which
has Item 7 for every 10-K filed 1993-2020, and our own fetch of Item 7 for a
sample of 2014-2023 filings (fetch_mdna.py). Both are joined to the feature
cache by accession number where the corpus filename carries one, else by CIK
and filing year.

Rung one of the text ladder: TF-IDF over words and bigrams, a regularised
logistic model, and the model's score used as ONE extra column in the forest.
The text score for every training filing is out-of-fold (five folds by
company, so a firm's other years cannot leak), and the test filings are scored
by a model fit on all training text. Selection -- the regularisation strength --
happens on the 2017-18 validation years.

Later rungs (sentence embeddings, a fine-tuned encoder) plug in the same way:
one column, out-of-fold, paired against the model without it.

  python text_stack.py
"""

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from edge import impute
from events_test import joined
from final import TEST_START, ci, paired
from relabel import submissions

CORPUS = Path("data/raw/edgar_corpus")
MDNA = Path("data/raw/mdna")
ACC = re.compile(r"\d{10}-\d{2}-\d{6}")
MAX_CHARS = 60_000      # ~10k words of MD&A is plenty for bag-of-words


def load_text(adsh, subs):
    """adsh -> MD&A text. EDGAR-CORPUS first for every filing it has; our own fetch only for
    filings the corpus lacks (2021 on, and its gaps).

    Order matters, and it once bit us: the fetch sampled every restated filing, so when
    fetched text took precedence the restated class came from one parser and the clean
    class from another, and a bag-of-words model scored 0.90 out of fold by learning the
    parser. With the corpus first, both classes share one source in 2014-2020; in 2021-23
    every filing with text is fetch-sourced, so the source carries no label either way."""
    by_cik_year = defaultdict(list)
    for a in adsh:
        cik, filed, _ = subs.get(a, (None, None, None))
        if cik:
            by_cik_year[(cik, filed.year)].append(a)
    want = set(adsh)
    text, source = {}, {}
    n_corpus = 0
    for f in sorted(CORPUS.glob("*/*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sec = r.get("section_7") or ""
                if len(sec) < 1500:
                    continue
                m = ACC.search(r.get("filename", ""))
                if m and m.group(0) in want:
                    targets = [m.group(0)]
                else:
                    cik = str(int(r["cik"])) if str(r.get("cik", "")).isdigit() else None
                    try:
                        yr = int(r.get("year"))
                    except (TypeError, ValueError):
                        yr = None
                    targets = [a for a in by_cik_year.get((cik, yr), []) if a not in text] if cik and yr else []
                for a in targets:
                    if a not in text:
                        text[a] = sec[:MAX_CHARS]; source[a] = "corpus"; n_corpus += 1
        print(f"  {f.parent.name}/{f.name}: corpus matches so far {n_corpus:,}", flush=True)
    n_fetch = 0
    idx = MDNA / "index.jsonl"
    if idx.exists():
        for l in idx.open(encoding="utf-8"):
            r = json.loads(l)
            if r["status"] == "ok" and r["adsh"] not in text:
                p = MDNA / f"{r['adsh']}.txt.gz"
                if p.exists():
                    with gzip.open(p, "rt", encoding="utf-8") as g:
                        text[r["adsh"]] = g.read()[:MAX_CHARS]; source[r["adsh"]] = "fetch"; n_fetch += 1
    print(f"  text from corpus {n_corpus:,}, from our fetch (corpus lacked it) {n_fetch:,}")
    load_text.source = source
    return text


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"].tolist(), z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    subs = submissions()
    text = load_text(adsh, subs)
    has = np.array([a in text for a in adsh])
    print(f"\nMD&A text for {has.sum():,} of {len(adsh):,} filings ({100*has.mean():.0f}%)")
    print(f"restatement rate with text {y[has].mean():.2%}   without {y[~has].mean():.2%}")
    trn, tst = filed < TEST_START, filed >= TEST_START
    print(f"train with text {int((trn & has).sum()):,} ({int(y[trn & has].sum())})   "
          f"test with text {int((tst & has).sum()):,} ({int(y[tst & has].sum())})\n")

    # ---- text model, out-of-fold on training filings, grouped by company ----
    ciks = np.array([subs.get(a, (None,))[0] or "" for a in adsh])
    # Train the text model only where the text came from the corpus: fetch-sourced rows
    # are label-stratified and a model could learn the parser instead of the prose.
    corpus = np.array([load_text.source.get(a) == "corpus" for a in adsh])
    tr_idx = np.flatnonzero(trn & has & corpus)
    print(f"text model trains on {len(tr_idx):,} corpus-sourced filings ({int(y[tr_idx].sum())} restated)")
    te_idx = np.flatnonzero(tst & has)
    docs_tr = [text[adsh[i]] for i in tr_idx]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, max_features=200_000,
                          sublinear_tf=True, dtype=np.float32)
    Xt_tr = vec.fit_transform(docs_tr)
    Xt_te = vec.transform([text[adsh[i]] for i in te_idx])
    print(f"tf-idf vocabulary {Xt_tr.shape[1]:,}")

    best = None
    for C in (0.03, 0.1, 0.3, 1.0):
        oof = np.zeros(len(tr_idx))
        for f_tr, f_va in GroupKFold(5).split(Xt_tr, y[tr_idx], ciks[tr_idx]):
            m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xt_tr[f_tr], y[tr_idx][f_tr])
            oof[f_va] = m.decision_function(Xt_tr[f_va])
        # select on the 2017-18 rows of the out-of-fold scores
        vm = np.isin(years[tr_idx], (2017, 2018))
        a_val = roc_auc_score(y[tr_idx][vm], oof[vm])
        a_all = roc_auc_score(y[tr_idx], oof)
        print(f"  C={C:<5} out-of-fold AUC all train {a_all:.3f}   2017-18 {a_val:.3f}")
        if best is None or a_val > best[0]:
            best = (a_val, C, oof)
    a_val, C, oof = best
    m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xt_tr, y[tr_idx])
    s_te = m.decision_function(Xt_te)
    print(f"\ntext alone (C={C}): test AUC {roc_auc_score(y[te_idx], s_te):.3f} on {len(te_idx):,} filings")
    top = np.argsort(m.coef_[0])
    names = vec.get_feature_names_out()
    print("  most restatement-like terms: " + ", ".join(names[top[-25:]][::-1]))
    print("  most clean-like terms:       " + ", ".join(names[top[:25]]))

    # ---- stack: one text column onto the full financial block ----
    R, _ = joined(z["adsh"], "data/out/features_raw.npz")
    R2, n2 = joined(z["adsh"], "data/out/features_raw2.npz")
    R2 = R2[:, [i for i, n in enumerate(n2) if n != "prevrpt"]]
    Xfin = np.hstack([z["X"], R, R2])
    tcol = np.full(len(adsh), np.nan)
    tcol[tr_idx] = oof
    tcol[te_idx] = s_te
    Xtxt = np.hstack([Xfin, tcol.reshape(-1, 1)])

    def brf(seed):
        return BalancedRandomForestClassifier(n_estimators=600, min_samples_leaf=5, sampling_strategy="all",
                                              replacement=True, bootstrap=False, random_state=seed, n_jobs=-1)

    def fit_score(X, rows_tr, rows_te, seeds=5):
        Xtr, Xte = impute(X[rows_tr], X[rows_te])
        return np.mean([brf(s).fit(Xtr, y[rows_tr]).predict_proba(Xte)[:, 1] for s in range(seeds)], axis=0)

    for label, rows_tr, rows_te in (("filings with text, train and test", trn & has, tst & has),
                                    ("all filings, text NaN where missing", trn, tst)):
        print(f"\n=== {label} ===")
        yt = y[rows_te]
        a = fit_score(Xfin, rows_tr, rows_te)
        b = fit_score(Xtxt, rows_tr, rows_te)
        for name, s in (("financial block (242)", a), ("+ MD&A text score (243)", b)):
            lo, hi = ci(yt, s)
            print(f"  {name:<28} AUC {roc_auc_score(yt, s):.3f}  [{lo:.3f}-{hi:.3f}]  "
                  f"p@100 {yt[np.argsort(-s)[:100]].mean():.0%}")
        d, (lo, hi), p = paired(yt, a, b)
        print(f"  text on top of financials: {d:+.3f} [{lo:+.3f},{hi:+.3f}]  P(<=0) {p:.3f}")
    np.save("data/out/text_score.npy", tcol)


if __name__ == "__main__":
    main()
