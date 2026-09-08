#!/usr/bin/env python3
"""Contradictions: where the filing says one thing and its own numbers, or another part
of itself, says the opposite.

Every restatement is, at bottom, a company saying one thing while the figures say
another. These features look for that inside a single 10-K, on filing day:

  words vs numbers
    tone_7                   (positive - negative words) / words in Item 7
    upbeat_while_ni_fell     tone_7 above the year's median while net income fell >10%
    strong_words_while_loss  "strong / record / robust / solid" per 1,000 words, and the year was a loss
    growth_claim_rev_fell    "revenue increased / growth in revenue" while revenue fell >5%
    adjusted_over_gaap_loss  "adjusted / non-GAAP" mentions per 1,000 words, and GAAP net income < 0
    tone_minus_roa_rank      within-year rank of tone minus rank of ROA (words ahead of numbers)
  words vs words
    effective_but_adverse    management says controls effective; the auditor's opinion on them is adverse
    effective_but_weakness   says effective; a material weakness is mentioned in the same section
    effective_but_corrected  says effective; the notes disclose a correction of an error / restated figures
  words vs behaviour
    effective_but_late       says effective; the filing was later than its deadline
    effective_but_nt         says effective; an NT 10-K in the prior two years
    effective_but_letters    says effective; SEC comment letters in the prior year
  numbers vs silence
    no_gc_but_distressed     no going-concern language; equity negative, a loss, cash under 5% of assets
    gc_but_profitable        going-concern doubt while net income and equity are positive

Text for 2014-2020 from EDGAR-CORPUS, for later filings from our saved documents; the
flags, numbers and behaviour from the blocks already built.

  python contradiction_features.py   -> data/out/features_contradiction.npz
"""

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from events_test import joined
from relabel import submissions
from scrutiny_extend import sections as cut_sections

CORPUS = Path("data/raw/edgar_corpus")
MDNA = Path("data/raw/mdna")
ACC = re.compile(r"\d{10}-\d{2}-\d{6}")
POS = re.compile(r"\b(strong|stronger|strongest|record|robust|solid|growth|improved|improvement|improving|exceeded|momentum|"
                 r"successful|success|favorable|favourable|increase|increased|increases|gains?|outperform\w*)\b", re.I)
NEG = re.compile(r"\b(decline|declined|declines|decrease|decreased|decreases|weak|weaker|weakness|loss|losses|adverse|"
                 r"challenging|difficult|difficulties|deteriorat\w*|impairment|shortfall|unfavorable|unfavourable|"
                 r"downturn|uncertain\w*)\b", re.I)
STRONG = re.compile(r"\b(strong|record|robust|solid)\b", re.I)
GROWTH_CLAIM = re.compile(r"(revenue|sales|net revenue)s?\s+(increased|grew|growth)|growth in (net )?(revenue|sales)|increase in (net )?(revenue|sales)", re.I)
ADJUSTED = re.compile(r"\badjusted\b|non-GAAP", re.I)
WORDS = re.compile(r"[A-Za-z]{2,}")
NAMES = ["tone_7", "upbeat_while_ni_fell", "strong_words_while_loss", "growth_claim_rev_fell", "adjusted_over_gaap_loss",
         "tone_minus_roa_rank", "effective_but_adverse", "effective_but_weakness", "effective_but_corrected",
         "effective_but_late", "effective_but_nt", "effective_but_letters", "no_gc_but_distressed", "gc_but_profitable"]


def text_stats(s7):
    n = max(1, len(WORDS.findall(s7)))
    return {"tone": (len(POS.findall(s7)) - len(NEG.findall(s7))) / n,
            "strong": 1000.0 * len(STRONG.findall(s7)) / n,
            "growth_claim": float(bool(GROWTH_CLAIM.search(s7))),
            "adjusted": 1000.0 * len(ADJUSTED.findall(s7)) / n}


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"], z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    alist = adsh.tolist()
    subs = submissions()
    want = set(alist); pos = {a: i for i, a in enumerate(alist)}
    by_cik_year = defaultdict(list)
    for a in alist:
        cik, f, _ = subs.get(a, (None, None, None))
        if cik:
            by_cik_year[(cik, f.year)].append(a)

    # ---- Item 7 text statistics: corpus first, saved documents for the rest ----
    ts = {}
    for f in sorted(CORPUS.glob("*/*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s7 = r.get("section_7") or ""
                if len(s7) < 1500:
                    continue
                m = ACC.search(r.get("filename", ""))
                targets = [m.group(0)] if (m and m.group(0) in want) else []
                if not targets:
                    cik = str(int(r["cik"])) if str(r.get("cik", "")).isdigit() else None
                    try:
                        yr = int(r.get("year"))
                    except (TypeError, ValueError):
                        yr = None
                    targets = [a for a in by_cik_year.get((cik, yr), []) if a not in ts] if cik and yr else []
                st = text_stats(s7) if targets else None
                for a in targets:
                    ts.setdefault(a, st)
        print(f"  {f.parent.name}/{f.name}: {len(ts):,}", flush=True)
    n_fetch = 0
    for p in MDNA.glob("*.full.html.gz"):
        a = p.name.replace(".full.html.gz", "")
        if a in pos and a not in ts:
            with gzip.open(p, "rt", encoding="utf-8") as g:
                sec = cut_sections(g.read())
            if len(sec.get("section_7", "")) >= 1500:
                ts[a] = text_stats(sec["section_7"]); n_fetch += 1
    print(f"Item 7 statistics for {len(ts):,} filings ({n_fetch:,} from saved documents)")

    # ---- numbers, flags and behaviour from the blocks already built ----
    R2, n2 = joined(adsh, "data/out/features_raw2.npz")
    S, sn = joined(adsh, "data/out/features_scrutiny.npz")
    F2, fn = joined(adsh, "data/out/features_flags2.npz")
    L, ln = joined(adsh, "data/out/features_letters.npz")
    X0 = z["X"]
    from features import ALL_VARS
    col = lambda names, k: names.index(k)
    ni_c, ni_p = R2[:, col(n2, "net_income_cur")], R2[:, col(n2, "net_income_pri")]
    rev_c, rev_p = R2[:, col(n2, "revenue_cur")], R2[:, col(n2, "revenue_pri")]
    equity = R2[:, col(n2, "equity_cur")]
    late = R2[:, col(n2, "late_days")]
    cash_ratio = X0[:, ALL_VARS.index("cash_ratio")]
    roa = X0[:, ALL_VARS.index("roa")]
    loss = X0[:, ALL_VARS.index("loss")]
    neg_eq = X0[:, ALL_VARS.index("neg_equity")]
    not_eff, adverse, mw, gc = S[:, col(sn, "icfr_not_effective")], S[:, col(sn, "icfr_adverse")], S[:, col(sn, "material_weakness")], S[:, col(sn, "going_concern")]
    corrected = np.nan_to_num(F2[:, col(fn, "correction_of_error")]) + np.nan_to_num(F2[:, col(fn, "restated_in_notes_n")])
    nt2 = L[:, col(ln, "nt_10k_2y")]; letters1 = L[:, col(ln, "sec_letters_1y")]

    def unlog(v):
        return np.sign(v) * (10 ** np.abs(v) - 1)
    ni_chg = (unlog(ni_c) - unlog(ni_p)) / np.maximum(1.0, np.abs(unlog(ni_p)))
    rev_chg = (unlog(rev_c) - unlog(rev_p)) / np.maximum(1.0, np.abs(unlog(rev_p)))

    tone = np.array([ts[a]["tone"] if a in ts else np.nan for a in alist])
    strong = np.array([ts[a]["strong"] if a in ts else np.nan for a in alist])
    growth = np.array([ts[a]["growth_claim"] if a in ts else np.nan for a in alist])
    adjusted = np.array([ts[a]["adjusted"] if a in ts else np.nan for a in alist])

    # within-year ranks for the words-ahead-of-numbers feature
    tone_rank = np.full(len(alist), np.nan); roa_rank = np.full(len(alist), np.nan)
    for yr in np.unique(years):
        m = (years == yr) & np.isfinite(tone) & np.isfinite(roa)
        if m.sum() > 10:
            idx = np.flatnonzero(m)
            tone_rank[idx] = np.argsort(np.argsort(tone[idx])) / (len(idx) - 1)
            roa_rank[idx] = np.argsort(np.argsort(roa[idx])) / (len(idx) - 1)
    med_tone = {yr: np.nanmedian(tone[years == yr]) for yr in np.unique(years)}
    tone_hi = np.array([tone[i] > med_tone[years[i]] if np.isfinite(tone[i]) else np.nan for i in range(len(alist))])
    effective = np.where(np.isfinite(not_eff), 1 - not_eff, np.nan)

    def both(a, b):
        out = a * b
        out[~(np.isfinite(a) & np.isfinite(b))] = np.nan
        return out
    X = np.column_stack([
        tone,
        both(tone_hi, (ni_chg < -0.10).astype(float)) if True else None,
        both(strong, loss),
        both(growth, (rev_chg < -0.05).astype(float)),
        both(adjusted, loss),
        tone_rank - roa_rank,
        both(effective, adverse), both(effective, (mw > 0).astype(float)), both(effective, (corrected > 0).astype(float)),
        both(effective, (late > 0).astype(float)), both(effective, (nt2 > 0).astype(float)), both(effective, (letters1 > 0).astype(float)),
        both(np.where(np.isfinite(gc), 1 - gc, np.nan), ((neg_eq > 0) & (loss > 0) & (cash_ratio < 0.05)).astype(float)),
        both(gc, ((loss == 0) & (neg_eq == 0)).astype(float)),
    ])
    # rows with no text at all get NaN everywhere so the block's coverage is honest
    X[~np.isfinite(tone)] = np.nan
    Path("data/out").mkdir(exist_ok=True)
    np.savez("data/out/features_contradiction.npz", X=X, adsh=adsh, names=np.array(NAMES))
    from sklearn.metrics import roc_auc_score
    print(f"\ncontradiction features for {int(np.isfinite(X[:, 0]).sum()):,} filings")
    for k, nm in enumerate(NAMES):
        v = X[:, k]; ok = np.isfinite(v)
        if ok.sum() and 0 < y[ok].sum() < ok.sum() and len(np.unique(v[ok])) > 1:
            line = f"  {nm:<24} coverage {100*ok.mean():3.0f}%  AUC alone {roc_auc_score(y[ok], v[ok]):.3f}"
            if set(np.unique(v[ok])) <= {0.0, 1.0}:
                line += f"   rate when 1: {y[ok][v[ok]==1].mean():6.2%} (n={int((v[ok]==1).sum()):>6,})   when 0: {y[ok][v[ok]==0].mean():.2%}"
            print(line)


if __name__ == "__main__":
    main()
