#!/usr/bin/env python3
"""Who is already worried: the company, its auditor, its industry, and its own past.

The restatement label is misstatement times detection. The statements speak to
the first; these features speak to the second, from free, dated, public sources.

From the 10-K text itself (EDGAR-CORPUS, every filing 2014-2020; NaN after):
  icfr_not_effective   Item 9A says internal control over financial reporting was not effective
  material_weakness    Item 9A mentions a material weakness (count of mentions)
  remediation          Item 9A talks about remediating
  going_concern        auditor or management raises substantial doubt (Items 8 / 7)
  big4                 audit report signed by Deloitte, EY, KPMG or PwC
  icfr_adverse         auditor's adverse opinion on internal control
  legal_len            log10 length of Item 3 (legal proceedings)
  class_action         Item 3 mentions a securities class action
  sec_investigation    Items 3 / 7 mention an SEC investigation, subpoena or Wells notice
  risk_len             log10 length of Item 1A

From the label file and the panel (every filing):
  prior_402            Item 4.02 announcements by this firm before this filing date
  days_since_402       days since the last one (capped at 10 years)
  revised_last_year    this 10-K's comparatives change last year's net income by >5%
                       -- last year's silent revision, visible the day this one is filed
  industry_wave_1y     Item 4.02 announcements in the same 2-digit SIC in the prior 365 days
  industry_wave_rate   the same, per 100 filers in that industry

  python scrutiny_features.py   -> data/out/features_scrutiny.npz (X, adsh, names)
"""

import csv
import io
import json
import math
import re
import zipfile
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from littler2 import collect, rel_diff
from relabel import submissions
from train import load_labels

CORPUS = Path("data/raw/edgar_corpus")
ACC = re.compile(r"\d{10}-\d{2}-\d{6}")
NAMES = ["icfr_not_effective", "material_weakness", "remediation", "going_concern", "big4", "icfr_adverse",
         "legal_len", "class_action", "sec_investigation", "risk_len",
         "prior_402", "days_since_402", "revised_last_year", "industry_wave_1y", "industry_wave_rate"]

NOT_EFF = re.compile(r"(internal control over financial reporting|disclosure controls and procedures)[^.]{0,200}?"
                     r"\b(were|was|are|is)\s+not\s+effective|\bnot\s+effective\b|\bineffective\b", re.I)
MW = re.compile(r"material weakness", re.I)
REMED = re.compile(r"remediat", re.I)
GC = re.compile(r"substantial doubt[^.]{0,120}going concern|going concern[^.]{0,120}substantial doubt", re.I)
BIG4 = re.compile(r"Ernst\s*&\s*Young|Deloitte|KPMG|PricewaterhouseCoopers|Pricewaterhouse\s*Coopers", re.I)
ADVERSE = re.compile(r"adverse opinion[^.]{0,200}internal control|internal control[^.]{0,200}adverse opinion", re.I)
CLASS_ACTION = re.compile(r"(securities|shareholder|stockholder)\s+class\s+action|putative class action", re.I)
SEC_INV = re.compile(r"(SEC|Securities and Exchange Commission)[^.]{0,120}(investigation|subpoena|Wells notice|enforcement)|"
                     r"(formal|informal)\s+(order of )?investigation", re.I)


def text_flags(r):
    s9 = r.get("section_9A") or ""; s8 = r.get("section_8") or ""; s7 = r.get("section_7") or ""
    s3 = r.get("section_3") or ""; s1a = r.get("section_1A") or ""
    return [float(bool(NOT_EFF.search(s9))), float(len(MW.findall(s9))), float(bool(REMED.search(s9))),
            float(bool(GC.search(s8) or GC.search(s7))), float(bool(BIG4.search(s8))),
            float(bool(ADVERSE.search(s8))), math.log10(1 + len(s3)), float(bool(CLASS_ACTION.search(s3))),
            float(bool(SEC_INV.search(s3) or SEC_INV.search(s7))), math.log10(1 + len(s1a))]


def main():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh = z["adsh"].tolist()
    subs = submissions()
    want = set(adsh)
    by_cik_year = defaultdict(list)
    for a in adsh:
        cik, filed, _ = subs.get(a, (None, None, None))
        if cik:
            by_cik_year[(cik, filed.year)].append(a)

    # ---- text flags from the corpus ----
    flags = {}
    for f in sorted(CORPUS.glob("*/*.jsonl")):
        n = 0
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = ACC.search(r.get("filename", ""))
                targets = [m.group(0)] if (m and m.group(0) in want) else []
                if not targets:
                    cik = str(int(r["cik"])) if str(r.get("cik", "")).isdigit() else None
                    try:
                        yr = int(r.get("year"))
                    except (TypeError, ValueError):
                        yr = None
                    targets = [a for a in by_cik_year.get((cik, yr), []) if a not in flags] if cik and yr else []
                if targets and (r.get("section_9A") or r.get("section_8")):
                    fl = text_flags(r)
                    for a in targets:
                        flags.setdefault(a, fl); n += 1
        print(f"  {f.parent.name}/{f.name}: +{n:,} (total {len(flags):,})", flush=True)

    # ---- history: prior 4.02s, last year's silent revision, industry wave ----
    labels = load_labels()
    sic = {}
    for zp in sorted(Path("data/raw").glob("*q?.zip")):
        with zipfile.ZipFile(zp).open("sub.txt") as fh:
            for r in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t"):
                sic[r["adsh"]] = (r["sic"] or "00")[:2]
    # announcements by industry: date list per SIC-2, built from the firms' SICs in our panel
    cik_sic = {}
    for a in adsh:
        c = subs.get(a, (None,))[0]
        if c and c not in cik_sic:
            cik_sic[c] = sic.get(a, "00")
    ind_ann = defaultdict(list)
    for c, dates in labels.items():
        for d in dates:
            ind_ann[cik_sic.get(c, "00")].append(d)
    for v in ind_ann.values():
        v.sort()
    filers = defaultdict(set)
    for a in adsh:
        filers[sic.get(a, "00")].add(subs.get(a, (None,))[0])

    own, comp, meta = collect()          # each filing's own figures and comparatives
    own_by_key = {}
    for a, (cik, P) in meta.items():
        if a in own:
            own_by_key[(cik, P)] = own[a]

    X = np.full((len(adsh), len(NAMES)), np.nan)
    for i, a in enumerate(adsh):
        cik, filed, _ = subs.get(a, (None, None, None))
        if cik is None:
            continue
        row = flags.get(a, [np.nan] * 10)
        anns = [d for d in labels.get(cik, []) if d < filed]
        prior = len(anns)
        since = min((filed - max(anns)).days, 3650) if anns else 3650
        # this filing's comparatives vs. the previous filing's own figures for the same period
        rev = np.nan
        m = meta.get(a)
        if m is not None:
            _, P = m
            prev = f"{int(P[:4]) - 1}{P[4:]}"
            d = comp.get((cik, prev)); o = own_by_key.get((cik, prev))
            if d is not None and o is not None:
                rd = rel_diff(o.get("net_income"), d.get("net_income"))
                if rd is not None:
                    rev = float(rd[0] > 0.05 and rd[1] > 1e5)
        g = sic.get(a, "00")
        ds = ind_ann.get(g, [])
        lo = bisect_left(ds, filed - timedelta(days=365)); hi = bisect_left(ds, filed)
        wave = hi - lo
        nf = max(1, len(filers.get(g, ())))
        X[i] = row + [prior, since, rev, wave, 100.0 * wave / nf]
    Path("data/out").mkdir(exist_ok=True)
    np.savez("data/out/features_scrutiny.npz", X=X, adsh=np.array(adsh), names=np.array(NAMES))

    y = z["y"]
    from sklearn.metrics import roc_auc_score
    print(f"\ntext flags for {len(flags):,} filings; history features for all {len(adsh):,}")
    for k, n in enumerate(NAMES):
        v = X[:, k]; ok = np.isfinite(v)
        if ok.sum() and 0 < y[ok].sum() < ok.sum():
            extra = ""
            if set(np.unique(v[ok])) <= {0.0, 1.0}:
                extra = f"   rate when 1: {y[ok][v[ok]==1].mean():.2%} (n={int((v[ok]==1).sum()):,})  when 0: {y[ok][v[ok]==0].mean():.2%}"
            print(f"  {n:<20} coverage {100*ok.mean():3.0f}%  AUC alone {roc_auc_score(y[ok], v[ok]):.3f}{extra}")


if __name__ == "__main__":
    main()
