#!/usr/bin/env python3
"""More of what worked: specific disclosures the 10-K is required to make, read as facts.

The Item 9A "controls not effective" flag was the biggest gain since the raw
items; a fine-tuned language model on the same text was not. The lesson is that
the valuable text in a filing is the disclosure the law forces, and the way to
read it is to ask for it. This block asks for thirty more things, across Items
1, 1A, 3, 5, 7, 8, 9, 9A and 9B.

  build    from EDGAR-CORPUS (every filing 2014-2020) -> data/out/features_flags2.npz
  extend   fill 2019+ test rows from full documents saved by fetch_mdna.py / fetch_9a.py,
           so both classes in 2021-23 come from one parser and training rows stay corpus-sourced

  python flags2_features.py build
  python flags2_features.py extend
"""

import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from relabel import submissions
from scrutiny_extend import HEAD, NEXT, sections as cut_sections
from text_features import TAG, WS

CORPUS = Path("data/raw/edgar_corpus")
MDNA = Path("data/raw/mdna")
OUT = Path("data/out/features_flags2.npz")
ACC = re.compile(r"\d{10}-\d{2}-\d{6}")

# ---- Item 9A: which weakness, and the shape of the disclosure ----
WEAK_TYPES = {
    "mw_revenue": r"material weakness[^.]{0,300}(revenue|sales)|revenue[^.]{0,200}material weakness",
    "mw_itgc": r"information technology general controls|IT general controls|ITGC|user access|change management controls",
    "mw_segregation": r"segregation of duties",
    "mw_close": r"(period-end|period end|financial close|closing) (financial reporting )?process",
    "mw_staff": r"(insufficient|lack of|limited|inadequate)[^.]{0,60}(accounting (personnel|staff|resources|expertise)|qualified personnel|technical accounting)",
    "mw_nonroutine": r"(non-routine|nonroutine|complex|unusual) (transactions|accounting)",
    "mw_tax": r"material weakness[^.]{0,300}income tax|income tax[^.]{0,200}material weakness",
    "mw_override": r"management override",
}
DCP_NOT_EFF = re.compile(r"disclosure controls and procedures[^.]{0,250}(were|was|are|is) not effective|disclosure controls and procedures[^.]{0,250}ineffective", re.I)
ICFR_CHANGED = re.compile(r"(change|changes) in (our|the Company's|the registrant's) internal control over financial reporting[^.]{0,300}(that (has|have) materially affected|materially affected)(?![^.]{0,40}not)", re.I)
NO_ATTEST = re.compile(r"(does not include|not subject to|exempt from)[^.]{0,120}attestation report", re.I)
# ---- Item 8: the auditor's report and the notes ----
TENURE = re.compile(r"served as the (Company|Company's|registrant's)[^.]{0,40}auditor since (\d{4})", re.I)
CAM = re.compile(r"critical audit matter", re.I)
EMPHASIS = re.compile(r"emphasis of (a )?matter", re.I)
EXCEPT_FOR = re.compile(r"\bexcept for\b[^.]{0,200}(effects|adjustments|matter)", re.I)
CORRECTION = re.compile(r"correction of (an |a |prior period |prior-period )?(error|errors|misstatement)|correction of immaterial errors?", re.I)
REVISION = re.compile(r"revision of (previously|prior)[^.]{0,80}(issued|reported|period)|revised (its |our )?(previously|prior)[^.]{0,80}(issued|reported)|as revised", re.I)
OUT_OF_PERIOD = re.compile(r"out-of-period|out of period adjustment", re.I)
IMMATERIAL_ERR = re.compile(r"immaterial (error|errors|misstatement|correction)", re.I)
PREV_REPORTED = re.compile(r"as previously reported", re.I)
RESTATED_NOTE = re.compile(r"\brestat(ed|ement)\b", re.I)
MIDTIER = re.compile(r"BDO|Grant Thornton|RSM US|McGladrey|Moss Adams|Crowe|CohnReznick|Marcum|Baker Tilly|Plante|EisnerAmper|Cherry Bekaert|Withum|Mazars|CBIZ|Mayer Hoffman", re.I)
# ---- Items 3, 1A, 9B, 5, 1: scrutiny and distress ----
INTERNAL_INV = re.compile(r"(internal|independent|special committee|audit committee)[^.]{0,60}investigation|special committee", re.I)
WHISTLE = re.compile(r"whistleblower|whistle-blower", re.I)
DOJ = re.compile(r"Department of Justice|DOJ|United States Attorney|grand jury", re.I)
DERIVATIVE_SUIT = re.compile(r"derivative (action|lawsuit|complaint|suit)", re.I)
RF_WEAKNESS = re.compile(r"(identified|reported|had|have had)[^.]{0,80}material weakness", re.I)
RF_SOX = re.compile(r"(fail|failure|unable) to maintain effective internal control", re.I)
DELIST = re.compile(r"(deficiency|delisting|non-compliance|noncompliance) (notice|letter|notification)|notice of (deficiency|delisting|non-compliance)|minimum bid price", re.I)
REVERSE_SPLIT = re.compile(r"reverse (stock )?split", re.I)
SHELL = re.compile(r"shell company|reverse merger|reverse acquisition|blank check", re.I)
EGC = re.compile(r"emerging growth company", re.I)
NONGAAP = re.compile(r"non-GAAP|adjusted EBITDA", re.I)

NAMES = (list(WEAK_TYPES) + ["mw_types_n", "dcp_not_effective", "icfr_changed", "no_attestation", "len_9a",
         "auditor_tenure", "cam_count", "emphasis_of_matter", "except_for", "correction_of_error", "revision_prior",
         "out_of_period", "immaterial_error", "previously_reported_n", "restated_in_notes_n", "midtier_auditor",
         "internal_investigation", "whistleblower", "doj", "derivative_suit", "rf_weakness_history", "rf_sox_risk",
         "delisting_notice", "reverse_split", "shell_history", "egc", "nongaap_n"])


def flags2(sec, year):
    s9a = sec.get("section_9A") or ""; s8 = sec.get("section_8") or ""; s3 = sec.get("section_3") or ""
    s1a = sec.get("section_1A") or ""; s1 = sec.get("section_1") or ""; s7 = sec.get("section_7") or ""
    s9b = sec.get("section_9B") or ""; s5 = sec.get("section_5") or ""; s9 = sec.get("section_9") or ""
    weak = [float(bool(re.search(p, s9a, re.I))) for p in WEAK_TYPES.values()]
    m = TENURE.search(s8)
    tenure = float(year - int(m.group(2))) if m and 1900 < int(m.group(2)) <= year else math.nan
    distress = s3 + " " + s9b + " " + s5 + " " + s1
    return weak + [
        float(sum(weak)), float(bool(DCP_NOT_EFF.search(s9a))), float(bool(ICFR_CHANGED.search(s9a))),
        float(bool(NO_ATTEST.search(s9a))), math.log10(1 + len(s9a)),
        tenure, float(len(CAM.findall(s8))), float(bool(EMPHASIS.search(s8))), float(bool(EXCEPT_FOR.search(s8))),
        float(bool(CORRECTION.search(s8))), float(bool(REVISION.search(s8) or REVISION.search(s7))),
        float(bool(OUT_OF_PERIOD.search(s8) or OUT_OF_PERIOD.search(s7))), float(bool(IMMATERIAL_ERR.search(s8))),
        float(len(PREV_REPORTED.findall(s8))), float(len(RESTATED_NOTE.findall(s8))), float(bool(MIDTIER.search(s8))),
        float(bool(INTERNAL_INV.search(s3) or INTERNAL_INV.search(s9a))), float(bool(WHISTLE.search(s3 + s9a))),
        float(bool(DOJ.search(s3))), float(bool(DERIVATIVE_SUIT.search(s3))),
        float(bool(RF_WEAKNESS.search(s1a))), float(bool(RF_SOX.search(s1a))),
        float(bool(DELIST.search(distress))), float(bool(REVERSE_SPLIT.search(distress + s7))),
        float(bool(SHELL.search(s1 + s7))), float(bool(EGC.search(s1 + s1a))), float(len(NONGAAP.findall(s7))),
    ]


def build():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, filed = z["adsh"].tolist(), z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    subs = submissions()
    want = set(adsh); pos = {a: i for i, a in enumerate(adsh)}
    by_cik_year = defaultdict(list)
    for a in adsh:
        cik, f, _ = subs.get(a, (None, None, None))
        if cik:
            by_cik_year[(cik, f.year)].append(a)
    X = np.full((len(adsh), len(NAMES)), np.nan)
    n = 0
    for f in sorted(CORPUS.glob("*/*.jsonl")):
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
                    targets = [a for a in by_cik_year.get((cik, yr), []) if not np.isfinite(X[pos[a], 0])] if cik and yr else []
                if targets and (r.get("section_9A") or r.get("section_8")):
                    for a in targets:
                        i = pos[a]
                        if not np.isfinite(X[i, 0]):
                            X[i] = flags2(r, int(years[i])); n += 1
        print(f"  {f.parent.name}/{f.name}: total {n:,}", flush=True)
    np.savez(OUT, X=X, adsh=np.array(adsh), names=np.array(NAMES))
    report(X, z["y"])


def extend():
    z = np.load(OUT, allow_pickle=True)
    X, adsh = z["X"], z["adsh"].tolist()
    feat = np.load("data/out/features.npz", allow_pickle=True)
    filed = feat["filed"].astype("datetime64[D]"); years = filed.astype("datetime64[Y]").astype(int) + 1970
    pos = {a: i for i, a in enumerate(adsh)}
    # extra headings for the items scrutiny_extend did not cut
    HEAD.update({
        "section_1": re.compile(r"item\s*1(?![0-9a-b])[^a-z0-9]{0,80}business", re.I),
        "section_5": re.compile(r"item\s*5(?![0-9])[^a-z0-9]{0,80}market\s+for", re.I),
        "section_9": re.compile(r"item\s*9(?![0-9a-b])[^a-z0-9]{0,80}changes\s+in\s+and\s+disagreements", re.I),
        "section_9B": re.compile(r"item\s*9b(?![0-9])[^a-z0-9]{0,80}other\s+information", re.I),
    })
    filled = tried = 0
    for p in MDNA.glob("*.full.html.gz"):
        a = p.name.replace(".full.html.gz", "")
        i = pos.get(a)
        if i is None or np.isfinite(X[i, 0]) or filed[i] < np.datetime64("2019-01-01"):
            continue
        tried += 1
        with gzip.open(p, "rt", encoding="utf-8") as g:
            sec = cut_sections(g.read())
        if len(sec.get("section_9A", "")) < 200 and len(sec.get("section_8", "")) < 200:
            continue
        X[i] = flags2(sec, int(years[i])); filled += 1
        if tried % 1000 == 0:
            print(f"  {tried:,} documents, {filled:,} filled", flush=True)
    np.savez(OUT, X=X, adsh=np.array(adsh), names=np.array(NAMES))
    print(f"filled {filled:,} test rows from saved documents ({tried:,} read)")
    report(X, feat["y"])


def report(X, y):
    from sklearn.metrics import roc_auc_score
    has = np.isfinite(X[:, 0])
    print(f"\nflags for {has.sum():,} filings")
    for k, n in enumerate(NAMES):
        v = X[:, k]; ok = np.isfinite(v)
        if ok.sum() and 0 < y[ok].sum() < ok.sum() and len(np.unique(v[ok])) > 1:
            line = f"  {n:<24} coverage {100*ok.mean():3.0f}%  AUC alone {roc_auc_score(y[ok], v[ok]):.3f}"
            if set(np.unique(v[ok])) <= {0.0, 1.0}:
                line += f"   rate when 1: {y[ok][v[ok]==1].mean():6.2%} (n={int((v[ok]==1).sum()):>6,})   when 0: {y[ok][v[ok]==0].mean():.2%}"
            print(line)


if __name__ == "__main__":
    {"build": build, "extend": extend}[sys.argv[1]]()
