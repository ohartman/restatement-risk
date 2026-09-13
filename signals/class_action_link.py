#!/usr/bin/env python3
"""Does the restatement score predict securities lawsuits? The link a D&O underwriter needs.

Federal dockets with nature of suit 850 (CourtListener, free) name the defendant after "v.".
The defendant is matched to the panel's companies by normalised name (suffixes such as Inc,
Corp, Holdings, Ltd stripped; an exact match on the stem, then a match on the first two
words when unique). SEC enforcement actions ("Securities and Exchange Commission v. ...")
are tagged separately. A 10-K is "sued" if a matched docket was filed within three years
after the 10-K's filing date.

Scores are the headline model's out-of-sample scores for filings of 2019-2023; the model
never saw a lawsuit label, so this is a transfer test: AUC for the lawsuit label, lawsuit
rate by restatement-score decile, and the rate among filings later restated versus not.

  python signals/class_action_link.py
"""
import csv
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from final import ci  # noqa: E402
from relabel import submissions  # noqa: E402

SUFFIX = re.compile(r"\b(incorporated|inc|corporation|corp|company|co|limited|ltd|llc|lp|plc|holdings?|holding|group|international|intl|trust|"
                    r"technologies|technology|tech|industries|enterprises|systems|solutions|the|and|of|de|sa|nv|ag|se)\b")
PUNCT = re.compile(r"[^a-z0-9 ]")


def norm(name):
    s = PUNCT.sub(" ", (name or "").lower().replace("&", " and "))
    s = SUFFIX.sub(" ", s)
    return " ".join(s.split())


def defendant(case):
    if not case or " v. " not in case:
        return None
    d = case.split(" v. ", 1)[1]
    d = re.split(r",| et al| d/b/a| f/k/a| a/k/a", d, 1)[0]
    return d.strip()


def main():
    z = np.load(ROOT / "data/out/scores_headline_final.npz", allow_pickle=True)
    adsh, y, score, filed = z["adsh"], z["y"], z["s4"], pd.to_datetime(z["filed"])
    subs = submissions()
    names = {}
    for zp in sorted((ROOT / "data/raw").glob("*q?.zip")):
        with zipfile.ZipFile(zp).open("sub.txt") as fh:
            for r in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t"):
                if r["form"] == "10-K":
                    names[str(int(r["cik"]))] = r["name"]
    cik = np.array([subs.get(a, ("?",))[0] or "?" for a in adsh.tolist()])
    stem_to_cik = defaultdict(set); two_to_cik = defaultdict(set)
    for c, n in names.items():
        s = norm(n)
        if s:
            stem_to_cik[s].add(c); two_to_cik[" ".join(s.split()[:2])].add(c)
    dockets = [json.loads(l) for l in (ROOT / "data/raw/courtlistener/nos850.jsonl").open(encoding="utf-8")]
    print(f"{len(dockets):,} federal securities dockets since 2014; {len(names):,} panel companies")
    suits = defaultdict(list); sec = defaultdict(list); n_match = n_sec = 0
    for d in dockets:
        dname = defendant(d["case"]); s = norm(dname)
        if not s or not d.get("filed"):
            continue
        by_sec = (d["case"] or "").lower().startswith("securities and exchange commission")
        cands = stem_to_cik.get(s) or (two_to_cik.get(" ".join(s.split()[:2])) if len(s.split()) >= 2 else set())
        if cands and len(cands) == 1:
            c = next(iter(cands)); dt = datetime.strptime(d["filed"], "%Y-%m-%d").date()
            (sec if by_sec else suits)[c].append(dt); n_sec += by_sec; n_match += not by_sec
    print(f"matched to a panel company: {n_match:,} private suits, {n_sec:,} SEC actions; companies with a suit: {len(suits):,}")

    fd = [f.date() for f in filed]
    sued = np.zeros(len(adsh), dtype=int); sec_action = np.zeros(len(adsh), dtype=int); sued_before = np.zeros(len(adsh), dtype=int)
    for i in range(len(adsh)):
        for dt in suits.get(cik[i], []):
            if 0 <= (dt - fd[i]).days <= 1095: sued[i] = 1
            if dt < fd[i]: sued_before[i] = 1
        for dt in sec.get(cik[i], []):
            if 0 <= (dt - fd[i]).days <= 1095: sec_action[i] = 1
    print(f"\ntest filings {len(adsh):,}: sued within three years {int(sued.sum()):,} ({sued.mean():.1%}); SEC action {int(sec_action.sum()):,} ({sec_action.mean():.1%}); restated {int(y.sum()):,} ({y.mean():.1%})")
    print(f"restated filings sued: {sued[y == 1].mean():.1%}; not restated: {sued[y == 0].mean():.1%}   ({sued[y == 1].mean() / max(sued[y == 0].mean(), 1e-9):.1f}x)")
    lo, hi = ci(sued, score); print(f"\nAUC of the restatement score for the LAWSUIT label: {roc_auc_score(sued, score):.3f} [{lo:.3f}-{hi:.3f}]  (the model never saw a lawsuit)")
    lo, hi = ci(sec_action, score); print(f"AUC for an SEC enforcement action: {roc_auc_score(sec_action, score):.3f} [{lo:.3f}-{hi:.3f}]")
    years = filed.year.values
    dec = np.zeros(len(adsh), dtype=int)
    for yr in np.unique(years):
        m = years == yr; dec[m] = pd.qcut(pd.Series(score[m]).rank(method="first"), 10, labels=False).values + 1
    print("\nby restatement-score decile (within year), share of filings followed by a securities suit within three years:")
    print(f"{'decile':>7}{'n':>7}{'restated':>10}{'sued':>8}{'SEC action':>12}{'sued given restated':>21}{'sued given not':>16}")
    for d in range(1, 11):
        m = dec == d
        r1 = sued[m & (y == 1)].mean() if (m & (y == 1)).sum() > 20 else float("nan")
        print(f"{d:>7}{int(m.sum()):>7,}{y[m].mean():>10.1%}{sued[m].mean():>8.1%}{sec_action[m].mean():>12.1%}{r1:>21.1%}{sued[m & (y == 0)].mean():>16.1%}")
    top = dec == 10; rest = ~top
    print(f"\ntop decile sued {sued[top].mean():.1%} vs the rest {sued[rest].mean():.1%}: {sued[top].mean() / sued[rest].mean():.1f}x; "
          f"among filings never sued before: {sued[top & (sued_before == 0)].mean():.1%} vs {sued[rest & (sued_before == 0)].mean():.1%}")
    # does the score add to the restatement outcome itself? within restated filings, does the score still rank the sued ones?
    m = y == 1
    print(f"within the {int(m.sum())} restated filings, AUC of the score for being sued: {roc_auc_score(sued[m], score[m]):.3f}")
    pd.DataFrame({"adsh": adsh, "cik": cik, "filed": filed, "score": score, "restated": y, "sued_3y": sued, "sec_action_3y": sec_action}).to_csv(ROOT / "data/out/class_action_link.csv", index=False)
    print("CLASS ACTION LINK DONE")


if __name__ == "__main__":
    main()
