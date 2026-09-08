#!/usr/bin/env python3
"""Regex as the retriever, BERT as the reader: a digest of the sentences that matter.

Fine-tuning ModernBERT on the opening of Item 7 reached 0.61 on validation: the
first 512 tokens of the MD&A are the business overview. The disclosures that
predicted restatements -- controls not effective, material weaknesses, the
auditor's doubts, corrections of prior periods, investigations -- live in Items
9A, 8, 3 and 1A, far past any context window we can afford.

So the regexes that found those disclosures as yes/no flags are used here to
pull the sentences themselves, in a fixed order with a section tag, into a
digest of a few hundred words. The flags say whether such a sentence exists;
a reader can say what it says: remediated or open, one weakness or five, "we
identified" or "we may in the future". Where nothing matches, the digest is the
controls conclusion and the opening of Item 9A, so every filing has one.

  build    from EDGAR-CORPUS (2014-2020)                    -> data/raw/digest.jsonl
  extend   2019+ test rows from full documents on disk       (appended to the same file)

  python digest_text.py build
  python digest_text.py extend
"""

import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from relabel import submissions
from scrutiny_extend import HEAD, sections as cut_sections

CORPUS = Path("data/raw/edgar_corpus")
MDNA = Path("data/raw/mdna")
OUT = Path("data/raw/digest.jsonl")
ACC = re.compile(r"\d{10}-\d{2}-\d{6}")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
MAX_WORDS = 420

# (section, pattern, max sentences) in the order they appear in the digest
PICKS = [
    ("section_9A", r"concluded that|conclusion (that|regarding)|were (not )?effective|was (not )?effective|ineffective", 3),
    ("section_9A", r"material weakness", 6),
    ("section_9A", r"remediat|remedial", 2),
    ("section_9A", r"segregation of duties|accounting personnel|information technology general|period-end|revenue recognition", 2),
    ("section_8", r"restat|correction of|as previously reported|revis(ed|ion) (of )?(previously|prior)|out-of-period|immaterial error", 5),
    ("section_8", r"substantial doubt|going concern", 2),
    ("section_8", r"adverse opinion|except for|emphasis of (a )?matter|critical audit matter", 3),
    ("section_3", r"investigation|subpoena|Wells notice|class action|derivative|whistleblower|Department of Justice|Securities and Exchange Commission", 4),
    ("section_1A", r"material weakness|internal control over financial reporting", 2),
    ("section_7", r"restat|material weakness|going concern|investigation", 3),
]


def digest(sec):
    parts, seen = [], set()
    for key, pat, k in PICKS:
        text = sec.get(key) or ""
        if not text:
            continue
        rx = re.compile(pat, re.I)
        n = 0
        for s in SENT.split(text):
            s = s.strip()
            if 40 <= len(s) <= 600 and rx.search(s) and s not in seen:
                parts.append(f"[{key.replace('section_', '')}] {s}"); seen.add(s); n += 1
                if n >= k:
                    break
    if len(parts) < 3:
        # nothing much matched: the opening of Item 9A, or of Item 7, so the reader still gets the filing's voice
        lead = (sec.get("section_9A") or sec.get("section_7") or "")
        parts.append("[9A] " + " ".join(lead.split()[:150]))
    words = " ".join(parts).split()
    return " ".join(words[:MAX_WORDS])


def build():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh = z["adsh"].tolist()
    subs = submissions()
    want = set(adsh)
    by_cik_year = defaultdict(list)
    for a in adsh:
        cik, f, _ = subs.get(a, (None, None, None))
        if cik:
            by_cik_year[(cik, f.year)].append(a)
    done = set()
    n = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for f in sorted(CORPUS.glob("*/*.jsonl")):
            with f.open(encoding="utf-8") as src:
                for line in src:
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
                        targets = [a for a in by_cik_year.get((cik, yr), []) if a not in done] if cik and yr else []
                    if not targets or not (r.get("section_9A") or r.get("section_8") or r.get("section_7")):
                        continue
                    d = digest(r)
                    for a in targets:
                        if a not in done:
                            fh.write(json.dumps({"adsh": a, "source": "corpus", "text": d}) + "\n"); done.add(a); n += 1
            print(f"  {f.parent.name}/{f.name}: {n:,} digests", flush=True)
    print(f"{n:,} digests -> {OUT}")


def extend():
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh = z["adsh"].tolist(); filed = z["filed"].astype("datetime64[D]")
    pos = {a: i for i, a in enumerate(adsh)}
    done = {json.loads(l)["adsh"] for l in OUT.open(encoding="utf-8")}
    HEAD.setdefault("section_1", re.compile(r"item\s*1(?![0-9a-b])[^a-z0-9]{0,80}business", re.I))
    n = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for p in MDNA.glob("*.full.html.gz"):
            a = p.name.replace(".full.html.gz", "")
            i = pos.get(a)
            if i is None or a in done or filed[i] < np.datetime64("2019-01-01"):
                continue
            with gzip.open(p, "rt", encoding="utf-8") as g:
                sec = cut_sections(g.read())
            if not any(len(sec.get(k, "")) > 200 for k in ("section_9A", "section_8", "section_7")):
                continue
            fh.write(json.dumps({"adsh": a, "source": "fetch", "text": digest(sec)}) + "\n"); done.add(a); n += 1
            if n % 1000 == 0:
                print(f"  {n:,} test digests from saved documents", flush=True)
    print(f"{n:,} test-row digests appended -> {OUT}")


if __name__ == "__main__":
    {"build": build, "extend": extend}[sys.argv[1]]()
