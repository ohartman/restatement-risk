#!/usr/bin/env python3
"""Stream 10-K documents and extract text features, keeping none of the HTML.

A primary 10-K document runs to about 2.5 MB, so the corpus of 36,000 filings is
roughly 90 GB. None of it needs to survive: each document is fetched, reduced to
a few dozen numbers, and thrown away -- the same streaming approach the newspaper
archives needed, for the same reason.

The features are the ones the accounting literature keeps finding signal in:
how long the document is, how hard it is to read, how much hedging and negative
language it carries, and how much of it is litigation and uncertainty. Tone is
counted with word lists in the spirit of Loughran-McDonald; these are a working
subset rather than the full published dictionary, which matters if you compare
numbers with a paper that used the real thing.

  set SEC_CONTACT=Your Name you@example.com
  python text_features.py --sample 1500
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/out/text_features.jsonl")
PAUSE = 0.15          # the SEC allows ten a second; this is well under

# Working subsets in the spirit of Loughran-McDonald. Not the full dictionary.
UNCERTAIN = set("""approximate approximately assume assumed assumes assuming believe
believed believes contingency contingent could depend depended depending depends
doubt doubtful estimate estimated estimates fluctuate fluctuation indefinite
likelihood may maybe might nearly occasionally possible possibly precaution
predict predicted prediction preliminary presume probable probably random risk
risks risky roughly seems some somewhat sometimes suggest suggests tentative
uncertain uncertainty unclear unknown unpredictable unsure vague variability
volatile volatility""".split())

NEGATIVE = set("""adverse adversely against bad breach burden burdened cancel
cancelled challenge concern concerns concerned decline declined declines
deficiency deficit deteriorate difficult difficulty diminish disclose
discontinued dispute disruption downturn fail failed failure failures forfeit
fraud harm harmful impair impaired impairment inability inadequate ineffective
insufficient investigation lack late litigation loss losses material misstate
misstatement negative penalties penalty poor problem problems restate restated
restatement restructuring shortfall shut suspend termination unable unfavorable
violation weak weakness weaknesses worse writeoff writedown""".split())

LITIGIOUS = set("""allegation allege alleged appeal arbitration attorney claimant
claims counsel court defendant indemnify injunction judicial juries jury lawsuit
lawsuits legal liable litigation plaintiff prosecute settlement subpoena sue sued
testimony tribunal verdict""".split())

MODAL_WEAK = set("could may might possibly perhaps".split())
MODAL_STRONG = set("will must always never definitely clearly".split())

TEXT_VARS = ["n_words", "log_words", "avg_sentence", "pct_complex", "fog",
             "uncertain_rate", "negative_rate", "litigious_rate",
             "modal_weak_rate", "modal_strong_rate", "number_density",
             "risk_share", "unique_ratio"]

TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def contact():
    c = os.environ.get("SEC_CONTACT", "").strip()
    if not c or "@" not in c:
        sys.exit("Set SEC_CONTACT to a name and email; the SEC returns 403 without one.")
    return c


def get(url, ua, timeout=90):
    """Fetch and decompress. urllib will happily request gzip and hand back the
    compressed bytes, which is how 800 fetches failed identically and silently."""
    req = urllib.request.Request(url, headers={"User-Agent": ua,
                                               "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        enc = (r.headers.get("Content-Encoding") or "").lower()
    if enc == "gzip" or raw[:2] == bytes([0x1F, 0x8B]):
        import gzip
        raw = gzip.decompress(raw)
    elif enc == "deflate":
        import zlib
        raw = zlib.decompress(raw, -zlib.MAX_WBITS)
    return raw


def primary_document(cik, adsh, ua):
    """URL of the main 10-K document, skipping exhibits and graphics."""
    nod = adsh.replace("-", "")
    idx = f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/index.json"
    d = json.loads(get(idx, ua).decode("utf-8", "replace"))
    items = d["directory"]["item"]
    cands = [(i["name"], int(i.get("size") or 0)) for i in items
             if i["name"].lower().endswith((".htm", ".html"))
             and not re.search(r"(^|[-_])ex[-_]?\d|graphic|logo|image", i["name"], re.I)]
    if not cands:
        return None
    name = max(cands, key=lambda x: x[1])[0]
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/{name}"


def syllables(w):
    """Rough syllable count; only the three-plus threshold matters here."""
    w = w.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def features(html):
    text = TAG.sub(" ", html)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&#146;", "'").replace("&#160;", " "))
    text = WS.sub(" ", text)
    words = WORD.findall(text)
    n = len(words)
    if n < 500:
        return None
    lower = [w.lower() for w in words]
    sentences = max(1, text.count(". ") + text.count("? ") + text.count("! "))
    complex_words = sum(1 for w in lower if syllables(w) >= 3)
    pct_complex = complex_words / n
    avg_sentence = n / sentences
    counts = {
        "uncertain_rate": sum(1 for w in lower if w in UNCERTAIN) / n,
        "negative_rate": sum(1 for w in lower if w in NEGATIVE) / n,
        "litigious_rate": sum(1 for w in lower if w in LITIGIOUS) / n,
        "modal_weak_rate": sum(1 for w in lower if w in MODAL_WEAK) / n,
        "modal_strong_rate": sum(1 for w in lower if w in MODAL_STRONG) / n,
    }
    low = text.lower()
    ri = low.find("risk factors")
    risk_share = 0.0
    if ri >= 0:
        # How much of the document sits under risk factors, roughly.
        nxt = low.find("unresolved staff comments", ri)
        risk_share = ((nxt - ri) / len(low)) if nxt > ri else 0.0
    import math
    return {
        "n_words": float(n), "log_words": math.log10(n),
        "avg_sentence": avg_sentence, "pct_complex": pct_complex,
        "fog": 0.4 * (avg_sentence + 100 * pct_complex),
        "number_density": len(re.findall(r"\d", text)) / max(1, len(text)),
        "risk_share": risk_share,
        "unique_ratio": len(set(lower)) / n,
        **counts,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--window-days", type=int, default=1095)
    ap.add_argument("--from-cache", action="store_true",
                    help="fetch every filing in data/out/features.npz, natural rate, no sampling")
    args = ap.parse_args()
    ua = contact()

    from fscore import is_financial, read_quarter
    from train import load_labels
    labels = load_labels()

    pool = []
    for zp in sorted(RAW.glob("*q?.zip")):
        subs, _ = read_quarter(zp)
        for adsh, s in subs.items():
            if s["form"] != "10-K" or is_financial(s["sic"]):
                continue
            try:
                filed = datetime.strptime(s["filed"], "%Y%m%d").date()
            except ValueError:
                continue
            cik = str(int(s["cik"]))
            y = int(any(0 <= (x - filed).days <= args.window_days
                        for x in labels.get(cik, [])))
            pool.append({"adsh": adsh, "cik": cik, "filed": filed.isoformat(), "y": y})
        print(f"  indexed {zp.stem}", flush=True)

    pos = [r for r in pool if r["y"]]
    neg = [r for r in pool if not r["y"]]
    rng = random.Random(7)
    rng.shuffle(pos); rng.shuffle(neg)
    n_pos = min(len(pos), args.sample // 2)
    take = pos[:n_pos] + neg[:int(n_pos * args.neg_per_pos)]
    rng.shuffle(take)
    print(f"\npool {len(pool):,} filings ({len(pos):,} restated)")
    print(f"pilot sample {len(take):,} ({n_pos} restated)\n")

    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["adsh"])
            except Exception:
                pass
        print(f"resuming: {len(done)} already fetched")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    from collections import Counter
    reasons = Counter()
    started = time.time()
    with OUT.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(take, 1):
            if r["adsh"] in done:
                continue
            try:
                url = primary_document(r["cik"], r["adsh"], ua)
                time.sleep(PAUSE)
                if not url:
                    fail += 1
                    continue
                html = get(url, ua).decode("utf-8", "replace")
                f = features(html)
                if f is None:
                    fail += 1
                    continue
                fh.write(json.dumps({**r, "text": f}) + "\n")
                ok += 1
            except Exception as exc:
                fail += 1
                reasons[type(exc).__name__] += 1
                if fail <= 3:
                    print(f"    fetch failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(PAUSE)
            if i % 100 == 0:
                rate = i / (time.time() - started)
                fh.flush()
                print(f"  {i}/{len(take)}  ok {ok} fail {fail}  "
                      f"({rate:.1f}/s, {(len(take)-i)/max(rate,0.01)/60:.0f} min left)",
                      flush=True)
    print(f"\n{ok} filings with text features -> {OUT}  ({fail} failed)")


if __name__ == "__main__":
    main()
