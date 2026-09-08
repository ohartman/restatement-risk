#!/usr/bin/env python3
"""Auditor fee table parser, for Item 14 of a 10-K or the proxy statement.

Schedule 14A Item 9(e) fixes the four captions: Audit Fees, Audit-Related Fees,
Tax Fees, All Other Fees. Each caption is followed by the current and the prior
year's amount. Returns dollars (units of "in thousands" are scaled).
"""
import html as _html
import re

TAG = re.compile(r"<[^>]+>"); WS = re.compile(r"\s+")
HEAD = re.compile(r"(principal account(?:ant|ing) fees and services|fees (?:paid|billed) to (?:the )?independent|independent (?:registered public )?accounting firm(?:'s)? fees|audit(?:or)? fees and services|fees (?:of|for) (?:the )?independent|audit and non-audit fees|audit fees)", re.I)
LABELS = {"audit": r"audit\s*fees?", "related": r"audit[\s\-]*related\s*fees?", "tax": r"tax\s*fees?", "other": r"all\s*other\s*fees?", "total": r"total(?:\s*fees)?"}
AMT = re.compile(r"\$?\s?\(?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\)?|(?<![\w])([—–\-]|nil|none)(?![\w])", re.I)


def clean(doc):
    t = _html.unescape(TAG.sub(" ", doc)); t = t.replace("\xa0", " ")
    return WS.sub(" ", t)


AMT2 = re.compile(r"\$\s?\(?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\)?|(?<![\w$.,])(\d{1,3}(?:,\d{3})+)(?![\w.,])|(?<![\w])([—–]|-{1,2}|nil|none|n/a)(?![\w])", re.I)
LABEL_ANY = re.compile(r"audit[\s\-]*related\s*fees?|audit\s*fees?|tax\s*fees?|all\s*other\s*fees?|total(?:\s*fees)?", re.I)


def amounts(seg, n=2):
    """Dollar amounts in a stretch of text: '$ 1,234', '1,234', or a dash meaning zero. Bare
    small integers (page numbers, footnote marks, years) are ignored."""
    out = []
    for m in AMT2.finditer(seg):
        if m.group(3):
            out.append(0.0)
        else:
            raw = m.group(1) or m.group(2); v = float(raw.replace(",", ""))
            if m.group(2) is None and 1990 <= v <= 2035 and "," not in raw:
                continue                                   # a year
            out.append(v)
        if len(out) == n:
            break
    return out


def parse_fees(text):
    """text: cleaned text. Returns dict or None."""
    best = None
    for h in HEAD.finditer(text):
        seg = text[h.start(): h.start() + 6000]
        if re.search(r"incorporated (?:herein )?by reference", seg[:600], re.I) and not re.search(r"\$\s?[\d,]{4,}", seg[:600]):
            continue
        vals = {}
        for key, pat in LABELS.items():
            m = re.search(pat + r"\s*(?:\(\d\)|\(\w\)|\*+)?\s*:?", seg, re.I)
            if not m:
                continue
            tail = seg[m.end(): m.end() + 320]
            nxt = LABEL_ANY.search(tail)
            if nxt and nxt.start() > 0:
                tail = tail[: nxt.start()]                  # stop at the next caption: empty cells stay empty
            a = amounts(tail, 2)
            if a:
                vals[key] = a
        if "audit" not in vals or not vals["audit"]:
            continue
        thousands = bool(re.search(r"in thousands|\(000s?\)|\$000|000s omitted|\(in \$000", seg[:2000], re.I))
        scale = 1000.0 if thousands else 1.0
        cur = {k: v[0] * scale for k, v in vals.items()}
        pri = {k: v[1] * scale for k, v in vals.items() if len(v) > 1}
        if not thousands and 10 <= cur["audit"] < 1000 and all(v < 5000 for v in cur.values()):
            scale = 1000.0; cur = {k: v * 1000 for k, v in cur.items()}; pri = {k: v * 1000 for k, v in pri.items()}; thousands = True
        if cur["audit"] < 2000 or cur["audit"] > 5e8:
            continue
        for k in ("related", "tax", "other"):
            if cur.get(k, 0) > 20 * cur["audit"]:
                cur[k] = 0.0
        rec = {"audit": cur["audit"], "related": cur.get("related", 0.0), "tax": cur.get("tax", 0.0), "other": cur.get("other", 0.0),
               "audit_prior": pri.get("audit"), "total_prior": pri.get("total"), "thousands": thousands, "pos": h.start()}
        tot = cur.get("total")
        rec["total"] = tot if tot and tot >= rec["audit"] else rec["audit"] + rec["related"] + rec["tax"] + rec["other"]
        score = len(vals)
        if best is None or score > best[0]:
            best = (score, rec)
        if score >= 4:
            break
    return best[1] if best else None


if __name__ == "__main__":
    import glob, gzip, random, sys
    random.seed(3)
    files = glob.glob("data/raw/mdna/*.full.html.gz"); random.shuffle(files)
    n = ok = 0; shown = 0
    for f in files[:400]:
        t = clean(gzip.open(f, "rt", encoding="utf-8").read())
        if not re.search(r"principal account(?:ant|ing) fees and services", t, re.I):
            continue
        m = re.search(r"principal account(?:ant|ing) fees and services", t, re.I)
        r = parse_fees(t[m.start():]) if m else None
        segs = [x.start() for x in re.finditer(r"principal account(?:ant|ing) fees and services", t, re.I)]
        r = None
        for s0 in segs:
            r = parse_fees(t[s0: s0 + 8000])
            if r: break
        has_table = any(re.search(r"audit[\s\-]*fees", t[s0: s0 + 4000], re.I) and re.search(r"\$\s?[\d,]{3,}", t[s0: s0 + 4000]) for s0 in segs)
        if has_table:
            n += 1; ok += r is not None
            if r and shown < 6:
                shown += 1; print(f[-32:-13], {k: (int(v) if isinstance(v, float) else v) for k, v in r.items() if k != "pos"})
            elif not r and shown < 12:
                shown += 1; s0 = segs[-1]; print("MISS", f[-32:-13], t[s0: s0 + 300])
    print(f"\nparsed {ok}/{n} 10-Ks that carry a fee table")
    for p in sorted(glob.glob("data/raw/proxy/*.html.gz"))[:6]:
        t = clean(gzip.open(p, "rt", encoding="utf-8").read()); r = parse_fees(t)
        print("proxy", p[-24:-8], {k: (int(v) if isinstance(v, float) else v) for k, v in r.items() if k != "pos"} if r else None)
