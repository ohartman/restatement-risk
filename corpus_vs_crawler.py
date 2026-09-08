"""Same filings, two extractors: EDGAR-CORPUS's Item 9A vs our EDGAR-CRAWLER run's."""
import json, re, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scrutiny_features import text_flags, ACC
from relabel import submissions
z = np.load("data/out/features.npz", allow_pickle=True); y = z["y"]; adsh = z["adsh"].tolist(); pos = {a: i for i, a in enumerate(adsh)}
subs = submissions(); by_cy = {}
for a in adsh:
    cik, fd, _ = subs.get(a, (None, None, None))
    if cik: by_cy.setdefault((cik, fd.year), a)
corp = {}
for f in sorted(Path("data/raw/edgar_corpus").glob("2019/*.jsonl")) + sorted(Path("data/raw/edgar_corpus").glob("2016/*.jsonl")):
    for line in f.open(encoding="utf-8"):
        try: r = json.loads(line)
        except Exception: continue
        cik = str(int(r["cik"])) if str(r.get("cik", "")).isdigit() else None
        a = by_cy.get((cik, int(r.get("year") or 0)))
        if a: corp[a] = r
print(f"corpus rows matched (2016, 2019 files): {len(corp):,}")
craw = {}
for line in open("data/raw/ec_items.jsonl", encoding="utf-8"):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("adsh") in corp and r.get("status") == "ok":
        craw[r["adsh"]] = r
both = sorted(set(corp) & set(craw)); print(f"in both: {len(both):,}")
l9c = np.array([len(corp[a].get("section_9A") or "") for a in both]); l9w = np.array([len(craw[a].get("section_9A") or "") for a in both])
l8c = np.array([len(corp[a].get("section_8") or "") for a in both]); l8w = np.array([len(craw[a].get("section_8") or "") for a in both])
print(f"Item 9A chars: corpus median {np.median(l9c):,.0f} (short<200: {(l9c<200).mean():.1%})   crawler median {np.median(l9w):,.0f} (short: {(l9w<200).mean():.1%})")
print(f"Item 8 chars:  corpus median {np.median(l8c):,.0f} (short: {(l8c<200).mean():.1%})   crawler median {np.median(l8w):,.0f} (short: {(l8w<200).mean():.1%})")
print(f"crawler 9A longer than corpus 9A by >2x: {(l9w > 2*l9c).mean():.1%};  shorter by >2x: {(l9c > 2*l9w).mean():.1%}")
fc = np.array([text_flags(corp[a]) for a in both]); fw = np.array([text_flags(craw[a]) for a in both]); yy = y[[pos[a] for a in both]]
names = ["icfr_not_effective", "material_weakness", "remediation", "going_concern", "big4", "icfr_adverse", "legal_len", "class_action", "sec_investigation", "risk_len"]
for k, nm in enumerate(names[:6]):
    c, w = fc[:, k], fw[:, k]
    print(f"  {nm:<20} corpus AUC {roc_auc_score(yy, c):.3f}  crawler AUC {roc_auc_score(yy, w):.3f}   corpus>0 {int((c>0).sum()):,}  crawler>0 {int((w>0).sum()):,}"
          f"   corpus-only {int(((c>0)&(w==0)).sum()):,} (restated {yy[(c>0)&(w==0)].mean() if ((c>0)&(w==0)).any() else float('nan'):.1%})"
          f"   crawler-only {int(((c==0)&(w>0)).sum()):,} (restated {yy[(c==0)&(w>0)].mean() if ((c==0)&(w>0)).any() else float('nan'):.1%})")
# show three corpus-only not-effective cases
k = 0; ex = [a for a, c, w in zip(both, fc[:, k], fw[:, k]) if c > 0 and w == 0][:3]
for a in ex:
    s9c = corp[a].get("section_9A") or ""; s9w = craw[a].get("section_9A") or ""
    print(f"\n{a}: corpus 9A {len(s9c):,} chars, crawler 9A {len(s9w):,} chars")
    m = re.search(r"not effective", s9c, re.I); print("  corpus ...", s9c[max(0, m.start()-150): m.start()+100].replace("\n", " ") if m else "(no 'not effective')")
    m = re.search(r"not effective", s9w, re.I); print("  crawler ...", s9w[max(0, m.start()-150): m.start()+100].replace("\n", " ") if m else "(no 'not effective')")
    print("  crawler 9A head:", s9w[:300].replace("\n", " "))
