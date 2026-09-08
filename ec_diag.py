import json, numpy as np, collections
from sklearn.metrics import roc_auc_score
z = np.load("data/out/features.npz", allow_pickle=True); y = z["y"]; adsh = z["adsh"].tolist()
filed = z["filed"].astype("datetime64[D]"); years = filed.astype("datetime64[Y]").astype(int) + 1970
pos = {a: i for i, a in enumerate(adsh)}
n9 = np.full(len(adsh), np.nan); n8 = np.full(len(adsh), np.nan); n7 = np.full(len(adsh), np.nan)
for l in open("data/raw/ec_items.jsonl", encoding="utf-8"):
    try: r = json.loads(l)
    except Exception: continue
    i = pos.get(r["adsh"])
    if i is None or r.get("status") != "ok": continue
    n9[i] = len(r.get("section_9A", "") or ""); n8[i] = len(r.get("section_8", "") or ""); n7[i] = len(r.get("section_7", "") or "")
print("EDGAR-CRAWLER section lengths by filing year (median chars; share under 200 chars):")
for yr in range(2014, 2024):
    m = (years == yr) & np.isfinite(n9)
    print(f"  {yr}: n {m.sum():>5,}  9A med {np.nanmedian(n9[m]):>7,.0f} short {(n9[m] < 200).mean():>5.1%}   8 med {np.nanmedian(n8[m]):>8,.0f} short {(n8[m] < 200).mean():>5.1%}   7 med {np.nanmedian(n7[m]):>8,.0f} short {(n7[m] < 200).mean():>5.1%}")
blocks = {}
for tag, f in [("v2 corpus+ours", "features_scrutiny.npz"), ("v3 ours", "features_scrutiny_v3.npz"), ("v4 crawler", "features_scrutiny_ec.npz")]:
    s = np.load(f"data/out/{f}", allow_pickle=True); p2 = {a: i for i, a in enumerate(s["adsh"].tolist())}
    idx = np.array([p2.get(a, -1) for a in adsh]); X = np.where(idx[:, None] >= 0, s["X"][np.maximum(idx, 0)], np.nan)
    blocks[tag] = (X, list(s["names"]))
print("\nunivariate AUC by period (train <2019 / test 2019+):")
for nm in ["icfr_not_effective", "material_weakness", "remediation", "going_concern", "icfr_adverse"]:
    line = f"  {nm:<20}"
    for tag, (X, names) in blocks.items():
        v = X[:, names.index(nm)]
        for per, m in (("train", years < 2019), ("test", years >= 2019)):
            ok = m & np.isfinite(v)
            line += f"  {tag} {per} {roc_auc_score(y[ok], v[ok]):.3f}"
    print(line)
X3, n3 = blocks["v3 ours"]; X4, n4 = blocks["v4 crawler"]
t = years >= 2019; j = n3.index("icfr_not_effective")
a, b = X3[t, j], X4[t, j]; ok = np.isfinite(a) & np.isfinite(b)
print(f"\ntest rows: ours=1 & crawler=1 {int(((a==1)&(b==1)&ok).sum()):,}; ours=1 only {int(((a==1)&(b==0)&ok).sum()):,} (restated {y[t][ok&(a==1)&(b==0)].mean():.1%}); crawler=1 only {int(((a==0)&(b==1)&ok).sum()):,} (restated {y[t][ok&(a==0)&(b==1)].mean():.1%})")
short = t & (n9 < 200)
print(f"test rows with a short crawler 9A: {int(short.sum()):,}; ours flags not-effective on {int((X3[short, j]==1).sum()):,} of them")
