#!/usr/bin/env python3
"""Auditor fee block: audit, audit-related, tax and other fees for each 10-K, from
Item 14 of the 10-K itself when it carries the table, otherwise from the proxy
statement that follows it (fetch_proxy.py). Published predictors: abnormally low
audit fees precede restatements (Blankley, Hurtt & MacGregor 2012), the unexplained
part of the fee predicts fraud, restatements and comment letters (Hribar, Kravet &
Wilson 2014), non-audit fee share and tax fees (Frankel et al. 2002; Kinney et al. 2004).

  python fees_features.py parse10k     -> data/raw/fees_10k.jsonl   (saved 10-Ks, Item 14)
  python fees_features.py parseproxy   -> data/raw/fees_proxy.jsonl (fetched proxies)
  python fees_features.py build        -> data/out/features_fees.npz
"""
import gzip, json, re, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from fees_parse import clean, parse_fees

MDNA = Path("data/raw/mdna"); PROXY = Path("data/raw/proxy")
NAMES = ["fee_audit_log", "fee_total_log", "fee_nonaudit_share", "fee_tax_share", "fee_audit_to_assets", "fee_audit_change",
         "fee_audit_resid", "fee_source_proxy", "fee_related_any", "fee_other_any"]


def parse_10k(adsh):
    p = MDNA / f"{adsh}.full.html.gz"
    if not p.exists():
        return adsh, None
    try:
        t = clean(gzip.open(p, "rt", encoding="utf-8").read())
    except Exception:
        return adsh, None
    for m in re.finditer(r"principal account(?:ant|ing) fees and services", t, re.I):
        r = parse_fees(t[m.start(): m.start() + 8000])
        if r:
            r.pop("pos", None); return adsh, r
    return adsh, None


def parse_proxy(pacc):
    p = PROXY / f"{pacc}.html.gz"
    if not p.exists():
        return pacc, None
    try:
        r = parse_fees(clean(gzip.open(p, "rt", encoding="utf-8").read()))
    except Exception:
        return pacc, None
    if r:
        r.pop("pos", None)
    return pacc, r


def run(fn, keys, out, procs=4):
    done = set()
    if Path(out).exists():
        for l in open(out, encoding="utf-8"):
            try: done.add(json.loads(l)["key"])
            except Exception: pass
    todo = [k for k in keys if k not in done]
    print(f"{len(todo):,} to parse ({len(done):,} done)", flush=True)
    n = ok = 0
    with open(out, "a", encoding="utf-8") as fh, Pool(procs) as pool:
        for k, r in pool.imap_unordered(fn, todo, chunksize=20):
            fh.write(json.dumps({"key": k, "fees": r}) + "\n"); n += 1; ok += r is not None
            if n % 2000 == 0:
                fh.flush(); print(f"  {n:,} parsed, {ok:,} with a fee table", flush=True)
    print(f"{ok:,} of {n:,} parsed carry a fee table -> {out}")


def build():
    z = np.load("data/out/features.npz", allow_pickle=True); adsh = z["adsh"].tolist(); filed = z["filed"].astype("datetime64[D]")
    pos = {a: i for i, a in enumerate(adsh)}
    fees = {}
    for l in open("data/raw/fees_10k.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["fees"]: fees[r["key"]] = (r["fees"], 0.0)
    pmap = {}
    if Path("data/raw/proxy/proxy_for_10k.jsonl").exists():
        for l in open("data/raw/proxy/proxy_for_10k.jsonl", encoding="utf-8"):
            r = json.loads(l); pmap[r["adsh"]] = r["proxy"]
    pf = {}
    if Path("data/raw/fees_proxy.jsonl").exists():
        for l in open("data/raw/fees_proxy.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r["fees"]: pf[r["key"]] = r["fees"]
    for a, p in pmap.items():
        if a not in fees and p in pf:
            fees[a] = (pf[p], 1.0)
    # assets for scaling
    zr = np.load("data/out/features_raw2.npz", allow_pickle=True); names = list(zr["names"]); ai = names.index("assets_cur")
    XR = zr["X"]                                   # bind once: NpzFile re-reads the array on every access
    rpos = {x: i for i, x in enumerate(zr["adsh"].tolist())}
    assets = np.array([XR[rpos[a], ai] if a in rpos else np.nan for a in adsh])
    X = np.full((len(adsh), len(NAMES)), np.nan)
    for a, (f, src) in fees.items():
        i = pos[a]; au = f["audit"]; tot = max(f["total"], au)
        X[i, 0] = np.log(au); X[i, 1] = np.log(tot); X[i, 2] = (tot - au) / tot; X[i, 3] = f["tax"] / tot
        X[i, 4] = au / assets[i] if np.isfinite(assets[i]) and assets[i] > 0 else np.nan
        X[i, 5] = np.log(au / f["audit_prior"]) if f.get("audit_prior") and f["audit_prior"] > 0 else np.nan
        X[i, 7] = src; X[i, 8] = float(f["related"] > 0); X[i, 9] = float(f["other"] > 0)
    # abnormal fee: residual of log audit fee on log assets (+ square), fit on training years only
    trn = (filed < np.datetime64("2019-01-01")) & np.isfinite(X[:, 0]) & np.isfinite(assets) & (assets > 0)
    la = np.log(np.clip(assets, 1e3, None))
    A = np.column_stack([np.ones(len(adsh)), la, la ** 2])
    beta, *_ = np.linalg.lstsq(A[trn], X[trn, 0], rcond=None)
    ok = np.isfinite(X[:, 0]) & np.isfinite(assets) & (assets > 0)
    X[ok, 6] = X[ok, 0] - A[ok] @ beta
    np.savez("data/out/features_fees.npz", X=X, adsh=np.array(adsh), names=np.array(NAMES))
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    has = np.isfinite(X[:, 0])
    print(f"fees for {has.sum():,} of {len(adsh):,} filings ({int((X[:, 7] == 1).sum()):,} from proxies); coverage by year: " +
          " ".join(f"{yr}:{has[years == yr].mean():.0%}" for yr in range(2014, 2024)))
    y = z["y"]
    from sklearn.metrics import roc_auc_score
    for k, nm in enumerate(NAMES):
        v = X[:, k]; m = np.isfinite(v)
        if m.sum() > 100 and 0 < y[m].sum() < m.sum():
            print(f"  {nm:<22} coverage {m.mean():>4.0%}  AUC alone {roc_auc_score(y[m], v[m]):.3f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    z = np.load("data/out/features.npz", allow_pickle=True); adsh = z["adsh"].tolist()
    if mode == "parse10k":
        run(parse_10k, adsh, "data/raw/fees_10k.jsonl")
    elif mode == "parseproxy":
        keys = sorted({json.loads(l)["proxy"] for l in open("data/raw/proxy/proxy_for_10k.jsonl", encoding="utf-8")})
        run(parse_proxy, keys, "data/raw/fees_proxy.jsonl")
    elif mode == "build":
        build()
