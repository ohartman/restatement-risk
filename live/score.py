#!/usr/bin/env python3
"""Score 10-Ks the model has never seen, with calibrated probabilities and reasons.

Model: the validated headline forest (trained on filings through 2018, tuned on 2017-2018,
1:10 per tree, 15 features per split, five seeds) on the audited 325-column stack. It is
refitted here from the main data/out blocks - the same rows, the same settings.

Calibration: isotonic regression from the forest's score to the observed restatement rate,
fitted on its own out-of-sample scores for filings of 2019-2021 (every one of which has had
the full three-year window). A rank becomes "x% chance of an Item 4.02 within three years".

Reasons: per filing, the five features that moved its score most, by walking each tree's
path and attributing the change in leaf probability to the split feature (the tree-
interpreter decomposition), averaged over trees and seeds, in plain words.

  python live/score.py            -> live/data/out/live_scores.csv, live/data/out/live_scores.md
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

MAIN = Path(__file__).resolve().parent.parent
LIVE = MAIN / "live"
sys.path.insert(0, str(MAIN))
os.environ.setdefault("BRF_STRATEGY", "0.1"); os.environ.setdefault("BRF_MAX_FEATURES", "15")
from edge import impute  # noqa: E402
from events_test import MARKET_LEVEL, brf, joined  # noqa: E402
from relabel import submissions  # noqa: E402

TEST_START = np.datetime64("2019-01-01")
from features import ALL_VARS  # noqa: E402

RATIO = {"rsst_acc": "RSST accruals", "ch_rec": "change in receivables", "ch_inv": "change in inventory", "soft_assets": "soft assets share",
         "ch_cs": "change in cash sales", "ch_roa": "change in return on assets", "issue": "issued securities", "log_assets": "size (log assets)",
         "leverage": "leverage", "cash_ratio": "cash ratio", "asset_growth": "asset growth", "rev_growth": "revenue growth", "net_margin": "net margin",
         "roa": "return on assets", "accruals_ta": "accruals to assets", "dsri": "days-sales-in-receivables index", "gmi": "gross margin index",
         "aqi": "asset quality index", "sgi": "sales growth index", "depi": "depreciation index", "lvgi": "leverage index", "wc_ta": "working capital to assets",
         "rec_rev": "receivables to revenue", "inv_rev": "inventory to revenue", "ppe_ta": "PP&E to assets", "loss": "reported a loss", "neg_equity": "negative equity"}
ITEM = {"assets": "total assets", "receivables": "receivables", "ppe": "property, plant and equipment", "retained": "retained earnings", "equity": "equity",
        "ap": "accounts payable", "accrued": "accrued liabilities", "goodwill": "goodwill", "intangibles": "intangibles", "other_assets_nc": "other non-current assets",
        "prepaid": "prepaid expenses", "other_liab_nc": "other non-current liabilities", "aoci": "accumulated other comprehensive income", "apic": "paid-in capital",
        "treasury": "treasury stock", "allowance": "allowance for doubtful accounts", "deferred_rev": "deferred revenue", "dtl": "deferred tax liabilities",
        "shares_out": "shares outstanding", "net_income": "net income", "revenue": "revenue", "cfo": "operating cash flow", "cfi": "investing cash flow",
        "cff": "financing cash flow", "op_income": "operating income", "pretax": "pre-tax income", "tax": "income tax", "sbc": "stock compensation",
        "capex": "capital expenditure", "d_ar": "change in receivables (cash flow)", "d_inv": "change in inventory (cash flow)", "d_ap": "change in payables (cash flow)",
        "interest": "interest expense", "dda": "depreciation and amortisation", "sga": "SG&A", "opex": "operating expenses", "gross_profit": "gross profit",
        "cogs": "cost of goods sold", "rnd": "R&D", "buyback": "buybacks", "acquisitions": "acquisitions", "wavg_shares": "weighted shares", "eps": "EPS",
        "comprehensive": "comprehensive income", "liabilities": "total liabilities", "cash": "cash", "inventory": "inventory", "assets_current": "current assets",
        "liabilities_current": "current liabilities"}
EXTRA = {"sloan_accrual": "Sloan accrual", "cfo_ta": "operating cash flow to assets", "capex_ta": "capex to assets", "sbc_rev": "stock compensation to revenue",
         "goodwill_ta": "goodwill to assets", "retained_ta": "retained earnings to assets", "eps_x_shares_vs_ni": "EPS times shares vs net income (arithmetic check)",
         "benford_mad": "Benford digit deviation", "benford_chi2": "Benford chi-square", "benford_n": "figures tested for Benford", "round_share": "share of round figures",
         "n_facts": "XBRL facts reported", "n_custom_tags": "custom XBRL tags", "custom_share": "share of custom XBRL tags", "filing_lag": "days from fiscal year-end to filing",
         "late_days": "days filed past the deadline", "afs": "filer class", "wksi": "well-known seasoned issuer", "nciks": "co-registrants", "foreign": "foreign filer",
         "fye_changed": "fiscal year-end changed", "n_amend_3y": "10-K amendments in the prior three years", "n_10k_3y": "10-Ks in the prior three years",
         "len_9a": "length of Item 9A", "aud_days_since": "days since the last auditor change", "aud_n1y": "auditor changes in the prior year", "aud_n3y": "auditor changes in the prior three years",
         "cfo_days_since": "days since the last CFO change", "cfo_n1y": "CFO changes in the prior year", "cfo_n3y": "CFO changes in the prior three years",
         "n_8k_1y": "8-Ks in the prior year", "n_filings_1y": "filings in the prior year", "n_10ka_3y": "10-K/As in the prior three years", "n_10qa_3y": "10-Q/As in the prior three years",
         "nt_10k_2y": "late-filing notices in the prior two years", "nt_10q_2y": "late 10-Q notices in the prior two years", "sec_letters_1y": "SEC comment letters in the prior year",
         "sec_letters_2y": "SEC comment letters in the prior two years", "corresp_2y": "replies to the SEC in the prior two years", "days_since_letter": "days since the last SEC letter",
         "industry_wave_1y": "restatements in the industry in the prior year", "prior_402": "earlier restatements", "days_since_402": "days since the last restatement",
         "revised_last_year": "quietly revised last year's figures", "icfr_not_effective": "says its controls over financial reporting are not effective",
         "material_weakness": "mentions of a material weakness", "remediation": "says it is remediating a control weakness", "going_concern": "going-concern language",
         "big4": "Big 4 auditor", "icfr_adverse": "auditor's adverse opinion on controls", "legal_len": "length of Item 3 (legal proceedings)", "class_action": "class action mentioned",
         "sec_investigation": "SEC investigation mentioned", "risk_len": "length of the risk factors", "mw_types_n": "kinds of material weakness named",
         "dcp_not_effective": "says its disclosure controls are not effective", "icfr_changed": "reports a change in internal control", "no_attestation": "no auditor attestation on controls",
         "auditor_tenure": "auditor tenure", "cam_count": "critical audit matters", "emphasis_of_matter": "emphasis-of-matter paragraph", "except_for": "qualified opinion language",
         "correction_of_error": "correction of an error disclosed", "revision_prior": "revision of prior periods disclosed", "out_of_period": "out-of-period adjustment",
         "immaterial_error": "immaterial error disclosed", "previously_reported_n": "'as previously reported' mentions", "restated_in_notes_n": "'restated' in the notes",
         "midtier_auditor": "mid-tier auditor", "internal_investigation": "internal investigation", "whistleblower": "whistleblower mention", "doj": "DOJ mention",
         "derivative_suit": "derivative suit", "rf_weakness_history": "risk factor: past control weaknesses", "rf_sox_risk": "risk factor: SOX compliance", "delisting_notice": "delisting notice",
         "reverse_split": "reverse split", "shell_history": "shell-company history", "egc": "emerging growth company", "nongaap_n": "non-GAAP measures"}


def words(name):
    if name.startswith("missing:"):
        return "no " + words(name[8:]) + " reported"
    if name.startswith("ratio_"):
        k = int(name[6:]); return RATIO.get(ALL_VARS[k], ALL_VARS[k]) if k < len(ALL_VARS) else "Dechow F-score"
    if name in EXTRA:
        return EXTRA[name]
    for suf, tmpl in (("_cur", "this year's {}"), ("_pri", "last year's {}"), ("_chg", "change in {}"), ("_lag", "last year's {}")):
        if name.endswith(suf) and name[: -len(suf)] in ITEM:
            return tmpl.format(ITEM[name[: -len(suf)]])
    return name.replace("_", " ")


WORDS = {
    "filing_lag": "days from fiscal year-end to filing", "late_days": "days filed past the deadline", "prior_402": "earlier restatements",
    "days_since_402": "days since the last restatement", "icfr_not_effective": "says its controls over financial reporting are not effective",
    "material_weakness": "mentions of a material weakness", "remediation": "says it is remediating a control weakness",
    "going_concern": "going-concern language", "big4": "Big 4 auditor", "icfr_adverse": "auditor's adverse opinion on controls",
    "dcp_not_effective": "says its disclosure controls are not effective", "no_attestation": "no auditor attestation on controls",
    "egc": "emerging growth company", "nt_10k_2y": "late-filing notices in the prior two years", "sec_letters_2y": "SEC comment letters in the prior two years",
    "aud_n3y": "auditor changes in the prior three years", "cfo_n3y": "CFO changes in the prior three years", "afs": "filer class",
    "n_custom_tags": "custom XBRL tags", "custom_share": "share of custom XBRL tags", "revised_last_year": "quietly revised last year's figures",
    "auditor_tenure": "auditor tenure", "midtier_auditor": "mid-tier auditor", "delisting_notice": "delisting notice", "shell_history": "shell-company history",
}


def stack(out, adsh, scrutiny, flags2):
    z = np.load(out / "features.npz", allow_pickle=True)
    R, rn = joined(adsh, out / "features_raw.npz"); R2, n2 = joined(adsh, out / "features_raw2.npz")
    keep2 = [i for i, n in enumerate(n2) if n != "prevrpt"]
    fin = np.hstack([z["X"], R, R2[:, keep2]]); fnames = [f"ratio_{i}" for i in range(z["X"].shape[1])] + list(rn) + [n2[i] for i in keep2]
    E, en = joined(adsh, out / "features_events.npz"); I, inn = joined(adsh, out / "features_insider.npz")
    L, ln = joined(adsh, out / "features_letters.npz")
    S, sn = joined(adsh, scrutiny); ks = [i for i, n in enumerate(sn) if n != "industry_wave_rate"]; S = S[:, ks]; sn = [sn[i] for i in ks]
    F, fn = joined(adsh, flags2)
    mp = out / "features_market.npz"
    if mp.exists():
        M, mn = joined(adsh, mp); km = [i for i, n in enumerate(mn) if n not in MARKET_LEVEL]; M = M[:, km]; mn = [mn[i] for i in km]
    else:
        mn = ["ret_12m", "ret_6m", "ret_3m", "abn_12m", "vol_12m", "idio_vol", "max_dd", "neg_ret_days"]; M = np.full((len(adsh), len(mn)), np.nan)
    X = np.hstack([fin, E, I, L, S, F, M]); names = fnames + list(en) + list(inn) + list(ln) + sn + list(fn) + mn
    return X, names, M.shape[1]


def contributions(forests, Xte, n_feat):
    """Tree-interpreter attribution, vectorised: for every non-root node, the change in P(class 1)
    from its parent is credited to the parent's split feature; a sample's contribution vector is
    the sum over the nodes on its path, averaged over trees and seeds."""
    from scipy import sparse
    total = None; n_trees = 0
    for m in forests:
        for est in m.estimators_:
            t = est.tree_; n_nodes = t.node_count
            val = t.value[:, 0, 1] / np.maximum(t.value[:, 0, :].sum(axis=1), 1e-12)
            parent = np.full(n_nodes, -1)
            parent[t.children_left[t.children_left >= 0]] = np.flatnonzero(t.children_left >= 0)
            parent[t.children_right[t.children_right >= 0]] = np.flatnonzero(t.children_right >= 0)
            nonroot = np.flatnonzero(parent >= 0)
            D = sparse.csr_matrix((val[nonroot] - val[parent[nonroot]], (nonroot, t.feature[parent[nonroot]])), shape=(n_nodes, n_feat))
            part = est.decision_path(Xte) @ D
            total = part if total is None else total + part
            n_trees += 1
    return np.asarray(total.todense()) / n_trees


def main():
    out_main, out_live = MAIN / "data/out", LIVE / "data/out"
    zt = np.load(out_main / "features.npz", allow_pickle=True)
    adsh_t, y_t, filed_t = zt["adsh"], zt["y"], zt["filed"].astype("datetime64[D]")
    X_t, names, n_m = stack(out_main, adsh_t, out_main / "features_scrutiny_v3c.npz", out_main / "features_flags2_ec.npz")
    zl = np.load(out_live / "features.npz", allow_pickle=True)
    adsh_l, filed_l = zl["adsh"], zl["filed"].astype("datetime64[D]")
    X_l, names_l, _ = stack(out_live, adsh_l, out_live / "features_scrutiny_v3c.npz", out_live / "features_flags2_ec.npz")
    assert names == names_l, "live stack columns differ from the training stack"
    trn, tst = filed_t < TEST_START, filed_t >= TEST_START
    Xtr, Xte, Xlive = impute(X_t[trn], X_t[tst], X_l, plain_last=n_m)
    print(f"training rows {int(trn.sum()):,} ({int(y_t[trn].sum())} restated); out-of-sample rows {int(tst.sum()):,}; live rows {len(adsh_l):,} filed {filed_l.min()} to {filed_l.max()}", flush=True)
    forests = [brf(s).fit(Xtr, y_t[trn]) for s in range(5)]
    s_test = np.mean([m.predict_proba(Xte)[:, 1] for m in forests], axis=0)
    s_live = np.mean([m.predict_proba(Xlive)[:, 1] for m in forests], axis=0)
    yt, yrs = y_t[tst], filed_t[tst].astype("datetime64[Y]").astype(int) + 1970
    print(f"check: out-of-sample AUC {roc_auc_score(yt, s_test):.3f} (the README says 0.748)", flush=True)
    # calibration on mature out-of-sample years
    mature = yrs <= 2021
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s_test[mature], yt[mature])
    p_live = iso.predict(s_live); p_test = iso.predict(s_test[mature])
    print(f"calibration on {int(mature.sum()):,} filings of 2019-2021: mean predicted {p_test.mean():.1%} vs observed {yt[mature].mean():.1%}; "
          f"top decile predicted {np.sort(p_test)[-len(p_test)//10:].mean():.1%} vs observed {yt[mature][np.argsort(s_test[mature])[-len(p_test)//10:]].mean():.1%}", flush=True)
    # reasons
    print("attributing scores to features (this takes a few minutes)...", flush=True)
    imp_names = names + [f"missing:{n}" for n in names[: len(names) - n_m]]
    C = contributions(forests, Xlive, Xlive.shape[1])
    subs = submissions()
    import csv, glob, io, zipfile
    sub_name = {}
    for zp in sorted(glob.glob(str(LIVE / "data/raw/*q?.zip"))):
        with zipfile.ZipFile(zp).open("sub.txt") as fh:
            for r in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t"):
                sub_name[r["adsh"]] = r["name"].title()
    tick = json.load(open(MAIN / "data/raw/market/company_tickers.json")); by_cik = {}
    for v in tick.values():
        by_cik.setdefault(str(int(v["cik_str"])), (v["ticker"], v["title"]))
    rows = []
    for i, a in enumerate(adsh_l.tolist()):
        cik = subs.get(a, ("?",))[0] or "?"; tk, nm = by_cik.get(cik, ("", "")); nm = nm or sub_name.get(a, ""); tk = tk or "(none)"
        top = np.argsort(-np.abs(C[i]))[:5]
        reasons = "; ".join(f"{'+' if C[i, j] > 0 else '-'} {words(imp_names[j])}" for j in top)
        rows.append({"adsh": a, "cik": cik, "ticker": tk, "company": nm, "filed": str(filed_l[i]), "score": round(float(s_live[i]), 4),
                     "p_restate_3y": round(float(p_live[i]), 3), "reasons": reasons})
    df = pd.DataFrame(rows).sort_values("p_restate_3y", ascending=False)
    df.to_csv(out_live / "live_scores.csv", index=False)
    top = df.head(50)
    with open(out_live / "live_scores.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Restatement risk, 10-Ks filed {filed_l.min()} to {filed_l.max()}\n\nScored by the validated model (trained on filings through 2018); "
                 f"probabilities calibrated on its out-of-sample 2019-2021 scores. {len(df):,} filings; mean predicted probability {df.p_restate_3y.mean():.1%}. "
                 f"None of these labels exist yet - the check is the SEC's Item 4.02 filings through {str(filed_l.max())[:4]}-{int(str(filed_l.max())[:4]) + 3}.\n\n")
        fh.write("| rank | company | ticker | filed | P(restate within 3 years) | reasons |\n|---|---|---|---|---|---|\n")
        for k, r in enumerate(top.itertuples(), 1):
            fh.write(f"| {k} | {r.company} | {r.ticker} | {r.filed} | {r.p_restate_3y:.0%} | {r.reasons} |\n")
    print(f"\n{len(df):,} filings scored -> {out_live / 'live_scores.csv'}; distribution of calibrated probabilities: "
          f"median {df.p_restate_3y.median():.1%}, top decile >= {df.p_restate_3y.quantile(0.9):.1%}, top 1% >= {df.p_restate_3y.quantile(0.99):.1%}")
    print(top.head(15)[["company", "ticker", "filed", "p_restate_3y", "reasons"]].to_string(index=False, max_colwidth=90))
    print("LIVE SCORE DONE")


if __name__ == "__main__":
    main()
