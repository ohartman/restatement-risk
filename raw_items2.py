#!/usr/bin/env python3
"""Everything else the financial statements say, before we look anywhere else.

Three blocks, all from the quarterly zips already on disk:

1. Forty more raw statement items. The first raw-item block used 17 concepts
   and moved the headline more than anything else had; a scan of one quarter
   shows we were ignoring operating cash flow (in 93% of filings), operating
   income, income tax, share-based compensation, capex, goodwill, retained
   earnings and thirty others. Each is stored as signed log10 for this year,
   last year, and the change, as before. A dozen ratios the literature names
   are built explicitly on top -- the Sloan accrual (net income less operating
   cash flow) most of all.

2. Benford's law. Amiram, Bozanic & Rouen (2015) showed that how far a
   filing's leading digits stray from Benford's distribution predicts
   restatements. It is computed here over every dollar figure in the filing,
   segments and company-specific tags included, and is the one feature in the
   project where the numbers testify about themselves.

3. Filing behaviour from sub.txt: days from fiscal year-end to filing and how
   late that is against the filer's deadline, filer-status class, whether the
   fiscal year-end moved, how many 10-K/As the company filed in the prior three
   years, and how many company-specific XBRL tags the filing needed.

  python raw_items2.py       -> data/out/features_raw2.npz  (X, adsh, names)
"""

import csv
import io
import math
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from fscore import is_financial

RAW = Path("data/raw")
OUT = Path("data/out/features_raw2.npz")

# concept -> candidate tags, first found wins.  (S)tock at period end / (F)low over the year.
STOCK = {
    "assets": ["Assets"], "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "ppe": ["PropertyPlantAndEquipmentNet"],
    "retained": ["RetainedEarningsAccumulatedDeficit"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "ap": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    "accrued": ["AccruedLiabilitiesCurrent"],
    "goodwill": ["Goodwill"],
    "intangibles": ["IntangibleAssetsNetExcludingGoodwill"],
    "other_assets_nc": ["OtherAssetsNoncurrent"],
    "prepaid": ["PrepaidExpenseAndOtherAssetsCurrent", "PrepaidExpenseCurrent"],
    "other_liab_nc": ["OtherLiabilitiesNoncurrent"],
    "aoci": ["AccumulatedOtherComprehensiveIncomeLossNetOfTax"],
    "apic": ["AdditionalPaidInCapital", "AdditionalPaidInCapitalCommonStock"],
    "treasury": ["TreasuryStockValue"],
    "allowance": ["AllowanceForDoubtfulAccountsReceivableCurrent"],
    "deferred_rev": ["ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent"],
    "dtl": ["DeferredTaxLiabilitiesNoncurrent"],
    "shares_out": ["CommonStockSharesOutstanding"],
}
FLOW = {
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "cfi": ["NetCashProvidedByUsedInInvestingActivities"],
    "cff": ["NetCashProvidedByUsedInFinancingActivities"],
    "op_income": ["OperatingIncomeLoss"],
    "pretax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "tax": ["IncomeTaxExpenseBenefit"],
    "sbc": ["ShareBasedCompensation"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "d_ar": ["IncreaseDecreaseInAccountsReceivable"],
    "d_inv": ["IncreaseDecreaseInInventories"],
    "d_ap": ["IncreaseDecreaseInAccountsPayable", "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities"],
    "interest": ["InterestExpense"],
    "dda": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation"],
    "sga": ["SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"],
    "opex": ["OperatingExpenses", "CostsAndExpenses"],
    "gross_profit": ["GrossProfit"],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "buyback": ["PaymentsForRepurchaseOfCommonStock"],
    "acquisitions": ["PaymentsToAcquireBusinessesNetOfCashAcquired"],
    "wavg_shares": ["WeightedAverageNumberOfSharesOutstandingBasic"],
    "eps": ["EarningsPerShareBasic"],
    "comprehensive": ["ComprehensiveIncomeNetOfTax"],
}
TAG2 = {t: (c, "S") for c, ts in STOCK.items() for t in ts}
TAG2.update({t: (c, "F") for c, ts in FLOW.items() for t in ts})
RANK = {t: i for c, ts in list(STOCK.items()) + list(FLOW.items()) for i, t in enumerate(ts)}
CONCEPTS = list(STOCK) + list(FLOW)
BENFORD = np.log10(1 + 1 / np.arange(1, 10))
DEADLINE = {"1-LAF": 60, "2-ACC": 75}          # everyone else has 90 days
AFS_ORD = {"1-LAF": 1, "2-ACC": 2, "3-SRA": 3, "4-NON": 4, "5-SML": 5}


def slog(v):
    return math.nan if v is None else math.copysign(math.log10(1 + abs(v)), v)


def ratio(a, b):
    return math.nan if (a is None or not b) else a / abs(b)


def year_before(ddate):
    return f"{int(ddate[:4]) - 1}{ddate[4:]}"


def read_subs(zp):
    with zp.open("sub.txt") as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"),
                                   delimiter="\t"))


def filing_history():
    """cik -> sorted [(filed, form, fye)] over every quarter, for amendments and FYE changes."""
    hist = defaultdict(list)
    for p in sorted(RAW.glob("*q?.zip")):
        for r in read_subs(zipfile.ZipFile(p)):
            if r["form"] in ("10-K", "10-K/A", "10-KT"):
                try:
                    hist[str(int(r["cik"]))].append(
                        (datetime.strptime(r["filed"], "%Y%m%d").date(), r["form"], r["fye"]))
                except ValueError:
                    pass
    for v in hist.values():
        v.sort()
    return hist


def main():
    hist = filing_history()
    names = ([f"{c}_cur" for c in CONCEPTS] + [f"{c}_pri" for c in CONCEPTS]
             + [f"{c}_chg" for c in CONCEPTS]
             + ["sloan_accrual", "cfo_ta", "capex_ta", "sbc_rev", "goodwill_ta", "retained_ta",
                "tax_rate", "allowance_ar", "dda_ppe", "gross_margin", "cff_ta", "buyback_ta",
                "opinc_cfo_gap", "d_ar_rev", "eps_x_shares_vs_ni"]
             + ["benford_mad", "benford_chi2", "benford_n", "round_share", "n_facts",
                "n_custom_tags", "custom_share"]
             + ["filing_lag", "late_days", "afs", "wksi", "nciks", "foreign",
                "fye_changed", "n_amend_3y", "n_10k_3y"])
    X, adsh_out = [], []
    for p in sorted(RAW.glob("*q?.zip")):
        z = zipfile.ZipFile(p)
        subs = {r["adsh"]: r for r in read_subs(z)
                if r["form"] == "10-K" and not is_financial(r["sic"]) and r["period"]}
        vals = defaultdict(dict)              # adsh -> (concept, which) -> value
        best = defaultdict(dict)
        digits = defaultdict(lambda: np.zeros(9))
        rounds = defaultdict(lambda: [0, 0])
        nfacts, ncustom = defaultdict(int), defaultdict(set)
        with z.open("num.txt") as f:
            for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"),
                                    delimiter="\t"):
                a = r["adsh"]
                s = subs.get(a)
                if s is None:
                    continue
                try:
                    v = float(r["value"])
                except (TypeError, ValueError):
                    continue
                period = s["period"]
                # --- Benford and shape, over every current-year dollar figure ---
                if r["uom"] == "USD" and r["ddate"] == period:
                    nfacts[a] += 1
                    if r["version"].startswith("00"):
                        ncustom[a].add(r["tag"])
                    av = abs(v)
                    if av >= 10:
                        digits[a][int(str(int(av))[0]) - 1] += 1
                    if av >= 10000:
                        rounds[a][1] += 1
                        if int(av) % 1000 == 0:
                            rounds[a][0] += 1
                # --- named concepts, consolidated only ---
                if r["segments"] or r["coreg"]:
                    continue
                hit = TAG2.get(r["tag"])
                if hit is None:
                    continue
                concept, kind = hit
                if kind == "S" and r["qtrs"] != "0":
                    continue
                if kind == "F" and r["qtrs"] != "4":
                    continue
                if r["ddate"] == period:
                    which = "cur"
                elif r["ddate"] == year_before(period):
                    which = "pri"
                else:
                    continue
                key = (concept, which)
                prev = best[a].get(key)
                if prev is None or RANK[r["tag"]] < prev:
                    best[a][key] = RANK[r["tag"]]
                    vals[a][key] = v
        n = 0
        for a, s in subs.items():
            d = vals.get(a, {})
            g = lambda c, w="cur": d.get((c, w))
            if g("assets") is None:
                continue
            row = [slog(g(c)) for c in CONCEPTS] + [slog(g(c, "pri")) for c in CONCEPTS]
            for c in CONCEPTS:
                cu, pr = g(c), g(c, "pri")
                row.append(slog(cu - pr) if (cu is not None and pr is not None) else math.nan)
            ta = g("assets")
            ni, cfo = g("net_income"), g("cfo")
            row += [
                ratio(ni - cfo, ta) if (ni is not None and cfo is not None) else math.nan,
                ratio(cfo, ta), ratio(g("capex"), ta), ratio(g("sbc"), g("revenue")),
                ratio(g("goodwill"), ta), ratio(g("retained"), ta),
                ratio(g("tax"), g("pretax")), ratio(g("allowance"), g("receivables")),
                ratio(g("dda"), g("ppe")), ratio(g("gross_profit"), g("revenue")),
                ratio(g("cff"), ta), ratio(g("buyback"), ta),
                ratio(g("op_income") - cfo, ta) if (g("op_income") is not None and cfo is not None) else math.nan,
                ratio(g("d_ar"), g("revenue")),
                # EPS times shares should roughly equal net income; a gap is a
                # consistency check the filing performs on itself.
                ratio(g("eps") * g("wavg_shares") - ni, abs(ni) if ni else None)
                if (g("eps") is not None and g("wavg_shares") is not None and ni is not None) else math.nan,
            ]
            dg = digits[a]
            if dg.sum() >= 20:
                pobs = dg / dg.sum()
                mad = float(np.abs(pobs - BENFORD).mean())
                chi2 = float((dg.sum() * ((pobs - BENFORD) ** 2 / BENFORD)).sum())
            else:
                mad = chi2 = math.nan
            rs = rounds[a]
            row += [mad, chi2, float(dg.sum()), (rs[0] / rs[1]) if rs[1] >= 10 else math.nan,
                    float(nfacts[a]), float(len(ncustom[a])),
                    len(ncustom[a]) / nfacts[a] if nfacts[a] else math.nan]
            try:
                filed = datetime.strptime(s["filed"], "%Y%m%d").date()
                pend = datetime.strptime(s["period"], "%Y%m%d").date()
            except ValueError:
                continue
            lag = (filed - pend).days
            cik = str(int(s["cik"]))
            prior = [h for h in hist.get(cik, []) if h[0] < filed]
            prior3 = [h for h in prior if (filed - h[0]).days <= 1095]
            last_k = [h for h in prior if h[1] == "10-K"]
            row += [
                float(lag), float(lag - DEADLINE.get(s["afs"], 90)),
                float(AFS_ORD.get(s["afs"], math.nan)), float(s["wksi"] == "1"),
                # sub.txt also carries prevrpt, "subsequently amended" -- that is
                # knowledge from after the filing date and is deliberately left out.
                float(s["nciks"] or 1),
                float((s["countryba"] or "US") != "US"),
                float(bool(last_k) and last_k[-1][2] != s["fye"]),
                float(sum(1 for h in prior3 if h[1] == "10-K/A")),
                float(sum(1 for h in prior3 if h[1] == "10-K")),
            ]
            X.append(row); adsh_out.append(a); n += 1
        print(f"  {p.stem}  {n:>4}", flush=True)
    X = np.array(X, dtype=float)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, X=X, adsh=np.array(adsh_out), names=np.array(names))
    print(f"\n{X.shape[0]:,} filings x {X.shape[1]} features -> {OUT}")
    cov = np.isfinite(X).mean(axis=0)
    print("coverage: " + ", ".join(f"{nm} {100*c:.0f}%" for nm, c in zip(names, cov)
                                    if nm.endswith("_cur") or not nm[-4:] in ("_pri", "_chg")))


if __name__ == "__main__":
    main()
