#!/usr/bin/env python3
"""The Dechow, Ge, Larson and Sloan (2011) F-score, computed from SEC XBRL data.

Model 1 of "Predicting Material Accounting Misstatements" uses financial
statement variables only:

    predicted = -7.893 + 0.790*rsst_acc + 2.518*ch_rec + 1.191*ch_inv
              + 1.979*soft_assets + 0.171*ch_cs - 0.932*ch_roa + 1.029*issue

    F-score   = P(misstatement) / 0.0037

An F-score above 1 means the filing looks more like a misstatement than the
average firm-year does. This is the published baseline any model has to beat, so
it is implemented first: if the data cannot reproduce a peer-reviewed measure,
the problem is the data, not the model.

Two things make this messier than the formula suggests. Filers do not agree on
tag names -- cash appears under at least four -- so every concept carries a
fallback list. And the model needs year-over-year changes, which are available
inside a single filing because a 10-K restates the prior year as a comparative.
"""

import csv
import io
import zipfile
from collections import defaultdict

# Model 1 coefficients, Dechow et al. (2011), Table 7.
COEF = {"const": -7.893, "rsst_acc": 0.790, "ch_rec": 2.518, "ch_inv": 1.191,
        "soft_assets": 1.979, "ch_cs": 0.171, "ch_roa": -0.932, "issue": 1.029}
UNCONDITIONAL = 0.0037

# Canonical concept -> the US-GAAP tags filers actually use, best first.
TAGS = {
    "assets": ["Assets"],
    "assets_current": ["AssetsCurrent"],
    "liabilities": ["Liabilities"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
             "Cash"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent",
                    "AccountsAndOtherReceivablesNetCurrent",
                    "AccountsReceivableGrossCurrent"],
    "inventory": ["InventoryNet", "InventoryGross"],
    "ppe": ["PropertyPlantAndEquipmentNet"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet", "SalesRevenueGoodsNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueServicesNet", "RevenueFromContractWithCustomerNet",
                "RevenuesNetOfInterestExpense", "TotalRevenues"],
    "net_income": ["NetIncomeLoss", "ProfitLoss",
                   "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "st_debt": ["ShortTermBorrowings", "DebtCurrent", "LongTermDebtCurrent",
                "OtherShortTermBorrowings"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt",
                "LongTermDebtAndCapitalLeaseObligations"],
    "st_inv": ["ShortTermInvestments", "MarketableSecuritiesCurrent",
               "AvailableForSaleSecuritiesCurrent"],
    "lt_inv": ["LongTermInvestments", "MarketableSecuritiesNoncurrent",
               "AvailableForSaleSecuritiesNoncurrent"],
    "preferred": ["PreferredStockValue", "PreferredStockValueOutstanding"],
    "issue_stock": ["ProceedsFromIssuanceOfCommonStock",
                    "StockIssuedDuringPeriodValueNewIssues",
                    "ProceedsFromIssuanceOfPrivatePlacement"],
    "issue_debt": ["ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromNotesPayable",
                   "ProceedsFromIssuanceOfDebt"],
}
WANTED = {t for tags in TAGS.values() for t in tags}


def read_quarter(path):
    """(submissions, facts) from one quarterly zip.

    facts is adsh -> ddate -> concept -> value, consolidated figures only.
    """
    z = zipfile.ZipFile(path)

    def rows(name):
        with z.open(name) as f:
            yield from csv.DictReader(
                io.TextIOWrapper(f, encoding="utf-8", errors="replace"), delimiter="\t")

    subs = {r["adsh"]: r for r in rows("sub.txt")}
    tag_to_concept = {t: c for c, tags in TAGS.items() for t in tags}
    rank = {t: i for c, tags in TAGS.items() for i, t in enumerate(tags)}

    facts = defaultdict(lambda: defaultdict(dict))
    best = defaultdict(lambda: defaultdict(dict))
    for r in rows("num.txt"):
        tag = r["tag"]
        if tag not in WANTED:
            continue
        if r["segments"] or r["coreg"] or r["uom"] != "USD":
            continue
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            continue
        concept = tag_to_concept[tag]
        key = (r["adsh"], r["ddate"])
        # Flow items are reported over a period, stocks at a point in time.
        want_flow = concept in ("revenue", "net_income", "issue_stock", "issue_debt")
        if want_flow and r["qtrs"] not in ("4",):
            continue
        if not want_flow and r["qtrs"] != "0":
            continue
        prior = best[key].get(concept)
        if prior is None or rank[tag] < prior:
            best[key][concept] = rank[tag]
            facts[r["adsh"]][r["ddate"]][concept] = v
    return subs, facts


def is_financial(sic):
    """Banks and insurers are excluded, as in the original paper.

    Their balance sheets do not have the structure the model assumes -- loans
    are not receivables, there is no inventory, and "revenue" is interest
    income reported under entirely different tags.
    """
    try:
        return 6000 <= int(sic) <= 6999
    except (TypeError, ValueError):
        return False


def _avg(a, b):
    vals = [x for x in (a, b) if x is not None]
    return sum(vals) / len(vals) if vals else None


def variables(cur, pri):
    """The seven Model 1 inputs, or None when the filing lacks what they need."""
    g = lambda d, k: d.get(k)
    ta_c, ta_p = g(cur, "assets"), g(pri, "assets")
    if not ta_c or not ta_p:
        return None
    avg_ta = _avg(ta_c, ta_p)
    if not avg_ta:
        return None

    def z(d, k):                      # a missing line item is a zero balance
        return d.get(k) or 0.0

    def wc(d):
        return (z(d, "assets_current") - z(d, "cash")) - \
               (z(d, "liabilities_current") - z(d, "st_debt"))

    def nco(d):
        return (z(d, "assets") - z(d, "assets_current") - z(d, "lt_inv")) - \
               (z(d, "liabilities") - z(d, "liabilities_current") - z(d, "lt_debt"))

    def fin(d):
        return (z(d, "st_inv") + z(d, "lt_inv")) - \
               (z(d, "lt_debt") + z(d, "st_debt") + z(d, "preferred"))

    rsst = ((wc(cur) - wc(pri)) + (nco(cur) - nco(pri)) + (fin(cur) - fin(pri))) / avg_ta
    ch_rec = (z(cur, "receivables") - z(pri, "receivables")) / avg_ta
    ch_inv = (z(cur, "inventory") - z(pri, "inventory")) / avg_ta
    soft = (ta_c - z(cur, "ppe") - z(cur, "cash")) / ta_c

    # Cash sales strip out the change in receivables from revenue.
    rev_c, rev_p = g(cur, "revenue"), g(pri, "revenue")
    if rev_c is None or rev_p is None:
        return None
    cs_c = rev_c - (z(cur, "receivables") - z(pri, "receivables"))
    cs_p = rev_p
    ch_cs = (cs_c - cs_p) / abs(cs_p) if cs_p else 0.0

    ni_c, ni_p = g(cur, "net_income"), g(pri, "net_income")
    if ni_c is None or ni_p is None:
        return None
    ch_roa = (ni_c / avg_ta) - (ni_p / ta_p)

    issue = 1.0 if (z(cur, "issue_stock") > 0 or z(cur, "issue_debt") > 0) else 0.0

    v = {"rsst_acc": rsst, "ch_rec": ch_rec, "ch_inv": ch_inv,
         "soft_assets": soft, "ch_cs": ch_cs, "ch_roa": ch_roa, "issue": issue}
    # Extreme values are almost always a units or tagging error, not a signal.
    if any(abs(x) > 50 for k, x in v.items() if k != "issue"):
        return None
    return v


def fscore(v):
    """Predicted probability and the F-score it implies."""
    import math
    lin = COEF["const"] + sum(COEF[k] * v[k] for k in v)
    p = 1.0 / (1.0 + math.exp(-lin)) if -700 < lin < 700 else (0.0 if lin < 0 else 1.0)
    return p, p / UNCONDITIONAL
