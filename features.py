#!/usr/bin/env python3
"""Features for restatement prediction, beyond the seven the F-score uses.

The F-score's variables were chosen in 2011 and its coefficients fitted on
1979-2002 data. Two different things could be wrong with it today: the
coefficients could be stale, or the variables themselves could be insufficient.
Those need separating, so this keeps the original seven intact and adds a second
block around them -- size, leverage, growth, margins, and the accrual and asset-
quality ratios that the Beneish M-score is built from.

Anything derived from the filing itself is fair game. Anything that would not
have been knowable at filing time is not, and there is nothing here that fails
that test.
"""

import math

# Everything the F-score already provides, kept under its own names.
FSCORE_VARS = ["rsst_acc", "ch_rec", "ch_inv", "soft_assets", "ch_cs",
               "ch_roa", "issue"]

EXTRA_VARS = [
    "log_assets", "leverage", "cash_ratio", "asset_growth", "rev_growth",
    "net_margin", "roa", "accruals_ta", "dsri", "gmi", "aqi", "sgi", "depi",
    "lvgi", "wc_ta", "rec_rev", "inv_rev", "ppe_ta", "loss", "neg_equity",
]

ALL_VARS = FSCORE_VARS + EXTRA_VARS


def _safe(num, den, cap=50.0):
    """Ratios blow up on tiny denominators; a shell company is not a signal."""
    if den is None or abs(den) < 1e-6:
        return None
    v = num / den
    if not math.isfinite(v) or abs(v) > cap:
        return None
    return v


def extra(cur, pri):
    """The second feature block, or None when the filing cannot support it."""
    z = lambda d, k: d.get(k) or 0.0
    ta_c, ta_p = cur.get("assets"), pri.get("assets")
    rev_c, rev_p = cur.get("revenue"), pri.get("revenue")
    ni_c = cur.get("net_income")
    if not ta_c or not ta_p or rev_c is None or ni_c is None:
        return None

    equity_c = ta_c - z(cur, "liabilities")
    debt_c = z(cur, "st_debt") + z(cur, "lt_debt")
    debt_p = z(pri, "st_debt") + z(pri, "lt_debt")

    f = {
        "log_assets": math.log10(max(ta_c, 1.0)),
        "leverage": _safe(debt_c, ta_c),
        "cash_ratio": _safe(z(cur, "cash"), ta_c),
        "asset_growth": _safe(ta_c - ta_p, ta_p),
        "rev_growth": _safe(rev_c - (rev_p or 0), abs(rev_p) if rev_p else None),
        "net_margin": _safe(ni_c, abs(rev_c) if rev_c else None),
        "roa": _safe(ni_c, ta_c),
        # Total accruals as a share of assets: net income less the cash it did
        # not come with, which is where aggressive recognition shows up.
        "accruals_ta": _safe(ni_c - (z(cur, "cash") - z(pri, "cash")), ta_c),
        "wc_ta": _safe(z(cur, "assets_current") - z(cur, "liabilities_current"), ta_c),
        "rec_rev": _safe(z(cur, "receivables"), abs(rev_c) if rev_c else None),
        "inv_rev": _safe(z(cur, "inventory"), abs(rev_c) if rev_c else None),
        "ppe_ta": _safe(z(cur, "ppe"), ta_c),
        "loss": 1.0 if ni_c < 0 else 0.0,
        "neg_equity": 1.0 if equity_c < 0 else 0.0,
    }

    # Beneish-style indices: each is this year over last year, so a value above
    # one means the ratio moved in the direction associated with manipulation.
    dso_c = _safe(z(cur, "receivables"), abs(rev_c) if rev_c else None)
    dso_p = _safe(z(pri, "receivables"), abs(rev_p) if rev_p else None)
    f["dsri"] = _safe(dso_c, dso_p) if (dso_c is not None and dso_p) else None

    gm_c = _safe(rev_c - z(cur, "inventory"), abs(rev_c) if rev_c else None)
    gm_p = _safe((rev_p or 0) - z(pri, "inventory"), abs(rev_p) if rev_p else None)
    f["gmi"] = _safe(gm_p, gm_c) if (gm_p is not None and gm_c) else None

    # Asset quality: the share of assets that is neither PP&E nor current.
    aq_c = _safe(ta_c - z(cur, "assets_current") - z(cur, "ppe"), ta_c)
    aq_p = _safe(ta_p - z(pri, "assets_current") - z(pri, "ppe"), ta_p)
    f["aqi"] = _safe(aq_c, aq_p) if (aq_c is not None and aq_p) else None

    f["sgi"] = _safe(rev_c, abs(rev_p) if rev_p else None)
    dep_c = _safe(z(cur, "ppe"), ta_c)
    dep_p = _safe(z(pri, "ppe"), ta_p)
    f["depi"] = _safe(dep_p, dep_c) if (dep_p is not None and dep_c) else None

    lv_c = _safe(z(cur, "liabilities"), ta_c)
    lv_p = _safe(z(pri, "liabilities"), ta_p)
    f["lvgi"] = _safe(lv_c, lv_p) if (lv_c is not None and lv_p) else None
    return f


def vector(fvars, xvars):
    """One flat feature row, with missing values marked rather than guessed."""
    row, mask = [], []
    for k in ALL_VARS:
        v = (fvars or {}).get(k) if k in FSCORE_VARS else (xvars or {}).get(k)
        if v is None or not math.isfinite(v):
            row.append(0.0)
            mask.append(1.0)
        else:
            row.append(float(v))
            mask.append(0.0)
    return row + mask


FEATURE_NAMES = ALL_VARS + [f"{k}__missing" for k in ALL_VARS]
