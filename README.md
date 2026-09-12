# Predicting SEC restatements from XBRL financial data

Can a model read a company's 10-K and say, before anyone else does, that the
numbers in it will later have to be withdrawn?

This project builds that model from free public data only, evaluates it on a
strict out-of-time test, and compares it against the standard published
baseline (the Dechow et al. 2011 F-score). Everything runs on one laptop with a
6 GB GPU. Nothing was paid for.

## Questions a reviewer will ask first

- **Is the split temporal?** Yes. Train on 10-Ks filed before 2019, test on
  the 17,755 filed 2019–2023. Every selection decision used the 2017 and
  2018 validation years; the test set was scored once per reported row.
- **What exactly is the label?** An SEC Form 8-K Item 4.02 (non-reliance on
  previously issued financial statements) filed within three years of the
  10-K. Base rate 5.2%. Most are errors, not fraud.
- **How are gains judged?** A paired bootstrap on the same test filings; a
  block is claimed only if its interval excludes zero. Firm-clustered
  intervals are reported for the headline.
- **Could the model see the future?** Five ways it did were found and
  removed, each with its cost measured: an amendment flag, a text-parser
  fingerprint, 10-K text matched a year late, a price feed that knew about
  later delistings and splits, and comment letters dated before they were
  public. See "What did not go wrong, because we checked".
- **What would a forecaster on 1 January 2019 have scored?** 0.724, not
  0.748: training labels carry hindsight. See "What a forecaster would have
  known".
- **Is it memorising firms?** On filings from companies never seen restating
  it scores 0.740.
- **What did not work?** About sixty things, each with its number, in "What
  was tried, and what moved".

## Result

**From the financial statements alone: AUC 0.737 on 17,755 held-out 10-Ks filed
2019–2023, against 0.542 for the published F-score on the same filings. With
every free public source added — 8-K events, insider trades, prices, the SEC's
letters, and the company's own statements about its internal controls — 0.748
on the same filings.**

| model | inputs | AUC | 95% CI | top-100 hit rate* |
|---|---|---|---|---|
| Dechow F-score, published coefficients | 7 ratios | 0.542 | | |
| Dechow F-score, refit on our training data | 7 ratios | 0.541 | | |
| gradient boosting | 28 ratios | 0.691 | 0.674–0.707 | 19% |
| balanced random forest, default settings | 28 ratios + 51 raw items | 0.717 | 0.701–0.733 | 18% |
| balanced random forest, default settings | + 147 more items & ratios, Benford, filing behaviour (242) | 0.727 | 0.712–0.743 | 22% |
| **balanced random forest, tuned (1:10 per tree, 15 features per split)** | **the same 242 statement features** | **0.737** | **0.721–0.752** | **24%** |
| tuned forest, separate line | + auditor/CFO 8-Ks, insider trading, 12-month market ratios (266) | 0.740 | 0.724–0.756 | 25% |
| tuned forest, separate line | + SEC comment letters (dated by their public day) and late-filing notices (276) | 0.741 | 0.726–0.756 | 27% |
| tuned forest, full free stack | + the 10-K's own controls, auditor and litigation disclosures (290) | 0.745 | 0.730–0.760 | 25% |
| **tuned forest, full free stack** | **+ thirty more disclosure flags (325)** | **0.748** | **0.733–0.762**; firm-clustered 0.724–0.771 | **24%** |
| tuned forest | + year-over-year change block (338) | 0.748 | 0.734–0.764 | 25% |

\* share of the 100 highest-scored filings that were actually restated; base rate is 5.2%.
Precision at 500 for the full stack is 20.8%, at 1,000 18.0%. The events
block (+0.004 [+0.000, +0.007]), the ten flags (+0.003 [+0.001, +0.006]) and
the thirty flags (+0.003 [+0.001, +0.005]) pass the paired test on the full
test set. The letters block does not any more: dated by the day each letter
became public rather than the day it was written (see "Leakage through a
release date"), it is worth +0.001 [−0.001, +0.003] at its stage and the same
on the full stack, so, like the change block (+0.001 [−0.002, +0.003]), it is
shown and not claimed; without it the stack scores 0.747. The
learner is the same balanced random forest throughout; the two changes in
this table's lower half are the share of negatives each tree sees, 1:10
instead of 1:1, and the features tried per split, 15 instead of 26, both
chosen on the validation years (see "The learner"). Confidence intervals
resample filings; the headline also carries a firm-clustered interval, since
a company's five 10-Ks move together, and it is about a point wider each
side. The market block in this table is the audited version: its four
price-level columns are dropped and it is imputed without missing
indicators (see "Leakage through a price feed" below). An earlier version of
this table said 0.751. That number rested on 10-K text
matched a year late (see "Leakage through a year key" below) and was withdrawn;
every text number in this file was rebuilt from documents matched by accession
number.

The statement-only line is the clean scientific claim. The rows below it add
every free public source we could find — 8-K events, Forms 3/4/5 insider
trades, Yahoo prices, the SEC's comment letters and late-filing notices, and
finally what the 10-K itself says in Items 9A, 8 and 3 — each kept separate so
its contribution is visible. Financial-only to full stack is +0.012
[+0.007, +0.017], positive in every test year (+0.002, +0.009, +0.006, +0.020,
+0.015). Details below.

Every step from 0.691 to 0.727 is confirmed by a paired bootstrap on the same
test filings — resample, score both models on the resample, look at the
distribution of the difference:

| step | gain | 95% CI | P(gain ≤ 0) |
|---|---|---|---|
| 28 ratios → + 51 raw items, balanced forest | +0.026 | +0.017, +0.035 | 0.000 |
| → + 147 more items and ratios | +0.005 | −0.001, +0.011 | 0.041 |
| → + Benford and filing-shape statistics | +0.007 | +0.001, +0.013 | 0.011 |
| → + filing behaviour (the full 242) | +0.011 | +0.005, +0.017 | 0.001 |

The gain holds in every test year and survives dropping 2021, the year of the
SPAC warrant restatement wave:

| test year | filings | restated | 28 ratios | 79 | 242 |
|---|---|---|---|---|---|
| 2019 | 3,501 | 128 | 0.677 | 0.740 | 0.746 |
| 2020 | 3,416 | 156 | 0.682 | 0.717 | 0.725 |
| 2021 | 3,477 | 211 | 0.677 | 0.697 | 0.718 |
| 2022 | 3,719 | 226 | 0.703 | 0.716 | 0.723 |
| 2023 | 3,642 | 206 | 0.705 | 0.726 | 0.731 |

For scale: the best published result for statement-only models is Bao, Ke, Li,
Yu & Zhang (2020, *Journal of Accounting Research*), about 0.72 AUC on SEC
enforcement actions (AAERs) for 2003–2008. Our label is different and noisier —
every Item 4.02 restatement, not just the frauds that drew enforcement — and
our years are different. We also ran their race on their own data (below): our
learner ties theirs, and three attempts to pass it failed. The statement-only
ceiling is about 0.72 for everyone; the gains here come from columns their
Compustat panel does not have.

With free public data and an honest split, the published frontier for the
AAER fraud label is 0.74 (a fine-tuned 8B LLM reading MD&A text, Waffo Dzuyo
et al. 2026), with classical models on public financials at 0.70–0.72. For the
restatement label on free data there is no published number; these are the
first, and the full stack's 0.748 sits at that frontier on a noisier label,
with a forest and regular expressions on a laptop. The one published model
built for this exact label with paid data (Bertomeu et al. 2021) scores 0.629
on the same 2019–2020 filings where ours scores well above it; the
head-to-head, with its caveats, is below.

## The learner

One balanced random forest throughout: 600 trees, minimum leaf 5, five seeds
averaged, median imputation with a missing indicator per column. Its one
tuned setting is how many negatives each tree sees. The imbalanced-learn
default draws negatives 1:1 with the positives; Perols, Bowen, Zimmermann &
Samba (2017) found about 1:4 better for rare-event fraud detection. Swept on
the validation years, tested once:

| positives : negatives per tree | val 2017 | val 2018 | mean | test AUC, full stack (295) | vs 1:1 (paired) |
|---|---|---|---|---|---|
| 1:1 | 0.792 | 0.762 | 0.777 | 0.737 | |
| 1:2 | 0.812 | 0.786 | 0.799 | 0.744 | +0.007 [+0.004, +0.010] |
| 1:4 | 0.819 | 0.799 | 0.809 | 0.746 | +0.009 [+0.005, +0.014] |
| **1:10** | 0.831 | 0.808 | 0.8195 | **0.745** | +0.008 [+0.001, +0.015] |
| 1:20 | 0.829 | 0.813 | 0.821 | 0.740 | +0.003 |
| 1:50 and beyond | | | | | not feasible: the early training windows have too few negatives |

Validation peaks between 1:10 and 1:20, 0.0015 apart, inside one seed's
noise; the one-standard-error convention takes the less extreme setting in
that band, so the forest is 1:10. Stated plainly: the plain "best mean
validation" rule would take 1:20 and a headline of 0.740, and that rule was
the one first declared; the tie-break was chosen after both test numbers
were visible. A reader who prefers 0.740 has the table. Hierarchical
shrinkage of the same forest (Agarwal et al. 2022) hurt at every λ.

The second tuned setting is how many features each split may try. sklearn's
default is the square root of the column count, 26 of the 660 columns
(features plus missing indicators). Swept on the validation years at the
1:10 ratio, on the 330-column headline stack:

| features per split | val 2017 | val 2018 | mean | test AUC | vs 26 (paired) |
|---|---|---|---|---|---|
| 231 (0.35 of columns) | 0.804 | 0.788 | 0.796 | 0.725 | −0.022 |
| 132 (0.2 of columns) | 0.810 | 0.795 | 0.803 | 0.733 | −0.014 |
| 26 (sqrt, the default) | 0.830 | 0.810 | 0.820 | 0.747 | |
| **15** | 0.833 | 0.810 | 0.8215 | **0.749** | **+0.003 [+0.001, +0.004]**, P 0.003 |
| 12 | 0.834 | 0.810 | 0.822 | 0.750 | +0.003 [+0.001, +0.005], P 0.004 |
| 9 (log2) | 0.833 | 0.811 | 0.822 | 0.749 | +0.002 [−0.001, +0.005] |
| 7 | 0.833 | 0.807 | 0.820 | 0.750 | +0.003 [−0.000, +0.006] |
| 5 | 0.833 | 0.807 | 0.820 | 0.748 | +0.001 |
| 3 | 0.833 | 0.801 | 0.817 | 0.744 | −0.003 |

Fewer features per split is better down to about a dozen, then it turns.
Validation ties from 15 down to 9; the same one-standard-error convention as
for the ratio takes the setting nearest the default inside that band, 15.
Everything below the ratio row of the top table uses this forest. Many
settings were tried in one evening (`algo_wins.py`, `mf_sweep.py`: leaf
size, tree count, seeds, a rank-average across ratios, a LightGBM blend,
label-quality weights, training on the tight label), and only this one and
the ratio passed the paired test; the rest are in the ledger.

One more thing the ratio does not do, found by rerunning the as-of protocol
with it: trained on labels as known on 1 January 2019, the tuned forest
scores 0.724 — what the default forest scored, to the noise. The whole gain from
tuning lives in the hindsight labels: a wider ratio lets each tree see more
of the 2018 filings whose restatements were announced later, and the 2019
test year (0.795 with hindsight, 0.705 without) is where it shows. The
headline is reported under the protocol every comparator in this file uses;
the deployable number did not move. Script:
`simple_wins.py`, `ratio_more.py`; the ratio is `BRF_STRATEGY` in
`events_test.py`.

## Head-to-head with the paid-data model, same filings, same label

Bertomeu, Cheynel, Floyd & Pan (2021, *Review of Accounting Studies*) predict
the same thing — Item 4.02 non-reliance filings — with a gradient-boosted
tree on about fifty predictors from Compustat, CRSP, Audit Analytics and
I/B/E/S (returns, volatility, spreads, analyst forecast error and dispersion,
audit fees, tenure, opinions, short interest, plus the accounting ratios),
trained on fiscal 2001–2010. They publish their out-of-sample scores for
2011–2019 by gvkey and fiscal year. Linked to CIK through the farr table, their
score and ours land on the same 10-Ks: the 4,399 filings for fiscal 2018–2019
(filed 2019–2020) that fall inside our test period, where both models are
out-of-sample.

| label | Bertomeu et al. (paid data) | ours, statements only (242) | ours, full free stack (290) | ours, trained on 2014–16 only | ours, full stack, labels as known 1 Jan 2019 |
|---|---|---|---|---|---|
| any Item 4.02 within 3 years (ours; 154 positives) | 0.629 | 0.766 (+0.138 [+0.092, +0.185]) | **0.770 (+0.141 [+0.096, +0.186])** | 0.689 (+0.060 [+0.011, +0.107]) | 0.700 (+0.070 [+0.020, +0.121]) |
| the 10-K's own year later condemned (closest to theirs; 76 positives) | 0.649 | 0.692 (+0.043 [−0.028, +0.115], tie) | 0.698 (+0.050 [−0.019, +0.122], tie) | 0.649 (−0.001, tie) | 0.637 (−0.012, tie) |

Parentheses are the paired difference against their score on the same rows.
The last column removes the hindsight in our training labels (see "What a
forecaster would have known"). On our label a rank blend of the two models
scores below ours alone: their score carries nothing ours lacks. On their
label definition the blend is level with ours.

What it says: on the label this project predicts, free data beats the paid
model by fourteen points and the win survives every handicap we could give it.
On their own, stricter label definition — the 10-K's own fiscal year named
as misstated, 76 positives among 4,399 filings — the two models tie, and
the interval is wide enough to hide a real difference either way. What it
does not say, stated plainly: their model is eight years staler than ours;
the overlap is the Compustat-covered, larger-firm subset; their label
construction may differ from our reconstruction of it; and the 2020 filings
they cover are the last of their sample. An earlier version of this table
claimed +0.163 and +0.095; the difference was year-late 10-K text, since
withdrawn. The tuned forest and the audited stack throughout this table. Script: `bertomeu_compare.py`; their scores and code are linked from
their measure page (Google Sites, "Restatement Risk Measure").

## Bao et al. on their own track

Bao, Ke, Li, Yu & Zhang published their labelled panel and MATLAB. We ran
their race: their data, their splits (train 1991 to *t*−2, test year *t* for
*t* = 2003–2008), their serial-fraud step, their metrics.

| learner, their 28 raw Compustat items | avg AUC 2003–08 | NDCG@1% |
|---|---|---|
| Bao et al. 2020, as published | 0.725 | 0.049 |
| Bao et al. 2022 erratum | 0.723 | 0.024 |
| Walker 2022, rerunning their code | 0.721–0.723 | 0.017–0.018 |
| their RUSBoost, our reproduction (imbalanced-learn) | 0.711–0.714 | 0.012–0.020 |
| our balanced random forest | 0.717 | 0.008 |
| our gradient booster | 0.722 | 0.003 |

Pooled over the six test years and paired against our RUSBoost reproduction,
the forest is +0.006 [−0.004, +0.017] — a tie. Then three pre-declared
attempts to do better, each selected on validation years inside the training
period and run on the test years once:

| attempt | selected on | validation margin over RUSBoost | test | paired vs RUSBoost |
|---|---|---|---|---|
| prior-year values and changes + their ratios, forest | 2001 | +0.039 | 0.708 vs 0.714 | −0.011 [−0.031, +0.008] |
| same candidates, must win 3 of 4 years | 1998–2001 | +0.069 | 0.708 vs 0.714 | −0.012 [−0.031, +0.008] |
| + firm-level score smoothing over prior years, train from 1996 | 1998–2001 | +0.101 | 0.702 vs 0.714 | −0.016 [−0.038, +0.005] |
| + free EDGAR filing behaviour (lag, NT 10-K notices, amendments, 8-K counts, firm age), forest | 1998–2001 | +0.085 | 0.683 vs 0.706 | −0.023 [−0.044, −0.004] |
| same columns, *their* RUSBoost | — | — | 0.702 vs 0.706 | +0.004 [−0.014, +0.022] |

The fourth attempt added information their panel lacks: how each firm filed,
reconstructed from EDGAR's 1993–2009 indices through the farr package's
gvkey–CIK link (60% of firm-years matched to an annual report once `10KSB`,
`20-F` and `40-F` were included; every model scored on those rows only, since
the match itself tracks the label). Filing behaviour was the strongest block on
our restatement data. On AAER fraud in 1993–2008 it carries nothing: the forest
does worse with it, their learner is unchanged. Enforcement targets are large
firms that file on time, and the `NT 10-K` late notice only became routine after
2002.

Every candidate beat RUSBoost by four to ten points on 1998–2001 and lost on
2003–2008. That is not selection noise — the four-year rule and
the win-three-of-four filter exist to remove it. The 1990s and the 2000s are
different regimes, and features that fit one do not carry to the other. On
this data, 0.72 is where every competent learner lands, and we matched it
without passing it. The test years were touched three times; all three are
reported. Scripts: `race.py`, `race2.py`, `race3.py`, `race4.py`,
`edgar_behavior.py` + `race5.py`.

What transfers: validate on several years, never one; fraud is a firm trait
(smoothing added +0.02 to their own model on validation, though not on test);
and the ceiling on statement-only inputs is real. The room is in the columns.

## Beyond the statements: everything free we could add

The published models that clear 0.75 add market, audit and governance data
from paid sources (CRSP, Audit Analytics). We added the free equivalents and
measured each in turn, selecting on both validation years (2017 and 2018)
before touching the test set once per stage:

- **8-K events** — Item 4.01 auditor changes (13,773) and CFO-related Item 5.02
  filings (91,100), 2011–2023, from EDGAR full-text search. Counts in the
  prior one and three years, days since the last.
- **Insider trading** — the SEC's Forms 3/4/5 bulk data sets, 1.33 million
  open-market trades across 9,830 issuers. Sales, purchases, dollar values,
  distinct sellers, officer sales, last-90-day activity, all before the filing.
- **Market** — daily prices from Yahoo Finance via the SEC's CIK–ticker map, for
  the twelve months before each filing: returns, market-relative return,
  volatility, idiosyncratic volatility, drawdown, turnover, size,
  book-to-market, penny-stock flag.

| stage | n | val 2017 | val 2018 | test AUC | vs financial-only (paired) |
|---|---|---|---|---|---|
| **full sample**, market holes as NaN (optimistic) | | | | | |
| financial statements only | 242 | 0.785 | 0.760 | 0.727 | — |
| + 8-K events | 248 | 0.786 | 0.757 | 0.729 | +0.002 [+0.000, +0.003] |
| + insider trading | 258 | 0.788 | 0.760 | 0.729 | +0.002 [+0.000, +0.004] |
| + market | 270 | 0.783 | 0.757 | **0.731** | +0.003 [+0.001, +0.005] |
| **covered subsample** — the 19,933 filings with prices, every stage on the same rows | | | | | |
| financial statements only | 242 | 0.782 | 0.758 | 0.720 | — |
| + 8-K events | 248 | 0.783 | 0.759 | 0.722 | +0.002 [+0.000, +0.003] |
| + insider trading | 258 | 0.783 | 0.758 | 0.722 | +0.002 [−0.001, +0.004] |
| + market | 270 | 0.784 | 0.758 | 0.727 | +0.006 [+0.003, +0.009] |

- **The regulator's attention** — from EDGAR's 2010–2023 form indices: SEC
  comment letters (`UPLOAD`) and the company's replies (`CORRESP`) in the prior
  one and two years, days since the last letter, `NT 10-K` / `NT 10-Q`
  late-filing notices, 10-K/A and 10-Q/A amendments, 8-K counts. Letters are
  now dated by the day they became public (from EDGAR's daily indices,
  `fetch_daily_index.py`), not the day they were written; the table below
  is the first version, on letter dates, and overstates the block — see
  "Leakage through a release date".

| stage | n | val 2017 | val 2018 | test AUC | paired |
|---|---|---|---|---|---|
| financial statements only | 242 | 0.785 | 0.760 | 0.727 | — |
| + letters and notices | 252 | 0.788 | 0.764 | 0.730 | +0.002 [+0.001, +0.004] |
| full free stack (events, insider, market) | 270 | 0.783 | 0.757 | 0.731 | — |
| **full free stack + letters and notices** | 280 | 0.786 | 0.759 | **0.734** | +0.003 [+0.002, +0.005] |

On letter dates the block improved *both* validation years as well as the
test set; on public dates it is +0.001 [−0.001, +0.003] on the tuned stack
and is no longer claimed. An `NT 10-K` late-filing notice in the prior two
years remains the strongest single outside signal found: 8.85% of those
filings restate, against 3.48% without (2.5×; an auditor change is 2.4×). A
comment letter public in the prior two years: 4.74% vs 4.07%.

Four outside data sources, all with published support, together move the
model 0.003–0.007. Each is statistically clear of zero and practically small.
Some of what they found:

- **Insider selling runs the wrong way.** Filings preceded by insider sales
  restate at 3.3%; filings with none at 6.1%. Insiders sell at established
  firms with stock-compensation plans; the shells and micro-caps that restate
  most have no insider activity at all. It is a size and legitimacy proxy the
  balance sheet already supplies, and it adds nothing on top.
- **The market knows a little.** Returns and volatility before the filing add
  +0.006 where prices exist, but nothing on either validation year, which is
  the standard the Bao race taught us to require. The published market gains
  come from CRSP, which covers delisted firms; Yahoo covers 56% of ours, and
  the uncovered 44% restate more (4.9% vs 4.1%) — the survivorship hole, which
  is why the two-way table exists.
- **The statements were carrying almost everything.** Filing lag and filer
  class, both free metadata, did more than all three outside sources combined.

Scripts: `fetch_events.py`, `event_features.py`, `fetch_prices.py`,
`market_features.py`, `insider_features.py`, `record_test.py`,
`letters_features.py`, `letters_test.py` (needs the EDGAR quarterly form
indices in `data/raw/edgar_index/`).

## The filer's own confession: Items 9A, 8 and 3

The label is misstatement times detection, and every block above describes
the misstatement. The comment letters were the first block to describe
detection — someone already looking. The loudest detector turned out to be
the company itself: Item 9A of every 10-K states whether internal control over
financial reporting was effective and names any material weakness; Item 8
carries the auditor's report; Item 3 the litigation. Every 10-K in the panel
was fetched whole (`fetch_all_10k.py`, 35,610 documents, matched by accession
number) and the sections cut with our own extractor (`scrutiny_v3_clean.py`);
the same flags cut by EDGAR-CRAWLER, the open-source extractor behind
EDGAR-CORPUS, agree with ours on 99% of filings and score the same.

| flag, from the 10-K's own text | restatement rate when present | when absent | ratio | AUC alone |
|---|---|---|---|---|
| **controls not effective** (Item 9A) | **9.3%** | 3.3% | **2.8×** | 0.605 |
| remediation mentioned | 8.6% | 3.7% | 2.3× | 0.576 |
| auditor's adverse opinion on controls | 8.4% | 4.3% | 1.9× | |
| going-concern doubt | 6.8% | 3.7% | 1.8× | 0.567 |
| Big 4 auditor | 2.3% | 7.1% | 0.3× | |
| this 10-K revises last year's net income by >5% | 9.1% | 4.3% | 2.1× | |

The last row is the self-verification signal built into the filing: the
comparative column quietly correcting the prior year, visible the day the
filing lands. Filing lag remains the strongest single feature at 0.679 alone;
"controls not effective" is one sentence and scores 0.605.

Added to the full stack, selected on both validation years, tested once on
the full test set (17,646 of 17,755 test rows have flags; the rest are
filings whose saved document had no Item 9A or Item 8):

| stage (tuned forest) | val 2017 | val 2018 | test AUC | 95% CI | paired |
|---|---|---|---|---|---|
| stack without flags (276) | 0.828 | 0.806 | 0.741 | 0.726–0.756 | — |
| **with the ten flags (290)** | 0.832 | 0.809 | **0.745** | 0.730–0.760 | **+0.003 [+0.001, +0.006]**, P(≤0) 0.002 |
| **+ thirty more flags (325)** | 0.831 | 0.811 | **0.748** | 0.733–0.762 | **+0.003 [+0.001, +0.005]**, P(≤0) 0.000 |

At the 1:1 forest the same steps are +0.003 [+0.001, +0.005] and +0.001
[−0.001, +0.002]. Real, small, and a fifth of what it first appeared to be:
the first version of this block took its 2014–2020 text from EDGAR-CORPUS
matched by filing year, which for most companies is the *next* year's 10-K,
and scored +0.017. Matched by accession number the ten flags are worth
+0.003 to +0.004. The story of that leak is in "What did
not go wrong". The thirty further flags (weakness types, tenure, critical
audit matters, corrections in the notes, investigations, delisting notices,
shell history, emerging-growth status) are individually real — a company that
says its *disclosure* controls are not effective restates at 9.4% against
3.4%; an emerging-growth company at 7.5% against 4.0% — and worth +0.002 on
top of the ten at the tuned forest, which passes the paired test and puts
them in the headline.

Scripts: `fetch_all_10k.py`, `refetch_main.py` (296 filings whose saved
document was an exhibit, re-fetched by document type), `scrutiny_features.py`,
`scrutiny_v3_clean.py`, `ec_extract.py`, `ec_features.py`, `headline_test.py`.

**Text as facts vs text as prose.** The same sections were also handed to a
fine-tuned ModernBERT-base as a regex-retrieved digest (`digest_text.py`,
`bert_mdna.py --digest`). Its one encouraging number, a 0.776 pilot, came from
the same year-late corpus text; on digests cut from each filing's own
document it scores 0.656 on 2019–23 and its fusion with the forest adds
nothing. Read as facts by regular expressions, the disclosures are worth
+0.003; read as prose by a 150M-parameter model on a laptop GPU, nothing that
survives.

## What changed since last year

Companies leave boilerplate alone until something forces an edit (Cohen,
Malloy & Nguyen 2020, "Lazy Prices"). With consecutive 10-Ks from the same
firm, each section's drift from last year and each flag's transition become
features, along with the days from year-end to the auditor's signature. Cut
by one extractor for every year (`ec_features.py`), 26,665 filings have a
same-parser prior year.

| feature | AUC alone | note |
|---|---|---|
| **controls newly not effective** (effective last year) | 0.531 | **10.1% restate vs 3.9%: 2.6×** |
| Item 7 / 8 text similarity to last year | 0.40–0.46 (inverse) | the section *changing* is the signal |
| **audit report lag** (year-end to the auditor's signature date) | **0.669** | as strong as filing lag; 48% coverage |

On the full test set the block adds +0.001 [−0.002, +0.003] to the headline
stack (−0.000 at the untuned forest): the transitions are real alone and already
implied by the flags and the filing lag. An earlier version of this section reported +0.008
on paired rows; that was the year-late corpus text again, and it is
withdrawn. Script: `change_features.py`, `ec_features.py`.

## The auditor's record

The PCAOB publishes every inspection report it issues — per firm, the share
of audits reviewed that had a Part I.A deficiency — and, since 2017, Form AP
names the firm and engagement partner on every issuer audit. Both are free
(`data/raw/pcaob/`). Per filing, the auditor's latest published deficiency
rate before the filing date, its trend, the inspection tier, auditor and
partner changes, other participating firms, and the audit report's lag from
year-end.

Alone, the auditor's record predicts: clients of auditors in the highest
deficiency-rate quartile restate at 7.5%, the lowest at 4.4%; annually
inspected (large) firms' clients at 3.5%, triennially inspected at 8.4%; an
auditor change at 10.0%; audit-report lag scores 0.701 by itself, the
strongest single feature in the project. On top of the full stack the block
adds +0.001 [−0.001, +0.002]: inspection tier is firm size and the report lag
is the filing lag, both already present. Who is checking matters; the
statements and the filing calendar already said who. Script:
`auditor_features.py`.

## What the company pays its auditor

Four papers say the audit fee predicts restatements: abnormally low fees
precede them (Blankley, Hurtt & MacGregor 2012), the unexplained part of the
fee predicts fraud, restatements and comment letters (Hribar, Kravet & Wilson
2014), and the non-audit and tax fee shares speak to independence (Frankel
et al. 2002; Kinney et al. 2004). The fee table is public: Item 14 of the
10-K for smaller filers, the proxy statement for the rest.
`fetch_proxy.py` located and fetched the proxy following each 10-K from
the EDGAR indices (28,015 documents); `fees_parse.py` reads the four
standard captions in table or prose form; `fees_features.py` builds audit,
audit-related, tax and other fees, their shares, the year's change, fee to
assets, and the residual of log fee on log assets fitted on training years.
Coverage 82% of the panel.

| rows | stack | + fees | paired |
|---|---|---|---|
| test filings with a fee table (14,727, 758 restated) | 0.749 | 0.752 | +0.003 [+0.001, +0.006], P 0.004 |
| every test filing (17,646, 915 restated) | 0.748 | 0.748 | +0.001 [−0.001, +0.002] |

Real on the rows that have it, gone when the 18% without a fee table — the
filers that restate most — are counted, so under the rule it is not
claimed. Alone, the strongest column is the fee level, and it runs the other
way from the folklore: the more a company pays its auditor the less it
restates, which is mostly size (0.609 inverted). The timing also matters and
the literature glosses over it: a proxy is filed up to eight months after
the 10-K, so for the 21,793 proxy-sourced rows the fees are not public on
the day the 10-K is scored. Audit Analytics takes its fees from the same
proxies, so the published fee results carry the same lag.

## Is it worth money?

The score was built to rank restatement risk. Out of sample it also ranks
who loses money. `backtest.py` buys each test-period 10-K's stock at the
close after filing and holds twelve months against SPY; `backtest/portfolio.py`
runs a monthly long-short portfolio (long the least-likely decile, short the
most-likely, equal-weighted, rebalanced monthly, each stock's signal its
latest 10-K score within a year) with a price floor, a delisting return of
−30% for names whose price history ends, transaction costs, borrow-fee
scenarios, and a regression on the Fama-French five factors plus momentum
(Ken French's library, free).

| decile of restatement score, stocks above $10 at entry | later restated | 12-month return vs SPY, mean | median | lost more than half |
|---|---|---|---|---|
| 1 | 0.3% | −1.1% | −1.9% | 3% |
| 5 | 1.2% | −4.4% | −8.4% | 11% |
| 9 | 8.9% | −18.1% | −35.2% | 38% |
| **10** | 14.8% | **−28.5%** | **−62.0%** | **59%** |

Decile 10 minus the rest: −22.6% [−31.8%, −12.7%], monotone across the
deciles. Below $5 the raw mean flips positive — a few penny stocks rose
fifty-fold in 2020 — which is the standard reason shorting microcaps blows up.

| monthly long-short, April 2019 – June 2024 | $10 floor | $5 floor |
|---|---|---|
| gross | +27.6%/yr, Sharpe 0.64, worst month −49% | +13.8%/yr, Sharpe 0.29, worst month −69% |
| net of 0.25% one-way costs and a 2% / 10% / 30% borrow fee | +24.4% / +16.4% / −3.6% | +10.7% / +2.7% / −17.3% |
| six-factor alpha (Newey-West) | +9.4%/yr, t = 0.5 | −5.5%/yr, t = −0.3 |
| factor loadings with t > 2 | SMB −1.5, HML +1.1, RMW +1.9, CMA −1.3 | HML +1.2, RMW +2.0, CMA −1.6 |

The long-only use needs no borrow desk: hold the equal-weighted universe and
drop the riskiest decile.

| long only, monthly, April 2019 – June 2024 | $10 floor | $5 floor |
|---|---|---|
| universe, equal-weighted | +8.2%/yr, Sharpe 0.34 | +10.3%/yr, Sharpe 0.41 |
| universe minus decile 10 | +10.5%/yr, Sharpe 0.45 | +11.3%/yr, Sharpe 0.47 |
| difference | +2.3%/yr, better in 73% of months, worst month −4.8% | +1.0%/yr |

By year the short leg returned −37%, +97%, +8%, −48%, −46%, −20% (2019 to
mid-2024): it earns steadily and loses everything in a squeeze year. Read
plainly: the return is real and large gross, it is almost entirely the
market's known aversion to small, unprofitable, distressed companies rather
than anything specific to restatements, the alpha left after the factors is
not distinguishable from zero on 63 months, and whether any of it can be
collected depends on borrow fees that free data cannot see. Pre-registered
forecast: 5–10 points of alpha before costs on the $10 universe — right on
size, wrong on significance. Caveats: Yahoo drops delisted tickers, so the
36% of top-decile filings with no price history are the short's best
outcomes and are missing from the buy-and-hold table (the portfolio charges
the delisting return only for names that vanish mid-history); SPY is a
generous benchmark for small caps, which the factor regression corrects.

## Little r: the revisions nobody announces

A 10-K states net income for fiscal year *P*. The next year's 10-K states it
again, as the comparative column. When the two differ, the company revised its
prior year — and if it did so without an Item 4.02, no label in this project
knew. Comparing every 10-K to its successor across the 40 quarters gives a
second label built from nothing but the filings themselves. The self-check is
built in: firms that *did* later file an Item 4.02 should trip it far more
often than firms that did not.

| definition of a mismatch | share flagged | among later Item 4.02 filers | among the rest | enrichment | predictable? (own AUC) | added to training, scored on Item 4.02 |
|---|---|---|---|---|---|---|
| any of net income, assets, liabilities, revenue, equity > 1% | 11.7% | 29.8% | 10.9% | 2.7× | 0.621 | 0.605 |
| any of the five > 5% | 7.7% | 24.6% | 6.9% | 3.6× | 0.650 | 0.665 |
| net income > 1% | 4.2% | 20.9% | 3.5% | 6.0× | 0.616 | 0.691 |
| **net income > 5%** | **3.0%** | **16.7%** | **2.4%** | **7.0×** | 0.668 | 0.704 |
| net income > 10% | 1.8% | 11.1% | 1.4% | 7.9× | 0.685 | 0.709 |
| net income > 5%, skipping the ASC 606/842 adoption years | 2.8% | 15.7% | 2.2% | 7.2× | 0.691 | 0.706 |

(Item 4.02 alone, same features, same test filings: 0.730.)

Three things this settles:

- **The numbers do testify about themselves.** A prior-year net income that
  moves by more than 5% in the next filing is seven times more common at
  firms that go on to announce a restatement. That is a clean, free,
  filing-derived signal that the loose "any item" version buried under
  standard adoptions and reclassifications.
- **Silent revisions are harder to predict than announced ones** — 0.67–0.69
  from the same features, against 0.73 for Item 4.02. They are smaller
  corrections with less of the pattern the forest keys on.
- **Adding them to the training label hurts.** Every definition lowers the
  Item 4.02 AUC (0.730 → 0.605–0.709). The announced restatement stays the
  training target; little r is a finding, a secondary evaluation label, and a
  tool for reading the filings — not a substitute.

Scripts: `littler.py`, `littler2.py`.

## What the model predicts

An Item 4.02 8-K is the filing a company must make when it concludes that
previously issued financial statements should no longer be relied upon. It is
the public, dated, unambiguous marker of a material restatement ("Big R").

A 10-K is labelled positive if the same company files an Item 4.02 within
three years (1,095 days) after it. Labels come from the future relative to the
features, so the model is genuinely predicting, not describing.

Most restatements are errors, not fraud. This model predicts restatements.

## Data

All from the SEC, all free:

- **Financial Statement Data Sets** — quarterly zips of every XBRL fact in every
  filing (`sub.txt`, `num.txt`). 40 quarters, 2014q1–2023q4.
- **EDGAR full-text search** — for every 8-K with Item 4.02, 2016–2026 (2,561
  announcements, the labels); and, for the separate line, every Item 4.01
  auditor change (13,773) and CFO-related Item 5.02 (91,100), 2011–2023.

After the standard exclusion of financial firms (SIC 6000–6999, whose balance
sheets do not have the structure the features assume) and requiring two years
of consolidated figures: **35,852 10-Ks**.

| split | filed | filings | restated | rate |
|---|---|---|---|---|
| fit | 2014–2016 | 11,185 | 389 | 3.5% |
| validation (model selection) | 2017–2018 | 6,912 | 278 | 4.0% |
| **test (touched once per claim)** | **2019–2023** | **17,755** | **927** | **5.2%** |

The base rate roughly doubles across the period, which is itself a reason to
split by time rather than at random.

The SEC requires a name and email in the `User-Agent` of every request. Scripts
read it from the `SEC_CONTACT` environment variable and refuse to run without
it; it is not stored anywhere in this repository.

## Features — all from the 10-K and its metadata

**28 ratios.** The seven Dechow F-score variables (RSST accruals, change in
receivables, change in inventory, soft assets, change in cash sales, change in
ROA, securities issuance), the F-score itself, and 20 more from the accounting
literature: leverage, growth, margins, Beneish M-score style indices (DSRI, GMI,
AQI, SGI, DEPI, LVGI), loss and negative-equity flags.

**61 raw statement items × 3.** Bao et al.'s finding, which surprised the
field, was that raw dollar amounts beat hand-built ratios: the ratios throw
information away, and trees can build whatever ratio they need. Every widely
reported line item — assets, cash, receivables, inventory, PP&E, goodwill,
payables, accruals, equity, retained earnings, revenue, COGS, SG&A, R&D,
operating income, tax, net income, the three cash-flow totals, capex,
share-based compensation, buybacks, acquisitions, shares, EPS — as signed
log10 for this year, last year, and the change. Plus 15 named ratios including
the Sloan accrual (net income less operating cash flow) and an EPS × shares vs
net income consistency gap.

**Benford's law and filing shape.** Over every dollar figure in the filing,
segments and company-specific tags included: mean absolute deviation and
chi-square of the leading-digit distribution from Benford's law (Amiram,
Bozanic & Rouen 2015), the share of round numbers, the number of facts, and
the share reported under company-invented XBRL tags. The one block where the
numbers testify about themselves.

**Filing behaviour.** Days from fiscal year-end to filing and how late that is
against the filer's own deadline (60/75/90 days by filer class), filer class,
well-known-seasoned-issuer status, whether the fiscal year-end moved, and how
many 10-Ks and 10-K/As the company filed in the prior three years. Filing lag
is the single most important feature in the final model.

Missing values stay missing. The boosters handle that natively; the forest
gets a median fill plus a missing-indicator column.

## What was tried, and what moved

Everything below was selected on the 2017–18 validation slice and reported on
the 2019–23 test set. The test set was never used to choose anything.

| lever | result | verdict |
|---|---|---|
| Dechow F-score, published coefficients | 0.542 | the baseline |
| refit the F-score's 7 variables | 0.541 | the variables are the limit, not the coefficients |
| gradient boosting on 28 ratios | 0.691 | first headline |
| multi-year history features (3-year trends) | no gain | |
| industry-relative percentiles (by SIC) | no gain | |
| hyperparameter search (80 configs), recency weights, bagging, logistic blend | +0.03 on validation, 0.00 on test | 278 validation positives cannot tune anything |
| 10-K text features (length, fog, tone, litigious/uncertain word rates) | text alone 0.61; increment +0.004 [−0.01, +0.02] | text proxies for size, which the financials carry |
| tighter labels (only the fiscal year each 8-K condemns) | 0.641 | *worse* — see below |
| LightGBM, plain / weighted / focal loss | 0.702 / 0.703 / 0.707 (79 features) | |
| CatBoost, plain / balanced | 0.689 / 0.656 | |
| RUSBoost (Bao et al.'s learner) | 0.654 | |
| EasyEnsemble | 0.696 | |
| **raw statement items + balanced random forest** | **0.717** | second headline |
| blend of forest + EasyEnsemble | 0.711 | the forest alone is better |
| TabPFN v2 foundation model, 3 bagged 2,000-row contexts | 0.693 [0.676–0.709] | sees 2,000 rows at a time; the forest sees 18,000 |
| + 147 more raw items and ratios | 0.722 | diminishing returns on the raw-item idea |
| + Benford's law and filing shape | 0.720 alone (+0.004) | small, real |
| + filing behaviour | 0.725 alone (+0.008) | filing lag is the top feature |
| **all three blocks** | **0.727** | **current headline** |
| + prior auditor-change and CFO 8-Ks (separate line) | 0.729 (+0.002) | auditor changes carry it; CFO turnover as harvested is noise |
| + insider trading, Forms 3/4/5 | 0.729 (+0.000) | inverse signal, redundant with size |
| + 12-month market data (Yahoo, 56% coverage) | 0.731 full / +0.006 on covered rows | small, real where prices exist; nothing on validation |
| + SEC comment letters, NT 10-K notices, amendments (EDGAR indices) | 0.734 (+0.003) on letter dates; +0.001 [−0.001, +0.003] once letters are dated by their public day | NT 10-K notices are 2.5× alone; the letters themselves were partly a look-ahead |
| + 10-K text flags: controls not effective, material weakness, auditor opinion, going concern, litigation | **0.737 (+0.003 [+0.001, +0.005])** | the company's own statement; first measured at +0.017 on year-late corpus text, withdrawn and rebuilt |
| + firm history: prior 4.02s, last-year revision as a feature, industry wave | 0.736 (−0.001) | 2× univariately, redundant with the financials |
| + thirty more disclosure flags (weakness types, tenure, CAMs, corrections in the notes, investigations, delisting notices, shell history…) | 0.738 (+0.001 [−0.001, +0.002]) | individually real, collectively redundant with the ten |
| ModernBERT-base fine-tuned on the MD&A opening, corpus text only | validation 0.607; test alone 0.536; fusion with the forest −0.004 | the first 512 tokens of Item 7 are the business overview; a reader given the wrong pages learns nothing |
| ModernBERT-base on a regex-retrieved digest (the sentences the flags were built from), all 2014–16 training filings | validation **0.727**; test alone 0.656; fusion with the 242-feature forest +0.008 [−0.001, +0.017], concentrated in 2019 (+0.035) | the reader recovers the flags' information from the right pages, but does not transfer across the parser boundary (2020–23 ≈ 0); not tested against the flag-bearing stack |
| quarterly arithmetic from the 10-Qs: implied Q4 share, quarter volatility, YTD mismatch, missing quarters, asset swings | +0.001 [−0.001, +0.002] | quarter-end asset swings (0.624 alone) and revenue volatility (0.611) are real but the annual figures already carry them; 99% of firms file all three 10-Qs |
| year-over-year change: text drift per section, flag transitions, audit report lag | −0.000 [−0.002, +0.002] | newly-not-effective is 2.6× alone; the flags and filing lag already imply it |
| contradictions: upbeat tone while income fell, "effective controls" beside a weakness / a corrected error / a late filing / SEC letters, no going-concern language while distressed | −0.000 [−0.001, +0.001] | our own idea, and a clean zero; "says effective" dominates whatever follows it (3.3% vs 4.9% when a weakness is also mentioned), and the genuine conflicts are too rare to matter |
| the auditor's record: PCAOB inspection deficiency rate (latest report before filing), trend, inspection tier, Form AP auditor and partner changes, other participating firms, audit-report lag | +0.001 [−0.001, +0.002] | real alone (top-quartile deficiency auditors' clients restate 7.5% vs 4.4%; audit-report lag 0.701, the strongest single feature found) but redundant: inspection tier is firm size, report lag is filing lag |
| MD&A bag-of-words, first attempt | out-of-fold 0.90 | **a leak**: restated filings' text came from our parser, clean filings' from EDGAR-CORPUS; the model learned the parser |
| MD&A bag-of-words, corpus-sourced text only (TF-IDF, 200k terms, logistic) | 0.677 alone; stacked +0.002 [−0.001, +0.004] | nothing on top of the financials; its top terms are "restatement", "restated" — already carried by the flags |
| Bao et al.'s track, their data: our forest vs their RUSBoost | 0.717 vs 0.711–0.714, three attempts to pass all fail; a fifth, the 1:10 sampling ratio, scores 0.716 (pooled 0.718 vs 0.714, +0.004 [−0.007, +0.015]) | statement-only ceiling confirmed from the other side; the ratio gain that lifted our headline does not cross to their track, as the as-of test predicted |
| training labels truncated at the cutoff (what was known on 1 Jan 2019), and a yearly refit with as-of labels | 0.724 (−0.020 [−0.026, −0.015]) at the tuned forest; 0.725 (−0.013) at the default one | the honest deployable number is 0.724 and the learner tuning does not reach it |
| auditor fees, Item 14 of the 10-K plus 28,015 fetched proxies (audit, audit-related, tax, other; level, share, change, residual on assets; 82% coverage) | +0.003 [+0.001, +0.006] on the 82% of test rows with a fee table; +0.001 [−0.001, +0.002] on all rows | real where present, gone on the full set; and the proxy is filed months after the 10-K, so most of it is not public on the scoring date |
| hierarchical shrinkage on the forest (Agarwal et al. 2022), λ = 2 / 10 / 50 | −0.001 / −0.002 / −0.007 | leaf values were not the problem |
| forest re-tuned at 1:10 on the 330 stack: features per split (0.2 / 0.35 of columns / log2), leaf 2 / 12 / 25, 1,500 trees, fifteen seeds | −0.014 / −0.022 / +0.002; −0.002 / +0.001 / −0.001; +0.000; +0.000 | more features per split hurts sharply, fewer helps a little (log2 lifts both validation years, P 0.07); leaf 2 has the best validation and a worse test, a reminder not to follow validation blindly |
| rank-average of forests at 1:2, 1:4 and 1:10 | +0.002 [−0.000, +0.005] | diversity across the ratio, at the edge of noise |
| LightGBM with the same 1:10 negative subsample per round; its rank blend with the forest | 0.730 alone (−0.017); blend +0.003 [−0.002, +0.008] | boosting is worse here at every setting tried; the blend's gain is inside noise |
| label-quality weights (tight-label positives 1, other positives 0.5 / 0.25); training on the tight label alone | −0.002 / −0.007; −0.019 | the loose positives are signal, not noise |
| censoring-aware negatives under the as-of protocol: each recent "clean" training filing weighted by how mature its label was on the cutoff date (2018 filings: 3%), or the youngest negatives dropped | weights −0.001 / −0.003; dropping under-one-year negatives −0.045, under-two-years −0.081 | the young negatives are 95% correct and carry the freshest data; the label lag cannot be weighted away |
| multi-horizon ensemble: forests for restated within 1 / 2 / 3 years, rank-averaged | 1-year alone −0.020; ensemble −0.009 | the shorter horizons throw away most of the positives |
| firm lag block: each statement feature minus the firm's prior-year value (242 more columns) | −0.015 [−0.021, −0.009] | the raw items already carry both years; doubling the columns dilutes the split search |
| era-wise ranks: every continuous column replaced by its percentile within its filing year | 0.742 (-0.007 [-0.012, -0.003]) | the forest was not learning scale drift; ranking within year throws away cross-year level information |
| code audit of every feature builder for look-ahead (`leakfix_test.py`): market block without missing indicators, price-level columns dropped, industry-rate column dropped | 0.748 (−0.001 [−0.002, +0.001] vs 0.749); no market block at all 0.750 | two real channels, no signal in them; the market block is worth nothing |
| forest + LightGBM rank blend, boosting retuned on validation (31 leaves, lr 0.02, the forest's 1:10 subsample per round) | boosting alone 0.736 (−0.014); blend at 0.3 weight **0.752 (+0.003 [+0.000, +0.006], P 0.021)**; at 0.5 +0.001 | the best-looking null in the file: the interval touches zero, so it is not claimed; a second model of a different kind is worth about a quarter point here |
| material-weakness determinants bundle (Doyle, Ge & McVay 2007): firm age from the first EDGAR filing, business-segment and geography counts, restructuring charges, deferred tax expense, book-tax difference, valuation-allowance change | +0.001 [+0.000, +0.003], P 0.024 | firm age is real alone (young firms restate more; 0.575) and segments 0.56, but the statements already carry size and history; three of the planned columns cannot be built because the SEC's data sets now hold primary-statement facts only, not the notes |
| prior-year lookup on leap days and 52/53-week fiscal years (audit finding) | not rebuilt | affects 334 filings (0.7%): the data sets round period dates to month ends, which sidesteps most of it |
| **positives:negatives per tree 1:4 instead of 1:1** (Perols et al. 2017's undersampling ratio) | **0.746 (+0.009 [+0.005, +0.014])**; 1:2 +0.007, 1:10 +0.008; validation rises with the ratio | each tree sees more of the negatives; the first modelling change since the balanced forest itself to move the test set |
| the same flags cut by EDGAR-CRAWLER for every row instead of our extractor (1:1 forest) | 0.736 vs 0.737 | the two extractors agree; the number is the number |
| forecasting instead of detecting: the next 10-K's label, the one after, and a *new* restatement among currently clean filers | 0.739 / 0.765 / **0.698** | the first two are the detection label wearing a different date; the last is the real forecast number |
| Bertomeu et al. 2021's published paid-data scores on the same 4,399 test filings, same label | ours 0.770 vs theirs 0.629 (+0.141 [+0.096, +0.186]); on their own label definition 0.698 vs 0.649, a tie (P 0.087) | a blend never beats ours on our label; their model is trained 2001–2010, so part of the gap is age |
| little-r label from comparative mismatches, added to training | 0.605–0.709 on Item 4.02 | real label (7× enrichment), wrong training target |
| little-r model's score stacked as one column | 0.727 (−0.000) | a second view of the same 242 columns is not a second view |
| board/officer interlocks from Forms 3/4/5 (2.1M links, 139k people) | 0.728 (+0.000) | linked-to-a-restater rate 4.52% vs 4.43%: no contagion on this label |
| firm-level score persistence (decayed average with the firm's earlier 10-Ks) | 0.728 (+0.001) | restating is a firm trait, but the features already know the firm |
| event deduplication (each Item 4.02 counts once, not once per 10-K it labels) | 0.730 (+0.002, n.s.) | leans right, not distinguishable from zero |
| proximity weights (filings nearer the announcement weigh more) | 0.727 (−0.001) | |
| positive-unlabeled: down-weight suspicious negatives | 0.722 (−0.006) | *hurts*; the negatives the model doubts are the ones it needs |
| cascade: second forest on the top 30% | 0.729 (+0.001, n.s.) | |
| per-industry isotonic calibration | 0.722 (−0.006) | hurts; industry interleaving was already right |
| training-window ensemble (2014–18, 2015–18, 2016–18) | 0.727 (−0.001) | |

Three of these deserve a paragraph.

**Label noise was not the ceiling.** The loose label calls a clean 2018 10-K
"restated" if the company withdrew its 2020 statements. That looked like the
obvious limit on AUC, so we parsed every Item 4.02 to find which fiscal years
it actually names (99% parse rate) and rebuilt the label. Tightening halved the
positives and lowered AUC in every cell of the 2×2:

| trained on / scored on | loose | tight |
|---|---|---|
| loose | 0.691 | 0.658 |
| tight | 0.632 | 0.641 |

The "noisy" positives were carrying real signal. Restating is a property of
the firm more than of the year; the loose label captures the firm. Predicting
*which* year was misstated is a harder problem (0.641) than predicting that a
firm will restate something (0.691).

**Text did not add to the financials.** On a balanced pilot of 1,599 filings,
Loughran–McDonald-style tone and readability features reach 0.61 alone but add
0.004 on top of the financial features, indistinguishable from zero. The
things text measures — length, complexity, hedging — are mostly proxies for
company size and industry, which the balance sheet already states directly.

**Events add little once filing behaviour is in.** An auditor change in the
prior three years more than doubles the restatement rate (8.3% vs 3.4%) and
was worth +0.006 on top of the 79-feature model. On top of the full financial
block it is worth +0.002: filing lag and filer class already carry most of what
"the company is having trouble closing its books" says.

## What a forecaster would have known

The headline trains once on filings through 2018 and scores 2019–2023 with
that one model. Its training labels, though, are taken with hindsight: a
10-K filed in 2018 is positive if an Item 4.02 followed within three years,
including one filed in 2020. Of the 667 training positives, 161 were
announced after the cutoff, and 2018 is where it bites — 32 of its 127
positives were known on 1 January 2019. `asof_test.py` reruns the stack under
three protocols:

| protocol (tuned forest, audited 290 stack) | 2019 | 2020 | 2021 | 2022 | 2023 | pooled | vs headline (paired) |
|---|---|---|---|---|---|---|---|
| one fixed model, hindsight labels (the headline protocol) | 0.794 | 0.736 | 0.731 | 0.740 | 0.730 | 0.745 | |
| one fixed model, labels as known on 1 Jan 2019 | 0.706 | 0.712 | 0.722 | 0.736 | 0.730 | **0.724** | −0.020 [−0.026, −0.015] |
| one year ahead: refit each January on all earlier filings, labels as known that day | | | | | | 0.732 | −0.013 [−0.019, −0.007] |
| one year ahead, refit with hindsight labels (not obtainable) | | | | | | 0.801 | +0.056 [+0.049, +0.063] |

Three things follow. The hindsight in the headline is worth two points at
the tuned forest (one point at the default one), concentrated in the first
test year, so 0.724 is the number a model trained on 1 January 2019 would
have posted. Refitting every year with honest labels
buys nothing: the fresh filings arrive with most of their positives still
unknown, so the newest training year is always the emptiest. And the fourth
row, which a forecaster cannot have, shows what the recent years are worth
once their labels mature — 0.80 — so the model is limited by label lag, not
by data volume, and a paper that refits yearly on final labels can post a
number six points above anything a forecaster could have had. Bao et al. handle the same problem with a two-year gap between
training and test; that gap is built into the "as of" rows here through the
truncation itself.

## Forecasting, not detecting

Parker, Jiang, Cho & Vasarhelyi (2025, *The Accounting Review*) forecast
material misstatements one and two years ahead from the current year's data,
with Compustat, Audit Analytics and CRSP. Our headline label is detection:
whether *this* 10-K is later restated. `forecast_test.py` relabels each
filing with the loose label of the same company's next annual filing and of
the one after that, and trains and tests the clean stack on those labels with
the usual split (run before the features-per-split tuning and the market
audit, so its detection row reads 0.745 rather than 0.749; the rows are
comparable with one another):

| target | test filings (restated) | test AUC | 95% CI |
|---|---|---|---|
| detection: this 10-K restated (the headline label) | 17,755 (927) | 0.745 | 0.729–0.760 |
| forecast: the next 10-K restated | 12,325 (608) | 0.739 | 0.721–0.759 |
| forecast: the 10-K after next | 8,144 (389) | 0.765 | 0.742–0.787 |
| forecast: the next 10-K restated, statements only | 12,325 (608) | 0.732 | 0.712–0.749 |
| forecast a *new* restatement: next 10-K restated, this one not | 11,757 (188) | **0.698** | 0.665–0.730 |

The first three forecast rows are nearly the detection number because a
three-year label window makes consecutive filings share their events: if a
2020 announcement condemns the 2019 10-K it usually condemns the 2018 one
too. The honest forecast is the last row — among companies whose current
10-K is *not* later restated, which will be the first to fall — and there
the stack scores 0.698 on 188 events. That is the row to set beside a
published one-year-ahead number when one becomes readable; their paper's
table is behind a paywall and its AUC is not in any public summary.

## What did not go wrong, because we checked

- **Test-set leakage through selection.** Every selection decision used the
  2017–18 validation slice. The tuner that found +0.03 on validation and 0.00
  on test is the reason we trust the rest.
- **Leakage through a metadata column.** `sub.txt` carries `prevrpt`, which
  means "this submission was *subsequently* amended" — knowledge from after
  the filing date, and nearly the label itself (27.6% restatement rate when
  set vs 3.7%). With it the filing block scored 0.746; it was caught before it
  became a headline and is excluded. Every remaining filing feature is knowable
  on the day of filing.
- **Leakage through a text source.** MD&A text came from two parsers — ours
  for a sample that included every restated filing, EDGAR-CORPUS for the rest.
  A bag-of-words model scored 0.90 out of fold by learning which parser had
  produced the text. Caught because 0.90 is impossible here; fixed by giving
  every filing the corpus text where it exists, so both classes share a source.
  The first BERT pilot ran on the mixed text and its 0.787 validation is
  discarded for the same reason.
- **Leakage through which test rows have data.** After the 10-K text flags
  were filled for the fetch sample only, a "full test set" run printed 0.776:
  unflagged 2021–23 rows (almost all clean) were imputed as "controls
  effective", so the imputer, not the model, separated the classes. Every
  number reported on the full test set uses flags fetched for every test
  filing with one parser; subset numbers are labelled as subsets.
- **Moving labels.** The Item 4.02 corpus was growing while experiments ran,
  changing the base rate mid-comparison. The label file is pinned.
- **The 2021 SPAC wave.** 65% of 2021's Item 4.02 filings are SIC 6770
  blank-check companies restating warrant accounting. The financial-firm
  exclusion removes them; results are also reported with 2021 dropped.
- **A silent parser bug.** Since inline XBRL became mandatory for 8-Ks, filing
  directories contain rendered XBRL pages (`R1.htm`, ...) that can be larger
  than the 8-K itself. Picking "the biggest .htm" chose those for half of all
  filings from 2021 on, and the first tight-label result (0.605) was an
  artifact of parsing cover pages. The section-found rate by year exposed it;
  the fixed chooser prefers files named for the form and never takes an XBRL
  artifact.
- **Validation optimism.** The extended feature block gained +0.028 on
  validation and +0.005 on test. Validation numbers are reported alongside
  test throughout so this is visible rather than hidden.
- **Small-count metrics.** Precision at 100 moves by several points between
  seeds and is reported with that understanding; AUC over 927 positives is
  the number the claims rest on.

- **Leakage through a release date.** EDGAR dates an SEC comment letter by
  the day it was written, but the SEC releases correspondence weeks after the
  review closes. Six letters checked against EDGAR's daily indices, which
  list filings on the day they were disseminated, all became public 31 days
  after their date. So a letter sent in the month before a 10-K had been
  counted as known on the filing date when it was not. Fix: every daily
  index from 2013 to 2024 was fetched (`fetch_daily_index.py`), giving a
  public date for 207,928 letters and replies; letters without one are moved
  31 days later; the block was rebuilt and the stack rerun. Cost: the letters
  step falls from +0.002 [+0.001, +0.004] to +0.001 [−0.001, +0.003] and is
  no longer claimed; the headline moves from 0.749 to 0.748.
- **Leakage through a price feed.** Two channels, found by a line-by-line
  audit of every feature builder after the year-key leak. Yahoo drops
  delisted tickers, so "no prices" for an old filing means "delisted by the
  download date" — and the imputer's missing indicators handed that future
  fact to the forest, twelve columns of it; dropping the coverage column had
  never closed it. And Yahoo's closes are adjusted for splits after the
  window, so a reverse split after a 10-K rewrote that filing's price level,
  market cap, book-to-market and turnover. Fix: the market block is imputed
  without indicators and its four level columns are dropped; the
  within-window returns, volatility and drawdown stay. Cost to the headline:
  −0.001 [−0.002, +0.001] (`leakfix_test.py`). Without the market block at
  all the stack scores 0.750, so the block is worth nothing either way; it is
  kept only because the separate-line table needs it. The same audit found
  two look-ahead channels in blocks that had already been tested and
  rejected — the quarterly block read amended 10-Qs and later years'
  comparatives with no filing-date guard, and the Form AP columns came from
  documents typically filed after the 10-K — and a whole-panel denominator
  in the industry restatement-rate column, now dropped. It also found a bug
  in the evaluation script itself: the last stage of the headline table
  dropped the thirty flags when it added the change block, so the change
  block's paired test had been change-minus-flags. Fixed; the change block
  is still null.
- **Three things a feature audit cannot see, checked on the saved test scores
  (`leak_audit2.py`).** *The label:* 25 test positives had their Item 4.02
  filed the same day as the 10-K and 10 more within a month; the model does
  not read the announcement off the filing (their mean score is no higher
  than other positives'), and removing every positive announced within 90
  days leaves 0.744. *Firm memory across the split:* 914 test filings come
  from companies with a positive training filing, and they restate at 19.4%
  against 4.5%; the model scores 0.740 on the other 16,841 filings alone
  (firm-disjoint), 0.743 with the 99 positives that share an announcement
  with a training row removed, and 0.727 with every positive of a
  previously-positive firm removed. Recurrence is real history, not
  memorisation, but a reader who wants the number for firms the model has
  never seen restate has it: 0.740. *Label maturity:* the announcement panel
  runs to August 2026, so every test filing, including 2023's, had its full
  three-year window; on 2019–2020 alone the stack scores 0.764.
- **Leakage through a year key.** The 10-K text flags for 2014–2020 were
  first taken from EDGAR-CORPUS, whose rows are keyed by *fiscal* year. Our
  loader matched them by *filing* year, so a December year-end company's
  10-K for 2015 (filed in 2016) received the text of its 10-K for 2016 (filed
  in 2017). A restatement announced in between appears in that later Item 9A
  as "not effective" and "material weakness", so the flags were partly
  reading the answer. The tell was the rebuild: one extractor over every
  filing, matched by accession number, scored 0.737 where the corpus-fed
  version had scored 0.751, and the corpus's "not effective" flag alone
  scored 0.682 against the rebuild's 0.605 on the same filings. Matched by
  fiscal year the two extractors give identical flags (0.627 and 0.627); the
  corpus was never better, only a year ahead. `corpus_vs_crawler.py`. Every
  number that had touched the corpus — the 0.751 headline, the 2019 test year's
  0.790, the paid-data head-to-head at 0.793, the as-of line at 0.742, the
  thirty-flag, change and digest blocks, and the claim that our own parser
  found one in five fewer "not effective" conclusions — was withdrawn and
  rebuilt from documents matched by accession number. The corrected numbers
  are the ones in this file.

## Reproduce

The repository holds the code and this write-up. `data/` (about 38 GB of SEC
filings, extractions and feature caches) is rebuilt by the commands below,
and `tools/edgar-crawler` is a clone of github.com/nlpaueb/edgar-crawler
(GPL-3), used by `ec_extract.py`.

```
set SEC_CONTACT=Your Name you@example.com

python fetch_sec.py                # quarterly zips -> data/raw/ (about 4 GB)
python fetch_restatements.py 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
python tune.py --trials 1          # builds and caches the 28-feature matrix
python raw_items.py                # 17 raw items x3         -> features_raw.npz
python raw_items2.py               # 44 more, Benford, filing -> features_raw2.npz
python edge.py --raw data/out/features_raw.npz   # learner bake-off, validation-selected
python final.py                    # 79-feature paired bootstrap, per-year, drop-2021
python fin_test.py                 # staged financial-only test to 0.727

# the separate line: free outside data
python fetch_daily_index.py 2013 2024 8   # the day each comment letter became public
python fetch_events.py 4.01 2011 2023
python fetch_events.py 5.02 2011 2023
python event_features.py
python events_test.py
python fetch_prices.py             # Yahoo daily prices via the SEC CIK-ticker map
python market_features.py
python insider_features.py         # after downloading the SEC Forms 3/4/5 data sets to data/raw/insider/
python record_test.py              # staged: statements -> events -> insider -> market, two ways

# Bao et al.'s track (their CSV and MATLAB from github.com/JarFraud/FraudDetection in data/raw/bao/)
python race.py                     # reproduction and learner comparison
python race2.py; python race3.py; python race4.py   # the three attempts to pass them

# the letters and flags blocks behind the 0.737 headline (one parser, every row)
python letters_features.py; python letters_test.py
python fetch_all_10k.py            # every 10-K in the panel, saved whole (data/raw/mdna/)
python scrutiny_v3_clean.py        # Item 9A/8/3 flags, every row cut from its own saved 10-K
python ec_extract.py --procs 14; python ec_features.py   # EDGAR-CRAWLER sections -> flags2 / change blocks
SCRUTINY_NPZ=data/out/features_scrutiny_v3c.npz FLAGS2_NPZ=data/out/features_flags2_ec.npz CHANGE_NPZ=data/out/features_change_ec.npz python headline_test.py

# head-to-head with the paid-data model (their oos_2011_2019.csv in data/raw/bertomeu/)
python bertomeu_compare.py
python asof_test.py               # labels as known at the cutoff; one-year-ahead refits
python backtest.py; python backtest/portfolio.py --floor 10   # forward returns and the long-short test (Ken French factors in data/raw/factors/)
python fetch_proxy.py; python fees_features.py parse10k; python fees_features.py parseproxy; python fees_features.py build
python block_test.py data/out/features_fees.npz --all-rows   # the auditor-fee block
python forecast_test.py           # next year's and the year after's label; new restatements only
python simple_wins.py; python ratio_more.py   # sampling ratio, shrinkage, first-time filers
python algo_wins.py; python mf_sweep.py; python innovate.py   # learner experiments (ledger)
python leakfix_test.py            # the audited market block, firm-clustered intervals
BRF_STRATEGY=0.1 python race.py   # Bao's track with the chosen ratio (attempt five)
```

Other scripts: `train.py` / `head.py` / `baserate.py` (the F-score comparison,
precision@k with bootstrap, per-year base rates), `history.py` + `train2.py`
(history and industry features), `text_features.py` + `text_increment.py`
(streaming 10-K text), `fetch_periods.py` + `relabel.py` (the tight-label
experiment), `edge_tabpfn.py` (TabPFN v2, chunked for a 6 GB GPU).

Python 3.13, scikit-learn, LightGBM, CatBoost, imbalanced-learn, TabPFN 8.5
(v2 weights, which need no account). Feature building over 40 quarters takes
about 20 minutes per block; each test script 10–15 minutes; the TabPFN run
about 45 minutes on the 6 GB card.

## Limits

- **Restatements, not fraud.** Most Item 4.02 filings are errors.
- **The headline's training labels look ahead.** Announcements after the
  cutoff label training filings. Trained on what was known on 1 January
  2019 the 290-column stack scores 0.724, not 0.745; both are reported.
- **Big R only.** Silent revisions ("little r") that never trigger an 8-K are
  labelled negative. A comparative-mismatch label built from prior-year
  figures in later filings could recover some of them; not yet built.
- **US, non-financial, XBRL-era.** Nothing here transfers to banks, insurers,
  foreign filers, or pre-2009 filings.
- **Free market data has a survivorship hole.** Yahoo drops delisted tickers;
  56% of filings have prices, and the rest restate more. The block is
  imputed without missing indicators so the model cannot read "delisted
  later" off the gap, and its price-level columns are dropped because Yahoo
  adjusts them for later splits. What remains is worth nothing on top of the
  statements (0.750 without the block, 0.748 with). CRSP would close the
  hole; whether it would add signal is another question.
- **The comparison with the literature is one label away from clean.** Our
  learner ties Bao et al. on their AAER data; our features have not been run
  on an AAER label because the free AAER panel ends in 2014 and XBRL begins in
  2009, leaving too few positives to say anything.
- **The label depends on detection.** A misstatement only becomes a
  restatement if an auditor, short seller, regulator or new CFO finds it. That
  is a hard ceiling on any model built from the statements alone, and it is
  why nobody in the literature is far above 0.72.

## References

- Dechow, Ge, Larson & Sloan (2011). Predicting Material Accounting
  Misstatements. *Contemporary Accounting Research* 28(1).
- Bao, Ke, Li, Yu & Zhang (2020). Detecting Accounting Fraud in Publicly
  Traded U.S. Firms Using a Machine Learning Approach. *Journal of Accounting
  Research* 58(1). Code: github.com/JarFraud/FraudDetection.
- Bertomeu, Cheynel, Floyd & Pan (2021). Using machine learning to detect
  misstatements. *Review of Accounting Studies* 26. Out-of-sample scores and
  code published on the authors' Restatement Risk Measure page.
- Amiram, Bozanic & Rouen (2015). Financial statement errors: evidence from
  the distributional properties of financial statement numbers. *Review of
  Accounting Studies* 20.
- Beneish (1999). The Detection of Earnings Manipulation. *Financial Analysts
  Journal* 55(5).
- Hollmann et al. (2025). Accurate predictions on small data with a tabular
  foundation model. *Nature* 637.
- SEC Financial Statement Data Sets:
  sec.gov/dera/data/financial-statement-data-sets
