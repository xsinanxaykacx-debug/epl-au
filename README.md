# EPL Betting Market Research

A disciplined, pre-registered research program investigating whether reproducible economic betting edges exist in the English Premier League 1X2 market.

**Status:** 🔒 Research program closed
**Evidence base:** 2015/16–2024/25
**Blind data:** 2026/27 — locked and never used in completed research
**Conclusion:** No statistically reliable, sustainable OOS betting edge identified

---

## Executive Summary

This repository contains a multi-stage research program designed to test whether publicly available EPL pre-match market information can be transformed into a repeatable economic betting edge.

The project deliberately separates structural market relationships from profitability, discovery from confirmation, in-sample performance from out-of-sample performance, statistical significance from economic significance, and exploratory ideas from frozen hypothesis tests.

Across 20 completed research paths, no strategy satisfied the predefined requirements for a supported, statistically reliable and economically meaningful out-of-sample edge.

**This is not a claim that the EPL 1X2 market is mathematically efficient.**

Within the tested 2015/16–2024/25 EPL 1X2 data universe and the frozen hypotheses documented here, no sustainable and statistically reliable betting edge was demonstrated under blind OOS validation.

---

## What Was Tested?

| Research area | Result |
|---|---|
| Price-movement structure | STRONG OOS relationship |
| ROI / Edge signals | NEGATIVE |
| Feature engineering | PASS |
| Market-movement + Elo | INCONCLUSIVE / WEAK |
| Goal-difference exploratory models | WEAK / UNCERTAIN |
| Opening → closing ROI | NOT SUPPORTED |
| Draw movement | NOT SUPPORTED |
| Favourite–longshot bias | NOT SUPPORTED |
| Corner-form signal | NOT SUPPORTED |
| Optimal odds-band discovery | NOT SUPPORTED |
| Cross-bookmaker mispricing | NOT SUPPORTED |
| Pinnacle sharp vs retail lag | NOT SUPPORTED |
| ML/value betting — MLR | NOT SUPPORTED / RED FLAG |
| Context-adjusted ML — LightGBM | NOT SUPPORTED |
| xG expansion | REJECTED as a research extension |

### The central finding

Some relationships are real. The important finding is that **statistical predictability did not translate into a robust economic edge after blind OOS validation.**

---

## The Strongest Structural Result

### Chain 1 — Price Movement Structure

The first major research chain tested whether opening-to-closing market movements contained reproducible relationships with Asian Handicap and Over/Under market variables.

Blind OOS result:

- **10 / 12 primary tests successful**
- **No red flags**
- **STRONG OOS**

This established an important distinction:

> Market prices contain measurable structure. That does not automatically imply that the structure can be converted into profitable betting.

The subsequent ROI research failed to demonstrate a reliable conversion from this structure into economic edge.

---

## The Economic Tests

### H9 — Machine Learning Value Betting

Multinomial Logistic Regression estimated H/D/A probabilities from 49 pre-match contextual features.

Result:

- Train ROI: +9.48%
- Validation ROI: −5.60%
- OOS ROI: −3.80%
- OOS statistical support: failed
- **NOT SUPPORTED / RED FLAG**

### H17 — LightGBM Context Residual

A later frozen experiment compared LightGBM model probability with B365 opening implied probability and placed at most one bet per match when the residual exceeded the frozen threshold.

| Split | N | ROI | BCa lower | Permutation p | MDD |
|---|---|---|---|---|---|
| Train | 3,031 | +185.92% | +179.83% | 0.0001 | 4.00% |
| Validation | 366 | −6.71% | −19.48% | 0.3426 | 31.08% |
| OOS | 365 | +3.84% | −9.05% | 0.5847 | 31.97% |

**Interpretation:**

The enormous training result is not treated as proof of a real edge.

> Strong in-sample optimism / strong overfit signal; economic generalization failed.

The OOS result was positive in raw ROI, but the confidence interval crossed zero, the permutation test was non-significant, and maximum drawdown exceeded the frozen 20% limit.

Therefore: **H17 = NOT SUPPORTED.**

---

## Methodological Discipline

Every formal hypothesis followed the same core philosophy:

### 1. Pre-declared protocol

The hypothesis, feature definition, thresholds, sample split and decision criteria were frozen before the formal run.

### 2. Strict temporal separation

- Train: 2015/16–2022/23
- Validation: 2023/24
- Blind OOS: 2024/25

The blind OOS period was not used to tune the strategy.

### 3. No post-hoc rescue

A negative result was not converted into a positive result by changing thresholds, changing model parameters, selecting a more convenient odds band, changing the sample, or repeatedly rerunning until a positive result appeared.

A genuinely new idea requires a new hypothesis and a new frozen protocol.

### 4. Multiple-testing control

Where multiple related tests were performed, Benjamini–Hochberg FDR was used.

### 5. Economic validation

A strategy was not considered successful merely because it had positive ROI, a positive coefficient, predictive power, or a low raw p-value.

The decision framework also considered confidence interval, sample size, permutation significance, maximum drawdown, and multiple-testing correction.

---

## Data Integrity

The core research dataset covers 2015/16 → 2024/25, approximately 3,800 EPL matches.

The feature-engineering audit established:

- 3,800 / 3,800 rows
- deterministic processing
- no same-day leakage
- no self-reference
- no look-ahead
- 2026/27 never read by the completed H17 pipeline

The project uses only information that is available within the defined pre-match information boundary for each experiment.

---

## 2026/27 Blind Data

🔒 **E0 (10).csv**

The 2026/27 dataset is intentionally locked.

It is not used to discover hypotheses, tune thresholds, select features, validate strategies, or explain historical results.

This separation is intentional. The blind season remains protected so that any future protocol can be evaluated without contaminating the existing research record.

---

## Why xG Was Not Added

xG was considered as a possible data-expansion direction.

It was rejected for this research program because the project did not want to turn the study into an open-ended feature-engineering exercise where increasingly sophisticated representations of historical team performance are repeatedly searched for an edge.

> Do not extend the closed EPL 1X2 program simply by adding another commonly available performance descriptor.

This is a methodological stopping decision, not a claim that xG has no predictive value.

---

## Research Architecture

The repository separates market-structure analysis, predictive modeling, and economic validation. Each research path should retain its own frozen protocol, artifacts, test evidence, and conclusion. A structural or software test passing does not by itself establish a profitable betting edge.


### Completed sub-study: EPL Over/Under 2.5 (closed 2026-10-09)

This is a **separate market and experiment** from the repository's main 1X2 research. Its results must not be merged with or treated as evidence for the 1X2 experiments above.

**Status: CLOSED — profitability was not demonstrated.**

| Item | Reported result |
|---|---:|
| Historical coverage | 2015/16–2024/25 (10 seasons; approximately 3,800 matches) |
| First reported blind test | 2024/25 ROI: -1.94%; reported confidence interval included zero |
| Controlled comparison final test | 2023/24 |
| Final-test evaluated bets | 252 |
| OVER / UNDER bets | 17 / 235 |
| Net profit | -50.29 units |
| ROI | -19.96% |
| IID bootstrap 95% ROI interval | [-34.65%, -5.05%] |
| Decision | Not supported; closed |

The 2023/24 comparison reportedly trained on 2015/16–2021/22 (2,660 matches), selected among model variants using 2022/23 validation (380 matches), and evaluated on 2023/24 (380 matches). The selected variant was described as an 18-feature logistic-regression model with median imputation and balanced class weights. The 252 bets are a subset of the 380-match test season.

The reported negative ROI interval is conditional on the correctness of the underlying calculations and the IID bootstrap assumption. It does not account fully for temporal dependence or all model-selection uncertainty. It is evidence of negative performance in this test, **not proof that every future strategy must lose or that the whole market is perfectly efficient**.

#### Data, diagnostics, and reported test statuses

- A data audit was reported to cover 3,800 matches with clean match joins and zero score mismatches. The original audit output was not re-run as part of this repository update.
- The first diagnostic was reported to select OVER for 302 of 325 matches (92.9%); the always-OVER reference was reported to score 57.23% versus 56.92% for that model. These figures are preserved as previously reported, not independently revalidated here.
- Missing-feature analysis reportedly found 566 training rows and 55 test rows removed because pre-match form/goal-average features were unavailable early in seasons. This is a reported diagnostic and should be checked against its original output before reuse.
- `test_ips.py`: 5/5 PASS reported.
- `test_invariant.py`: 18/18 PASS reported.
- These simulation/invariant tests are **not** tests of betting profitability and do not independently validate the ROI calculations.
- The reported selected-model Brier score was 0.244570. The meaning of `BaseOver`/`BaseUnder` in the comparison CSV was not established, so no baseline-Brier claim is made.
- A few calibration bins were reported, including N=1, N=125, and N=221. These partial bin results do not establish a general causal explanation for the betting-side imbalance.
- The reason for the 17 OVER / 235 UNDER selection split, including any effect from `class_weight="balanced"`, remains an **unconfirmed hypothesis**.

#### Closure and reproducibility boundary

No new threshold or model was tested on the same 2023/24 final-test season after seeing these results. This sub-study is closed. Reopening it requires a genuinely new, pre-specified and independently testable hypothesis with a fresh evaluation protocol.

This record documents results reported during the research conversation. The raw `ou25_*.csv` outputs and source scripts were not available to this GitHub write operation for direct upload or re-execution. Therefore, the figures and test statuses above are explicitly labelled **reported**, not independently verified by this commit.
