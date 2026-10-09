# EPL Over/Under 2.5 — Closure Report

**Closure date:** 2026-10-09  
**Status:** CLOSED  
**Conclusion:** Profitability was not demonstrated in the reported tests.

> Evidence status: this document preserves results reported during the research conversation. The raw CSVs and scripts were not available to this GitHub write operation for direct re-execution. Numbers and test statuses below are therefore reported results, not independently verified by this commit.

## 1. Main controlled test

| Metric | Reported value |
|---|---:|
| Test season | 2023/24 |
| Test-season matches | 380 |
| Bets placed | 252 |
| OVER bets | 17 |
| UNDER bets | 235 |
| Net profit | -50.29 units |
| ROI | -19.96% |
| IID bootstrap 95% ROI interval | [-34.65%, -5.05%] |
| Reported selected-model Brier | 0.244570 |
| Decision | Not supported |

The reported bootstrap interval lies entirely below zero. This supports a negative result under the stated IID bootstrap procedure, conditional on the correctness of the data and calculation. It does not fully account for temporal dependence or model-selection uncertainty and does not prove that every future strategy will lose.

## 2. Reported temporal split

| Split | Seasons | Matches |
|---|---|---:|
| Train | 2015/16–2021/22 | 2,660 |
| Validation | 2022/23 | 380 |
| Final test | 2023/24 | 380 |

Five logistic-regression variants were reportedly compared, with selection based on validation Brier score. The selected variant was described as an 18-feature model using median imputation and balanced class weights. The bet file reportedly contains 252 selected bets, not all 380 matches.

## 3. Earlier test and diagnostics

- First reported blind test (2024/25): ROI -1.94%; the reported confidence interval included zero.
- First diagnostic: OVER selected for 302/325 matches (92.9%); reported accuracy 56.92%, compared with 57.23% for an always-OVER reference.
- Missing-feature diagnostic: 566 training rows and 55 test rows reportedly removed due to unavailable early-season form/goal-average features.
- Data audit: 3,800 matches, clean joins, zero score mismatches reportedly observed.

These are historical reported figures; the original artifacts were not re-run for this commit.

## 4. Reported test status

| Test / artifact | Reported status | Scope note |
|---|---|---|
| `test_ips.py` | 5/5 PASS | Simulation/IPS checks; not a profitability test |
| `test_invariant.py` | 18/18 PASS | Invariant checks; not a profitability test |
| `audit_ou25.py` | Clean joins; zero score mismatches reported | Original audit not re-run here |
| `epl_ou25_karsilastirma.py` v4 | Negative ROI reported | 2023/24 final test |

A PASS status for simulation or invariant tests must not be interpreted as evidence of betting profitability.

## 5. Limits and unresolved hypotheses

- The definition of `BaseOver` and `BaseUnder` in `ou25_karsilastirma.csv` was not verified. No baseline-Brier comparison is asserted.
- The reason for the 17 OVER / 235 UNDER bet split is unknown. A causal effect from `class_weight="balanced"` was not established.
- Previously reported calibration bins included N=1, N=125, and N=221. These partial bins do not prove a general systematic error or explain its cause.
- The IID bootstrap interval does not fully capture temporal dependence or model-selection uncertainty.
- The results do not prove that the entire EPL market is perfectly efficient or that all future strategies must lose.

## 6. Final decision

1. Do not treat this strategy as profitable.
2. Preserve the reported results without post-hoc changes.
3. Do not try new thresholds or models against the same 2023/24 final-test season.
4. Reopen only for a genuinely new, pre-specified, independently testable hypothesis and a fresh evaluation protocol.

**Final status: CLOSED.**
