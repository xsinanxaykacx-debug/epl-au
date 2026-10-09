# H9 Date/Match-Key Audit and Reproducibility Report

**Date:** 2026-10-09  
**GitHub Actions run:** https://github.com/xsinanxaykacx-debug/epl-au/actions/runs/37908019114  
**Report artifact:** https://github.com/xsinanxaykacx-debug/epl-au/actions/runs/37908019114/artifacts/11605955566

## Scope and safety

- Historical input allow-list: `E0.csv` through `E0 (9).csv` only.
- Blind file `E0 (10).csv`: **not read**.
- No source rows were deleted, renamed, or auto-joined.
- H9 Explore protocol settings and thresholds were not tuned.

## Data integrity results

| Check | Result |
|---|---:|
| Historical source rows | 3,800 |
| H1 feature rows | 3,800 |
| Unique robust source keys | 3,800 |
| Unique robust feature keys | 3,800 |
| Robust key intersection | 3,800 |
| Source-only unmatched rows | 0 |
| Feature-only unmatched rows | 0 |
| Invalid dates after robust parsing | 0 |
| Duplicate robust keys | 0 |
| H9 market merge | 3,800 |
| Rows after pre-registered AH-missing filter | 3,798 |
| H9 train / validation / OOS dataset rows after filter | 3,039 / 379 / 380 |
| Reads of `E0 (10).csv` | 0 |

The legacy `pd.to_datetime(..., dayfirst=True, errors="coerce")` parser returned `NaT` for all **380 dates in season 2016/17** in both the source and feature tables. The mixed-format parser recovered all 380 dates. The corrected H9 key builder also normalizes team labels conservatively (Unicode normalization, whitespace, case) and rejects invalid dates rather than merging them under a placeholder key.

The legacy-key diagnostic reported an intersection of 3,420, while the robust-key audit matched all 3,800. Use the robust-key results for the corrected pipeline; do not interpret the legacy key count as missing source matches.

## H1 feature rebuild comparison

The corrected `build_h1_features_v2.py` rebuilt all 3,800 rows and 56 columns from the 10 historical CSVs. A key-aligned comparison against the committed `h1_features_v2.csv` found:

- Common match keys: 3,800
- Before-only / after-only keys: 0 / 0
- Changed feature columns: 0
- Changed cells: 0
- Maximum absolute numeric difference: 0

So the date-parser repair did not change any of the existing feature values.

## H9 inventory validation

**Decision: PASS → Explore.** All 3,800 matches merge one-to-one; duplicate key count is zero; B365 opening odds are complete and positive; FTR contains only H/D/A; the AH-missing filter removes the expected two rows; the blind-file read counter remains zero.

## Frozen H9 Explore rerun

The H9 script now sorts the merged dataset by parsed match date, home team and away team before model fitting. This prevents the physical row order of the CSV from affecting the fitted model's numerical result. Two consecutive corrected runs produced identical result workbooks: all sheets, row keys and cell values matched exactly.

| Split | Dataset rows after AH filter | Bets | Wins | ROI | 95% ROI CI | Permutation p | MDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 3,039 | 2,888 | 990 | +9.60% | +2.88% to +16.66% | 1.0000 | 85.34% |
| Validation | 379 | 346 | 117 | -5.60% | -20.89% to +13.55% | 0.9935 | 35.67% |
| OOS (2024/25) | 380 | 346 | 118 | -3.80% | -19.66% to +14.49% | 0.9307 | 42.27% |

**H9 decision: NOT SUPPORTED.** OOS ROI is negative, its confidence interval includes zero, permutation p-value is not significant, and MDD exceeds the 20% limit. No profitability claim is supported.

## Existing workbook caveat

The previously committed `h9_explore_v1_results.xlsx` does not exactly match the corrected canonical-order run in the **Train** sheet (old: 2,885 bets / 988 wins; corrected: 2,888 bets / 990 wins). Validation and OOS summary metrics are unchanged. Treat the old workbook as a legacy baseline; use the regenerated workbook in the linked Actions artifact for the corrected code version.

## Code changes

- `audit_match_keys_v1.py`: row-count mismatches no longer stop diagnostics; invalid-date rows cannot count as valid key matches.
- `build_h1_features_v2.py`: robust mixed-format date parsing, explicit historical file allow-list, fail-closed validation for invalid dates and unexpected source row counts.
- `h9_inventory.py`: robust dates, normalized team labels, rejects invalid date/team keys.
- `h9_explore_v1.py`: same robust key logic plus canonical match ordering for repeatable model fitting.
- `compare_h1_features_rebuild.py` and `compare_h9_results.py`: key-aligned read-only comparisons.
- `.github/workflows/audit-match-keys.yml`: rebuilds, audits, reruns the frozen H9 protocol twice, and uploads reports/results as an artifact.
