# Repository Date Parsing Audit — 2026-10-09

## Status

**CI: PASS** — [workflow run](https://github.com/xsinanxaykacx-debug/epl-au/actions/runs/37910032864)  
**Artifact:** [audit reports and regenerated result workbooks](https://github.com/xsinanxaykacx-debug/epl-au/actions/runs/37910032864/artifacts/11606191641)

## Data-backed finding

The audit reads only the explicit historical allow-list: `E0.csv` through `E0 (9).csv`. It does not open `E0 (10).csv`.

- Historical data: 3,800 rows across 10 seasons, 380 rows per season.
- When all seasons are concatenated and the legacy default date parser is applied once, **all 380 dates in 2016/17 become NaT**.
- Parsing the same 3,800-row series with `format="mixed"` leaves **0 invalid dates**, recovers all 380 rows, and introduces 0 newly invalid rows.
- The issue depends on parsing the combined multi-season series. Parsing each season independently can conceal it; the audit therefore tests the concatenated corpus.

## Changes applied

- `h9_explore_v1.py`: H9 drawdown chronology now uses mixed-format date parsing.
- `h12_favourite_longshot.py`: H12 drawdown chronology now uses mixed-format date parsing.
- `audit_ou25.py`: full-feature date-key normalization now uses mixed-format parsing; the base directory is derived from the script location rather than a hard-coded Windows path.
- `build_h1_features.py`: both combined-series date parsing sites now use mixed-format parsing.
- `h13_corner_market.py`: mixed-format parsing is applied before chronological form calculations.
- `h14_optimal_odds_band.py`: mixed-format parsing is applied before date ordering and drawdown analysis.
- `audit_date_parsing_v1.py` and the GitHub Actions workflow now perform the combined-corpus audit and preserve before/after and repeat-run comparisons.

## Validation results

| Check | Result |
|---|---|
| H1 v2 rebuild | 3,800 rows × 56 columns; key-aligned comparison: 0 changed feature cells |
| Legacy H1 builder after parser fix | 3,800 rows × 56 columns; 3,800 common keys; 0 changed cells |
| Robust match-key audit | 3,800/3,800 keys matched; no unmatched rows or duplicate keys |
| H9 corrected run repeated | All corrected result sheets identical across runs |
| H12 corrected workbook vs previous workbook | All sheets identical |
| H12 corrected run repeated | All sheets identical |
| H13 corrected run repeated | All sheets identical |
| H14 corrected workbook vs previous workbook | All sheets identical |
| H14 corrected run repeated | All sheets identical |
| Blind file | `E0 (10).csv` reads: 0 |

### H13 baseline difference

The corrected H13 result differs from the committed baseline in 9 cells on the `Ozet` sheet (the `N`, `Win`, `Win_pct`, `Net_PL`, `ROI`, confidence interval, permutation p-value and MDD fields). The corrected run is repeatable. The 2024/25 OOS result remains **NOT SUPPORTED**: 375 bets, ROI −1.71%, 95% interval −11.33% to +7.41%, permutation p=0.7219, MDD 17.7%. H13 uses an approximate 1.90 odds assumption because actual corner-market odds are absent; this is not evidence of real betting ROI.

### H14 decision

The corrected H14 run is identical to the prior workbook and repeatable. The selected OOS band has only 12 bets, ROI −34.25%, 95% interval −100.00% to +9.58%, permutation p=0.3318 and BH-FDR q=0.9424. The frozen decision remains **NOT SUPPORTED**.

### H9 decision unchanged

Corrected H9 OOS summary remains: 346 bets, 118 wins, ROI −3.80%, 95% interval −19.66% to +14.49%, permutation p=0.9307 and MDD 42.27%. The value-betting signal remains **NOT SUPPORTED**; no profitability claim is warranted.

## Remaining static-scan matches, manually triaged

The static scan still lists five scripts that mention the affected source and use default date inference:

- `audit_match_keys_v1.py`: intentionally computes a legacy-parser comparison as a diagnostic.
- `date_sorunu_dogrula.py`: parses one season at a time; it does not reproduce the cross-season concatenation condition.
- `h10_opening_closing_roi.py` and `h11_explore_v1.py`: the affected early seasons are explicitly excluded from their actual input scope.
- `inventory_h1.py`: the date-order check receives one season at a time.

These are static review matches, not confirmed active defects in the corrected workflows. The audit remains conservative and reports them rather than silently suppressing them.

## Scope and interpretation

This work fixes date parsing and verifies reproducibility; it does not change betting thresholds, feature definitions, model selection, or acceptance criteria. H9, H12, H13 and H14 do not meet their frozen support criteria. No blind OOS file was opened.
