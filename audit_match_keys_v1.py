# -*- coding: utf-8 -*-
"""
audit_match_keys_v1.py
Read-only audit of EPL match-key mismatches between historical source CSVs
and h1_features_v2.csv.

SAFETY:
- Reads only E0.csv through E0 (9).csv (2015/16–2024/25).
- Never globs E0*.csv.
- Never opens E0 (10).csv.
- Does not modify source files or the feature file.
- Writes diagnostic reports to audit_match_keys_v1_output/.

Purpose:
1. Compare legacy pandas date parsing with robust per-value date parsing.
2. Separate date parsing issues from team-name/key mismatches.
3. Report missing/duplicate keys and row-level unmatched records.
4. Fail closed if expected inputs are missing or source season counts differ.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(".")
FEATURE_FILE = ROOT / "h1_features_v2.csv"
OUT_DIR = ROOT / "audit_match_keys_v1_output"

# Deliberately explicit allow-list. Do not replace with glob("E0*.csv").
SOURCE_FILES = [
    ("E0.csv", "2015/16"),
    ("E0 (1).csv", "2016/17"),
    ("E0 (2).csv", "2017/18"),
    ("E0 (3).csv", "2018/19"),
    ("E0 (4).csv", "2019/20"),
    ("E0 (5).csv", "2020/21"),
    ("E0 (6).csv", "2021/22"),
    ("E0 (7).csv", "2022/23"),
    ("E0 (8).csv", "2023/24"),
    ("E0 (9).csv", "2024/25"),
]
FORBIDDEN_FILE = "E0 (10).csv"
EXPECTED_PER_SEASON = 380
EXPECTED_TOTAL = 3800
KEY_COLS = ["Season", "Date", "HomeTeam", "AwayTeam"]


def normalize_team(value: object) -> str:
    """Conservative key normalization; no alias mapping or fuzzy matching."""
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    value = re.sub(r"\s+", " ", value).strip()
    return value.casefold()


def parse_dates_robust(series: pd.Series) -> pd.Series:
    """
    Parse common EPL CSV date formats without relying on the first row's format.
    Recognized explicit formats:
      DD/MM/YYYY, DD/MM/YY, YYYY-MM-DD, YYYY-MM-DD HH:MM:SS.
    Remaining values get a pandas mixed-format fallback when supported.
    """
    raw = series.astype("string").str.strip()
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    formats = [
        "%d/%m/%Y",
        "%d/%m/%y",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in formats:
        mask = result.isna() & raw.notna()
        if mask.any():
            parsed = pd.to_datetime(raw.loc[mask], format=fmt, errors="coerce")
            result.loc[mask] = parsed
    mask = result.isna() & raw.notna()
    if mask.any():
        try:
            parsed = pd.to_datetime(
                raw.loc[mask], format="mixed", dayfirst=True, errors="coerce"
            )
        except (TypeError, ValueError):
            # Compatibility fallback for pandas versions without format="mixed".
            parsed = pd.to_datetime(
                raw.loc[mask], dayfirst=True, errors="coerce"
            )
        result.loc[mask] = parsed
    return result


def legacy_dates(series: pd.Series) -> pd.Series:
    """Reproduce the old key builder's default-format inference."""
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def load_sources() -> pd.DataFrame:
    frames = []
    print("SOURCE INVENTORY (explicit allow-list)")
    for filename, season in SOURCE_FILES:
        path = ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required historical file missing: {filename}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        missing = [c for c in ["Date", "HomeTeam", "AwayTeam"] if c not in frame.columns]
        if missing:
            raise ValueError(f"{filename}: missing required columns: {missing}")
        print(f"  {filename:14s} season={season} rows={len(frame)}")
        if len(frame) != EXPECTED_PER_SEASON:
            raise ValueError(
                f"{filename} ({season}) has {len(frame)} rows; "
                f"expected {EXPECTED_PER_SEASON}. No rows will be silently dropped."
            )
        frame = frame[["Date", "HomeTeam", "AwayTeam"]].copy()
        frame["Season"] = season
        frame["SourceFile"] = filename
        frame["SourceRow"] = range(2, len(frame) + 2)
        frames.append(frame)

    sources = pd.concat(frames, ignore_index=True)
    if len(sources) != EXPECTED_TOTAL:
        raise ValueError(f"Historical source total={len(sources)}, expected={EXPECTED_TOTAL}")
    return sources


def prepare_keys(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    out[f"{prefix}_date_legacy"] = legacy_dates(out["Date"])
    out[f"{prefix}_date_robust"] = parse_dates_robust(out["Date"])
    out[f"{prefix}_team_home"] = out["HomeTeam"].map(normalize_team)
    out[f"{prefix}_team_away"] = out["AwayTeam"].map(normalize_team)
    out[f"{prefix}_date_key"] = out[f"{prefix}_date_robust"].dt.strftime("%Y-%m-%d")
    out[f"{prefix}_key_robust"] = (
        out["Season"].astype("string").str.strip()
        + "|" + out[f"{prefix}_date_key"].fillna("<INVALID_DATE>")
        + "|" + out[f"{prefix}_team_home"]
        + "|" + out[f"{prefix}_team_away"]
    )
    out[f"{prefix}_key_legacy"] = (
        out["Season"].astype("string").str.strip()
        + "|" + out[f"{prefix}_date_legacy"].dt.strftime("%Y-%m-%d").fillna("<INVALID_DATE>")
        + "|" + out[f"{prefix}_team_home"]
        + "|" + out[f"{prefix}_team_away"]
    )
    return out


def main() -> int:
    if (ROOT / FORBIDDEN_FILE).exists():
        print(f"BLIND-FILE GUARD: {FORBIDDEN_FILE} exists locally but is not opened.")
    print("BLIND-FILE GUARD: explicit source allow-list excludes E0 (10).csv.")
    print("No recursive/glob file discovery is used.")

    sources = prepare_keys(load_sources(), "src")
    if not FEATURE_FILE.is_file():
        raise FileNotFoundError(f"Required feature file missing: {FEATURE_FILE}")
    features = pd.read_csv(FEATURE_FILE, encoding="utf-8-sig")
    required = set(KEY_COLS)
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"{FEATURE_FILE.name}: missing key columns: {missing}")
    if len(features) != EXPECTED_TOTAL:
        raise ValueError(
            f"{FEATURE_FILE.name} has {len(features)} rows, expected {EXPECTED_TOTAL}; "
            "audit continues only after this hard invariant is corrected."
        )
    features = features.copy()
    features["FeatureRow"] = range(2, len(features) + 2)
    features = prepare_keys(features, "feat")

    # Row-level parsing diagnostics, by season.
    parse_rows = []
    for label, df, prefix in [
        ("source", sources, "src"),
        ("features", features, "feat"),
    ]:
        for season, group in df.groupby("Season", dropna=False):
            parse_rows.append({
                "dataset": label,
                "season": str(season),
                "rows": int(len(group)),
                "legacy_nat": int(group[f"{prefix}_date_legacy"].isna().sum()),
                "robust_nat": int(group[f"{prefix}_date_robust"].isna().sum()),
                "legacy_parse_recovered_by_robust": int(
                    (group[f"{prefix}_date_legacy"].isna()
                     & group[f"{prefix}_date_robust"].notna()).sum()
                ),
                "blank_team_rows": int(
                    ((group["HomeTeam"].map(normalize_team) == "")
                     | (group["AwayTeam"].map(normalize_team) == "")).sum()
                ),
                "duplicate_robust_keys": int(
                    group[f"{prefix}_key_robust"].duplicated(keep=False).sum()
                ),
            })
    parse_report = pd.DataFrame(parse_rows)

    src_keys = set(sources["src_key_robust"].dropna().astype(str))
    feat_keys = set(features["feat_key_robust"].dropna().astype(str))
    src_key_counts = sources["src_key_robust"].value_counts(dropna=False)
    feat_key_counts = features["feat_key_robust"].value_counts(dropna=False)

    source_only = sources[~sources["src_key_robust"].isin(feat_keys)].copy()
    feature_only = features[~features["feat_key_robust"].isin(src_keys)].copy()

    # Explain unmatched keys using weaker candidate keys. This is diagnostic only:
    # never auto-corrects names or joins records on partial keys.
    source_only["_date_team_pair"] = (
        source_only["Season"].astype("string").str.strip() + "|"
        + source_only["src_date_key"].fillna("<INVALID_DATE>") + "|"
        + source_only["src_team_home"] + "|" + source_only["src_team_away"]
    )
    feature_only["_date_team_pair"] = (
        feature_only["Season"].astype("string").str.strip() + "|"
        + feature_only["feat_date_key"].fillna("<INVALID_DATE>") + "|"
        + feature_only["feat_team_home"] + "|" + feature_only["feat_team_away"]
    )

    # Match by season + normalized teams to identify date-only mismatches.
    source_only["_season_teams"] = (
        source_only["Season"].astype("string").str.strip() + "|"
        + source_only["src_team_home"] + "|" + source_only["src_team_away"]
    )
    feature_only["_season_teams"] = (
        feature_only["Season"].astype("string").str.strip() + "|"
        + feature_only["feat_team_home"] + "|" + feature_only["feat_team_away"]
    )
    feature_team_keys = set(feature_only["_season_teams"])
    source_team_keys = set(source_only["_season_teams"])

    # Match by season + date to identify likely team-label mismatch.
    source_only["_season_date"] = (
        source_only["Season"].astype("string").str.strip() + "|"
        + source_only["src_date_key"].fillna("<INVALID_DATE>")
    )
    feature_only["_season_date"] = (
        feature_only["Season"].astype("string").str.strip() + "|"
        + feature_only["feat_date_key"].fillna("<INVALID_DATE>")
    )
    feature_date_keys = set(feature_only["_season_date"])
    source_date_keys = set(source_only["_season_date"])

    source_only["diagnostic_hint"] = "team/date key differs; inspect raw values"
    source_only.loc[source_only["src_date_robust"].isna(), "diagnostic_hint"] = "source date invalid"
    source_only.loc[
        source_only["_season_teams"].isin(feature_team_keys)
        & source_only["src_date_robust"].notna(),
        "diagnostic_hint"
    ] = "same season+teams exist in unmatched features; inspect date parsing/date value"
    source_only.loc[
        source_only["_season_date"].isin(feature_date_keys),
        "diagnostic_hint"
    ] = "same season+date exists in unmatched features; inspect team labels/order"

    feature_only["diagnostic_hint"] = "team/date key differs; inspect raw values"
    feature_only.loc[feature_only["feat_date_robust"].isna(), "diagnostic_hint"] = "feature date invalid"
    feature_only.loc[
        feature_only["_season_teams"].isin(source_team_keys)
        & feature_only["feat_date_robust"].notna(),
        "diagnostic_hint"
    ] = "same season+teams exist in unmatched sources; inspect date parsing/date value"
    feature_only.loc[
        feature_only["_season_date"].isin(source_date_keys),
        "diagnostic_hint"
    ] = "same season+date exists in unmatched sources; inspect team labels/order"

    # Legacy vs robust merge coverage, using normalized teams.
    legacy_source = set(sources["src_key_legacy"].astype(str))
    legacy_feature = set(features["feat_key_legacy"].astype(str))

    summary = {
        "audit": "audit_match_keys_v1",
        "blind_file": FORBIDDEN_FILE,
        "blind_file_opened": False,
        "source_rows": int(len(sources)),
        "feature_rows": int(len(features)),
        "source_unique_robust_keys": int(sources["src_key_robust"].nunique(dropna=False)),
        "feature_unique_robust_keys": int(features["feat_key_robust"].nunique(dropna=False)),
        "robust_key_intersection": int(len(src_keys & feat_keys)),
        "source_only_rows": int(len(source_only)),
        "feature_only_rows": int(len(feature_only)),
        "source_only_unique_keys": int(source_only["src_key_robust"].nunique()),
        "feature_only_unique_keys": int(feature_only["feat_key_robust"].nunique()),
        "source_legacy_nat": int(sources["src_date_legacy"].isna().sum()),
        "source_robust_nat": int(sources["src_date_robust"].isna().sum()),
        "feature_legacy_nat": int(features["feat_date_legacy"].isna().sum()),
        "feature_robust_nat": int(features["feat_date_robust"].isna().sum()),
        "legacy_key_intersection": int(len(legacy_source & legacy_feature)),
        "source_duplicate_robust_key_rows": int(sources["src_key_robust"].duplicated(keep=False).sum()),
        "feature_duplicate_robust_key_rows": int(features["feat_key_robust"].duplicated(keep=False).sum()),
        "source_duplicate_key_values": int((src_key_counts > 1).sum()),
        "feature_duplicate_key_values": int((feat_key_counts > 1).sum()),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parse_report.to_csv(OUT_DIR / "parse_report_by_season.csv", index=False, encoding="utf-8-sig")
    source_only.to_csv(OUT_DIR / "source_only_unmatched.csv", index=False, encoding="utf-8-sig")
    feature_only.to_csv(OUT_DIR / "feature_only_unmatched.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== MATCH-KEY AUDIT SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\n=== PARSE REPORT BY SEASON ===")
    print(parse_report.to_string(index=False))
    print(f"\nReports written to: {OUT_DIR.resolve()}")
    print("No data was deleted, renamed, or auto-joined.")
    print("Interpret partial-key hints as diagnostics, not confirmed causes.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nAUDIT FAILED CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
