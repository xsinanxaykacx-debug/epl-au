# -*- coding: utf-8 -*-
"""Compare committed H1 features with a clean rebuild; never edits either input."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLS = ["Season", "Date", "HomeTeam", "AwayTeam"]


def norm_team(v: object) -> str:
    if pd.isna(v):
        return ""
    s = unicodedata.normalize("NFKC", str(v))
    return re.sub(r"\s+", " ", s).strip().casefold()


def parse_dates(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, dayfirst=True, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        raw = s.astype("string").str.strip()
        out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            mask = out.isna() & raw.notna()
            if mask.any():
                out.loc[mask] = pd.to_datetime(raw.loc[mask], format=fmt, errors="coerce")
        return out


def add_key(df: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = sorted(set(KEY_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"{label}: missing key columns: {missing}")
    out = df.copy()
    dt = parse_dates(out["Date"])
    if dt.isna().any():
        raise ValueError(f"{label}: {int(dt.isna().sum())} invalid dates")
    out["_audit_key"] = (
        out["Season"].astype("string").str.strip()
        + "|" + dt.dt.strftime("%Y-%m-%d")
        + "|" + out["HomeTeam"].map(norm_team)
        + "|" + out["AwayTeam"].map(norm_team)
    )
    if out["_audit_key"].duplicated().any():
        raise ValueError(f"{label}: duplicate match keys={int(out['_audit_key'].duplicated(keep=False).sum())}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--out", default="audit_match_keys_v1_output")
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    before = add_key(pd.read_csv(args.before, encoding="utf-8-sig"), "before")
    after = add_key(pd.read_csv(args.after, encoding="utf-8-sig"), "after")
    bkeys, akeys = set(before["_audit_key"]), set(after["_audit_key"])
    common = sorted(bkeys & akeys)
    before_only, after_only = sorted(bkeys - akeys), sorted(akeys - bkeys)

    summary = {
        "before_file": str(args.before),
        "after_file": str(args.after),
        "before_rows": int(len(before)),
        "after_rows": int(len(after)),
        "before_columns": int(len(before.columns) - 1),
        "after_columns": int(len(after.columns) - 1),
        "before_unique_keys": int(len(bkeys)),
        "after_unique_keys": int(len(akeys)),
        "common_keys": int(len(common)),
        "before_only_keys": int(len(before_only)),
        "after_only_keys": int(len(after_only)),
        "common_columns": [],
        "different_columns": [],
        "rows_with_any_value_difference": 0,
        "numeric_cell_differences": 0,
        "categorical_cell_differences": 0,
        "max_abs_numeric_difference": 0.0,
    }

    if before_only:
        before.loc[before["_audit_key"].isin(before_only)].to_csv(out_dir / "feature_before_only.csv", index=False, encoding="utf-8-sig")
    if after_only:
        after.loc[after["_audit_key"].isin(after_only)].to_csv(out_dir / "feature_after_only.csv", index=False, encoding="utf-8-sig")

    b = before[before["_audit_key"].isin(common)].set_index("_audit_key").sort_index()
    a = after[after["_audit_key"].isin(common)].set_index("_audit_key").sort_index()
    columns = sorted((set(before.columns) & set(after.columns)) - set(KEY_COLS) - {"_audit_key", "_mac_id", "_target_tarih"})
    summary["common_columns"] = columns
    changed_rows = []
    column_rows = []

    for col in columns:
        x, y = b[col], a[col]
        x_num, y_num = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        numeric = (x_num.notna() | y_num.notna() | x.isna() | y.isna())
        # Treat as numeric only if every non-null value in both versions converts.
        all_convertible = (
            ((x.isna()) | x_num.notna()).all()
            and ((y.isna()) | y_num.notna()).all()
        )
        if all_convertible and (pd.api.types.is_numeric_dtype(x) or pd.api.types.is_numeric_dtype(y)):
            xv, yv = x_num.to_numpy(dtype=float), y_num.to_numpy(dtype=float)
            equal = np.isclose(xv, yv, rtol=0.0, atol=1e-12, equal_nan=True)
            valid = np.isfinite(xv) & np.isfinite(yv)
            diffs = np.abs(xv[valid] - yv[valid])
            max_abs = float(diffs.max()) if len(diffs) else 0.0
            n_diff = int((~equal).sum())
            summary["numeric_cell_differences"] += n_diff
        else:
            xs, ys = x.astype("string"), y.astype("string")
            equal = (xs.eq(ys) | (xs.isna() & ys.isna())).to_numpy(dtype=bool)
            max_abs = None
            n_diff = int((~equal).sum())
            summary["categorical_cell_differences"] += n_diff
        if n_diff:
            summary["different_columns"].append(col)
            column_rows.append({
                "column": col,
                "changed_cells": n_diff,
                "common_rows": int(len(common)),
                "max_abs_numeric_difference": max_abs,
            })
            for i in np.flatnonzero(~equal)[:2000]:
                key = b.index[i]
                if len(changed_rows) < 20000:
                    changed_rows.append({
                        "key": key,
                        "column": col,
                        "before": str(x.iloc[i]),
                        "after": str(y.iloc[i]),
                    })

    if changed_rows:
        pd.DataFrame(changed_rows).to_csv(out_dir / "feature_value_differences.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(column_rows).to_csv(out_dir / "feature_rebuild_column_differences.csv", index=False, encoding="utf-8-sig")
    # Count unique rows touched by any changed cell.
    summary["rows_with_any_value_difference"] = int(len({r["key"] for r in changed_rows}))
    (out_dir / "feature_rebuild_comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== FEATURE REBUILD COMPARISON ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Reports: {out_dir.resolve()}")
    # Different feature values are a finding, not an automatic CI failure.
    # Fail only on structural key coverage, which must be complete.
    return 0 if len(common) == len(before) == len(after) else 2


if __name__ == "__main__":
    raise SystemExit(main())
