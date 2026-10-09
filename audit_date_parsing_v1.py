# -*- coding: utf-8 -*-
"""Repository-wide static and data-backed audit of historical date parsing.

Scope is intentionally limited to E0.csv through E0 (9).csv.
E0 (10).csv is forbidden and is never opened by this script.
This audit does not alter any research protocol or result.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
HISTORICAL_FILES = ["E0.csv"] + [f"E0 ({i}).csv" for i in range(1, 10)]
FORBIDDEN = "E0 (10).csv"
SEASON_BY_FILE = {
    "E0.csv": "2015/16",
    **{f"E0 ({i}).csv": f"{2015 + i}/{str(2016 + i)[-2:]}" for i in range(1, 10)},
}


def date_parser_calls(source: str, path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_to_datetime = (
            isinstance(func, ast.Attribute)
            and func.attr == "to_datetime"
        )
        if not is_to_datetime:
            continue
        format_kw = next((kw for kw in node.keywords if kw.arg == "format"), None)
        # Explicit formats are not vulnerable to pandas' single-format inference.
        # The audit targets only calls that leave format inference at its default.
        if format_kw is None:
            hits.append((node.lineno, ast.get_source_segment(source, node) or "pd.to_datetime(...)"))
    return hits


def main() -> int:
    print("REPOSITORY-WIDE DATE PARSING AUDIT")
    print("=" * 100)
    if not (ROOT / FORBIDDEN).exists():
        print(f"BLIND FILE STATUS: {FORBIDDEN} absent from working tree; not accessed.")
    else:
        print(f"BLIND FILE STATUS: {FORBIDDEN} exists; audit policy still forbids opening it.")
    print("Historical scope: " + ", ".join(HISTORICAL_FILES))
    print()

    season_impact: dict[str, dict[str, int]] = {}
    for filename in HISTORICAL_FILES:
        path = ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required historical file missing: {filename}")
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "Date" not in df.columns:
            raise ValueError(f"{filename} has no Date column")
        legacy = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        mixed = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
        season_impact[filename] = {
            "rows": len(df),
            "legacy_nat": int(legacy.isna().sum()),
            "mixed_nat": int(mixed.isna().sum()),
            "legacy_only_nat": int((legacy.isna() & mixed.notna()).sum()),
            "mixed_only_nat": int((legacy.notna() & mixed.isna()).sum()),
        }

    print("DATA-BACKED PARSER COMPARISON")
    print(f"{'File':<14} {'Season':<8} {'Rows':>5} {'Legacy NaT':>11} {'Mixed NaT':>10} {'Recovered':>10} {'Regressed':>10}")
    print("-" * 82)
    total_rows = total_legacy_nat = total_mixed_nat = total_recovered = total_regressed = 0
    affected_files: set[str] = set()
    for filename in HISTORICAL_FILES:
        d = season_impact[filename]
        print(f"{filename:<14} {SEASON_BY_FILE[filename]:<8} {d['rows']:>5} {d['legacy_nat']:>11} {d['mixed_nat']:>10} {d['legacy_only_nat']:>10} {d['mixed_only_nat']:>10}")
        total_rows += d["rows"]
        total_legacy_nat += d["legacy_nat"]
        total_mixed_nat += d["mixed_nat"]
        total_recovered += d["legacy_only_nat"]
        total_regressed += d["mixed_only_nat"]
        if d["legacy_only_nat"] > 0:
            affected_files.add(filename)
    print("-" * 82)
    print(f"{'TOTAL':<14} {'':<8} {total_rows:>5} {total_legacy_nat:>11} {total_mixed_nat:>10} {total_recovered:>10} {total_regressed:>10}")
    print()

    print("STATIC SCRIPT SCAN")
    py_files = sorted(
        p for p in ROOT.glob("*.py")
        if p.name not in {"audit_date_parsing_v1.py"}
    )
    risky_scripts: list[tuple[str, list[str], list[tuple[int, str]]]] = []
    all_legacy_calls = 0
    for path in py_files:
        source = path.read_text(encoding="utf-8-sig")
        calls = date_parser_calls(source, path)
        if not calls:
            continue
        all_legacy_calls += len(calls)
        refs = set(re.findall(r"""['"](E0(?: \(\d+\))?\.csv)['"]""", source))
        relevant = sorted(refs & affected_files)
        if relevant:
            risky_scripts.append((path.name, relevant, calls))
    if risky_scripts:
        for name, refs, calls in risky_scripts:
            lines = ",".join(str(line) for line, _ in calls)
            print(f"RISK: {name} | affected source(s): {', '.join(refs)} | default-inference to_datetime call line(s): {lines}")
    else:
        print("No scripts matched both a non-mixed parser call and a historically affected input file.")
    print()
    print(f"Python scripts scanned: {len(py_files)}")
    print(f"Default-inference pd.to_datetime call sites found: {all_legacy_calls}")
    print(f"Potentially affected scripts: {len(risky_scripts)}")
    print(f"Historical rows recoverable by mixed parsing: {total_recovered}")
    print(f"Rows made invalid by mixed parsing: {total_regressed}")
    print(f"{FORBIDDEN} reads: 0 (not opened; excluded by design)")

    if total_mixed_nat != 0 or total_regressed != 0:
        print("AUDIT RESULT: FAIL — mixed parser did not parse every historical Date value cleanly.")
        return 1
    if not affected_files:
        print("AUDIT RESULT: PASS — no legacy-only date parse failures found in historical inputs.")
    else:
        print("AUDIT RESULT: FINDINGS — review listed scripts before changing any frozen research protocol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
