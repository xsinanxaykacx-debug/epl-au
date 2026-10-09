# -*- coding: utf-8 -*-
"""Read-only sheet/cell comparison for the frozen H9 result workbook."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def same_cell(a, b, numeric):
    if pd.isna(a) and pd.isna(b):
        return True, 0.0
    if pd.isna(a) or pd.isna(b):
        return False, None
    if numeric:
        try:
            x, y = float(a), float(b)
            if np.isclose(x, y, rtol=0.0, atol=1e-10):
                return True, abs(x-y)
            return False, abs(x-y)
        except (TypeError, ValueError):
            pass
    return str(a) == str(b), None

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--out", default="audit_match_keys_v1_output")
    args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    old=pd.read_excel(args.before, sheet_name=None)
    new=pd.read_excel(args.after, sheet_name=None)
    report={"before":args.before,"after":args.after,"same_sheet_names":list(old)==list(new),"sheets":{}}
    for name in sorted(set(old)|set(new)):
        if name not in old or name not in new:
            report["sheets"][name]={"status":"missing_sheet","before_present":name in old,"after_present":name in new}
            continue
        a,b=old[name],new[name]
        info={"before_shape":list(a.shape),"after_shape":list(b.shape),"different_cells":0,"different_columns":[],"max_abs_numeric_difference":0.0}
        if a.shape != b.shape:
            info["status"]="shape_mismatch"
            report["sheets"][name]=info
            continue
        changed=[]
        for j,col in enumerate(a.columns):
            col_changed=0
            numeric=pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b.iloc[:,j])
            for i in range(len(a)):
                equal,diff=same_cell(a.iloc[i,j],b.iloc[i,j],numeric)
                if not equal:
                    col_changed+=1
                    if len(changed)<10000:
                        changed.append({"sheet":name,"row":i+2,"column":str(col),"before":str(a.iloc[i,j]),"after":str(b.iloc[i,j]),"abs_diff":diff})
                if diff is not None:
                    info["max_abs_numeric_difference"]=max(info["max_abs_numeric_difference"],diff)
            if col_changed:
                info["different_columns"].append({"column":str(col),"different_cells":col_changed})
                info["different_cells"]+=col_changed
        info["status"]="same" if info["different_cells"]==0 else "values_differ"
        report["sheets"][name]=info
        if changed:
            pd.DataFrame(changed).to_csv(out/f"h9_results_diff_{name}.csv",index=False,encoding="utf-8-sig")
    report["all_sheets_identical"] = report["same_sheet_names"] and all(x.get("status")=="same" for x in report["sheets"].values())
    (out/"h9_result_reproducibility.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("=== H9 RESULT REPRODUCIBILITY ===")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
