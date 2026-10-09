# -*- coding: utf-8 -*-
"""Read-only, key-aligned comparison for frozen H9 result workbooks."""
from __future__ import annotations
import argparse, json, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

KEY_COLS = ["Season", "Date", "HomeTeam", "AwayTeam"]

def norm_team(v):
    if pd.isna(v): return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(v))).strip().casefold()

def parse_dates(s):
    try:
        return pd.to_datetime(s, dayfirst=True, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        raw=s.astype("string").str.strip()
        out=pd.Series(pd.NaT,index=s.index,dtype="datetime64[ns]")
        for fmt in ("%d/%m/%Y","%d/%m/%y","%Y-%m-%d","%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M"):
            mask=out.isna() & raw.notna()
            if mask.any(): out.loc[mask]=pd.to_datetime(raw.loc[mask],format=fmt,errors="coerce")
        return out

def align_sheet(df, label):
    if not all(c in df.columns for c in KEY_COLS):
        return df.copy(), False
    out=df.copy()
    dates=parse_dates(out["Date"])
    if dates.isna().any():
        raise ValueError(f"{label}: invalid dates in result sheet")
    out["_match_key"]=(
        out["Season"].astype("string").str.strip()+"|"+
        dates.dt.strftime("%Y-%m-%d")+"|"+
        out["HomeTeam"].map(norm_team)+"|"+out["AwayTeam"].map(norm_team)
    )
    if out["_match_key"].duplicated().any():
        raise ValueError(f"{label}: duplicate match keys in result sheet")
    return out.set_index("_match_key").sort_index(), True

def cell_equal(a,b,numeric):
    if pd.isna(a) and pd.isna(b): return True,0.0
    if pd.isna(a) or pd.isna(b): return False,None
    if numeric:
        try:
            x,y=float(a),float(b)
            if np.isclose(x,y,rtol=0.0,atol=1e-10): return True,abs(x-y)
            return False,abs(x-y)
        except (TypeError,ValueError): pass
    return str(a)==str(b),None

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--before",required=True)
    p.add_argument("--after",required=True)
    p.add_argument("--out",default="audit_match_keys_v1_output")
    args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    old=pd.read_excel(args.before,sheet_name=None)
    new=pd.read_excel(args.after,sheet_name=None)
    report={"before":args.before,"after":args.after,"same_sheet_names":list(old)==list(new),"sheets":{}}
    for name in sorted(set(old)|set(new)):
        if name not in old or name not in new:
            report["sheets"][name]={"status":"missing_sheet","before_present":name in old,"after_present":name in new}
            continue
        a,key_a=align_sheet(old[name],f"before/{name}")
        b,key_b=align_sheet(new[name],f"after/{name}")
        use_keys=key_a and key_b
        info={"comparison":"match_key" if use_keys else "row_order","before_shape":list(old[name].shape),"after_shape":list(new[name].shape),"before_only_rows":0,"after_only_rows":0,"different_cells":0,"different_columns":[],"max_abs_numeric_difference":0.0}
        if use_keys:
            ka,kb=set(a.index),set(b.index)
            info["before_only_rows"]=len(ka-kb)
            info["after_only_rows"]=len(kb-ka)
            common=sorted(ka&kb)
            a,b=a.loc[common],b.loc[common]
            cols=sorted((set(a.columns)&set(b.columns))-{"_match_key"})
        else:
            if a.shape!=b.shape:
                info["status"]="shape_mismatch"
                report["sheets"][name]=info
                continue
            cols=[c for c in a.columns if c in b.columns]
            if list(a.columns)!=list(b.columns):
                info["column_order_or_names_differ"]=True
        diff_records=[]
        for col in cols:
            x,y=a[col],b[col]
            numeric=pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)
            n_diff=0
            for i in range(len(a)):
                equal,diff=cell_equal(x.iloc[i],y.iloc[i],numeric)
                if not equal:
                    n_diff+=1
                    if len(diff_records)<10000:
                        key=str(a.index[i]) if use_keys else f"row={i+2}"
                        diff_records.append({"sheet":name,"key_or_row":key,"column":str(col),"before":str(x.iloc[i]),"after":str(y.iloc[i]),"abs_diff":diff})
                if diff is not None:
                    info["max_abs_numeric_difference"]=max(info["max_abs_numeric_difference"],diff)
            if n_diff:
                info["different_columns"].append({"column":str(col),"different_cells":n_diff})
                info["different_cells"]+=n_diff
        if diff_records:
            pd.DataFrame(diff_records).to_csv(out/f"h9_results_diff_{name}.csv",index=False,encoding="utf-8-sig")
        info["status"]="same" if info["different_cells"]==0 and info["before_only_rows"]==0 and info["after_only_rows"]==0 else "values_differ"
        report["sheets"][name]=info
    report["all_sheets_identical"]=report["same_sheet_names"] and all(x.get("status")=="same" for x in report["sheets"].values())
    (out/"h9_result_reproducibility.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("=== H9 RESULT REPRODUCIBILITY (KEY-ALIGNED) ===")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if "Ozet" in old and "Ozet" in new:
        print("=== BASELINE OZET ===")
        print(old["Ozet"].to_string(index=False))
        print("=== REGENERATED OZET ===")
        print(new["Ozet"].to_string(index=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
