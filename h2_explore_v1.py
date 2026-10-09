# -*- coding: utf-8 -*-
"""
h2_explore_v1.py
H2 EXPLORE PROTOKOLÜ v1.0 (FROZEN) — birebir implementasyon
Tarih: 2026-10-08

Kapsam:
- Train:       2019/20 + 2020/21 + 2021/22 + 2022/23
- Validation:  2023/24
- OOS:         2024/25 (tek seferlik)

KİLİTLER:
- E0 (10).csv ASLA OKUNMAZ.
- Train'de fit edilen katsayılar Validation/OOS'ta YENİDEN FIT EDİLMEZ.
  Validation/OOS yalnızca Train modelinin out-of-sample tahminidir.
- Family-based eksik maç filtresi (envanter_v2.xlsx → Eksik_Maclar, AH ailesi).
- Polinom yok, threshold yok, betting/ROI yok.
- OOS tek seferlik.

KİLİTLİ KARARLAR:
- Outcome: GoalDiff = FTHG − FTAG (sürekli)
- Değişkenler: Movement = AHCh − AHh (sürekli), Elo_Diff (sürekli)
- Modeller:
    M0 = GoalDiff ~ Elo_Diff
    M1 = GoalDiff ~ Elo_Diff + Movement
    M2 = GoalDiff ~ Elo_Diff + Movement + Elo_x_Move
    M3 = GoalDiff ~ Movement
- Ana kriter: ΔR²_OOS = R²_OOS(M1) − R²_OOS(M0)
- Çıktı: h2_explore_v1_results.xlsx

DÜZELTME (bu sürüm):
- _okuma_sayaci = defaultdict(int) — KeyError giderildi.
  (Yalnızca loglama ile ilgili teknik hata; sonuç değişmez.)
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ==================================================================
# 0. SABİTLER
# ==================================================================
H1_FEATURES = "h1_features.csv"
ENVANTER = "envanter_v2.xlsx"
ENVANTER_SHEET = "Eksik_Maclar"
CIKTI = "h2_explore_v1_results.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

MARKET_DOSYALARI = {
    "2019/20": "E0 (4).csv",
    "2020/21": "E0 (5).csv",
    "2021/22": "E0 (6).csv",
    "2022/23": "E0 (7).csv",
    "2023/24": "E0 (8).csv",
    "2024/25": "E0 (9).csv",
}
MARKET_SUTUNLARI = ["Season", "Date", "HomeTeam", "AwayTeam", "AHh", "AHCh"]

TRAIN_SEZONLAR = ["2019/20", "2020/21", "2021/22", "2022/23"]
VALIDATION_SEZON = "2023/24"
OOS_SEZON = "2024/25"

_okuma_sayaci = defaultdict(int)


# ==================================================================
# 1. YARDIMCILAR
# ==================================================================
def _guvenli_oku(dosya):
    ad = os.path.basename(dosya)
    if ad in YASAKLI_DOSYALAR:
        raise RuntimeError(f"YASAKLI DOSYA: {ad}")
    _okuma_sayaci[ad] += 1
    return pd.read_csv(dosya, encoding="utf-8-sig")


def _mac_anahtari(df):
    tarih = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
    return (df["Season"].astype(str).str.strip()
            + "|" + tarih.fillna("NA")
            + "|" + df["HomeTeam"].astype(str).str.strip()
            + "|" + df["AwayTeam"].astype(str).str.strip())


# ==================================================================
# 2. VERİ YÜKLEME (family-based AH filtresi)
# ==================================================================
def veri_yukle():
    if not os.path.exists(H1_FEATURES):
        raise RuntimeError(f"{H1_FEATURES} bulunamadı.")
    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")

    # --- Eksik maç filtresi (AH ailesi) ---
    eksik = pd.read_excel(ENVANTER, sheet_name=ENVANTER_SHEET)
    eksik["_key"] = _mac_anahtari(eksik)
    eksik_ah = set(eksik.loc[eksik["AH_Eksik"] == True, "_key"].dropna().astype(str))

    feat["_key"] = _mac_anahtari(feat)

    # --- Market yükleme ---
    frames = []
    for sezon, dosya in MARKET_DOSYALARI.items():
        ad = os.path.basename(dosya)
        if ad in YASAKLI_DOSYALAR:
            raise RuntimeError(f"YASAKLI DOSYA: {ad}")
        df = _guvenli_oku(dosya)
        df["Season"] = sezon
        eksik_sutun = [s for s in MARKET_SUTUNLARI if s not in df.columns]
        if eksik_sutun:
            raise RuntimeError(f"{ad} içinde eksik sütunlar: {eksik_sutun}")
        frames.append(df[MARKET_SUTUNLARI].copy())
    market = pd.concat(frames, ignore_index=True)
    market["_key"] = _mac_anahtari(market)

    # --- Merge ---
    df = feat.merge(market[["_key", "AHh", "AHCh"]], on="_key", how="inner")

    # --- AH ailesi eksik maç filtresi ---
    df["_eksik_ah"] = df["_key"].isin(eksik_ah)
    onceki = len(df)
    df = df[~df["_eksik_ah"]].copy().reset_index(drop=True)
    sonraki = len(df)
    print(f"  AH ailesi eksik maç filtresi: {onceki} → {sonraki} (dışlanan: {onceki-sonraki})")

    # --- Türetilmiş değişkenler ---
    df["Movement"] = df["AHCh"] - df["AHh"]
    df["Elo_Diff"] = df["Home_Elo_Pre"] - df["Away_Elo_Pre"]
    df["GoalDiff"] = df["FTHG"] - df["FTAG"]
    df["Elo_x_Move"] = df["Elo_Diff"] * df["Movement"]

    df = df.dropna(subset=["Movement", "Elo_Diff", "GoalDiff", "Elo_x_Move"]).reset_index(drop=True)
    return df


# ==================================================================
# 3. MODEL SPESİFİKASYONLARI
# ==================================================================
def model_X(df, model_kod):
    if model_kod == "M0":
        return df[["Elo_Diff"]]
    elif model_kod == "M1":
        return df[["Elo_Diff", "Movement"]]
    elif model_kod == "M2":
        return df[["Elo_Diff", "Movement", "Elo_x_Move"]]
    elif model_kod == "M3":
        return df[["Movement"]]
    else:
        raise ValueError(f"Bilinmeyen model: {model_kod}")


MODEL_META = {
    "M0": "Elo-only (baseline)",
    "M1": "Elo + Movement (ana)",
    "M2": "Elo + Movement + Interaction",
    "M3": "Movement-only (baseline)",
}


# ==================================================================
# 4. METRİKLER
# ==================================================================
def metrikler(y_true, y_pred, k):
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if n > k and ss_tot > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
    else:
        adj_r2 = np.nan

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "N": n,
        "R2": round(r2, 6),
        "Adj_R2": round(adj_r2, 6),
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
    }


# ==================================================================
# 5. EĞİTİM / DEĞERLENDİRME
# ==================================================================
def train_fit(df_train, model_kod):
    X = model_X(df_train, model_kod)
    X = sm.add_constant(X, has_constant="add")
    y = df_train["GoalDiff"]
    return sm.OLS(y, X).fit()


def predict_with_train_model(model, df_test, model_kod):
    X = model_X(df_test, model_kod)
    X = sm.add_constant(X, has_constant="add")
    X = X[model.params.index.tolist()]
    return model.predict(X)


def katsayi_tablosu(model, model_kod, bolme, fit_kaynagi):
    params = model.params
    bse = model.bse
    tvals = model.tvalues
    pvals = model.pvalues
    ci = model.conf_int()
    ci.columns = ["CI_lo", "CI_hi"]

    rows = []
    for name in params.index:
        rows.append({
            "Bolme": bolme,
            "Model": model_kod,
            "Katsayi": name,
            "Deger": round(float(params[name]), 6),
            "SE": round(float(bse[name]), 6),
            "t": round(float(tvals[name]), 6),
            "p": round(float(pvals[name]), 6),
            "CI_lo": round(float(ci.loc[name, "CI_lo"]), 6),
            "CI_hi": round(float(ci.loc[name, "CI_hi"]), 6),
            "Fit_kaynagi": fit_kaynagi,
        })
    return rows


# ==================================================================
# 6. ANA AKIŞ
# ==================================================================
def main():
    print("=" * 80)
    print("H2 EXPLORE v1.0 — FROZEN")
    print("E0 (10).csv KESİNLİKLE OKUNMAYACAK.")
    print("Train katsayıları Validation/OOS'ta YENİDEN FIT EDİLMEZ.")
    print("=" * 80)

    print("\n--- Veri yükleniyor ---")
    df = veri_yukle()
    print(f"  Toplam satır: {len(df)}")
    print(f"  Sezonlar: {sorted(df['Season'].unique())}")

    train = df[df["Season"].isin(TRAIN_SEZONLAR)].copy().reset_index(drop=True)
    val   = df[df["Season"] == VALIDATION_SEZON].copy().reset_index(drop=True)
    oos   = df[df["Season"] == OOS_SEZON].copy().reset_index(drop=True)

    print(f"\n  Train N: {len(train)}")
    print(f"  Validation N: {len(val)}")
    print(f"  OOS N: {len(oos)}")

    # -------- TRAIN: 4 model fit (yalnızca Train'de) --------
    print("\n--- TRAIN: modeller fit ediliyor (yalnızca bu bölmede) ---")
    train_models = {}
    for kod in ["M0", "M1", "M2", "M3"]:
        m = train_fit(train, kod)
        train_models[kod] = m
        print(f"  {kod} ({MODEL_META[kod]}): R²={m.rsquared:.6f}, AdjR²={m.rsquared_adj:.6f}")

    # -------- ÖZET --------
    ozet_rows = []
    katsayi_train_rows = []
    katsayi_val_rows = []
    katsayi_oos_rows = []

    for kod in ["M0", "M1", "M2", "M3"]:
        m = train_models[kod]
        k = len(m.params)

        y_pred_train = m.predict(
            sm.add_constant(model_X(train, kod), has_constant="add")[m.params.index]
        )
        met_train = metrikler(train["GoalDiff"], y_pred_train, k)

        y_pred_val = predict_with_train_model(m, val, kod)
        met_val = metrikler(val["GoalDiff"], y_pred_val, k)

        y_pred_oos = predict_with_train_model(m, oos, kod)
        met_oos = metrikler(oos["GoalDiff"], y_pred_oos, k)

        ozet_rows.append({
            "Model": kod, "Aciklama": MODEL_META[kod],
            "Train_N": met_train["N"], "Train_R2": met_train["R2"],
            "Train_AdjR2": met_train["Adj_R2"], "Train_MAE": met_train["MAE"],
            "Train_RMSE": met_train["RMSE"],
            "Val_N": met_val["N"], "Val_R2": met_val["R2"],
            "Val_AdjR2": met_val["Adj_R2"], "Val_MAE": met_val["MAE"],
            "Val_RMSE": met_val["RMSE"],
            "OOS_N": met_oos["N"], "OOS_R2": met_oos["R2"],
            "OOS_AdjR2": met_oos["Adj_R2"], "OOS_MAE": met_oos["MAE"],
            "OOS_RMSE": met_oos["RMSE"],
        })

        katsayi_train_rows.extend(
            katsayi_tablosu(m, kod, "Train", "Train (yeni fit)")
        )
        katsayi_val_rows.extend(
            katsayi_tablosu(m, kod, "Validation",
                            "Train (katsayı uygulandı, refit YOK)")
        )
        katsayi_oos_rows.extend(
            katsayi_tablosu(m, kod, "OOS",
                            "Train (katsayı uygulandı, refit YOK)")
        )

    ozet_df = pd.DataFrame(ozet_rows)

    print("\n--- ÖZET TABLO ---")
    print(ozet_df[["Model", "Train_R2", "Val_R2", "OOS_R2",
                    "Train_MAE", "Val_MAE", "OOS_MAE"]].to_string(index=False))

    # -------- ΔR²_OOS --------
    r2_oos_m0 = ozet_df.loc[ozet_df["Model"] == "M0", "OOS_R2"].values[0]
    r2_oos_m1 = ozet_df.loc[ozet_df["Model"] == "M1", "OOS_R2"].values[0]
    r2_oos_m2 = ozet_df.loc[ozet_df["Model"] == "M2", "OOS_R2"].values[0]
    r2_oos_m3 = ozet_df.loc[ozet_df["Model"] == "M3", "OOS_R2"].values[0]

    delta_r2_m1_m0 = r2_oos_m1 - r2_oos_m0
    delta_r2_m2_m0 = r2_oos_m2 - r2_oos_m0

    print(f"\n--- INCREMENTAL OOS R² ---")
    print(f"  R²_OOS(M0) Elo-only         : {r2_oos_m0:.6f}")
    print(f"  R²_OOS(M1) Elo+Movement     : {r2_oos_m1:.6f}")
    print(f"  R²_OOS(M2) Elo+Mov+Inter    : {r2_oos_m2:.6f}")
    print(f"  R²_OOS(M3) Movement-only    : {r2_oos_m3:.6f}")
    print(f"  ΔR²_OOS (M1 − M0)           : {delta_r2_m1_m0:+.6f}   ← ANA KRİTER")
    print(f"  ΔR²_OOS (M2 − M0)           : {delta_r2_m2_m0:+.6f}")

    print("\n--- KATSAYI STABİLİTESİ (Train) ---")
    for kod in ["M1", "M2"]:
        m = train_models[kod]
        if "Movement" in m.params.index:
            print(f"  {kod} Movement katsayısı (Train): {float(m.params['Movement']):+.6f}")

    # -------- Excel --------
    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        ozet_df.to_excel(w, sheet_name="Ozet_Ana_Tablo", index=False)

        inc_df = pd.DataFrame([{
            "R2_OOS_M0_Elo_only": round(r2_oos_m0, 6),
            "R2_OOS_M1_Elo_Movement": round(r2_oos_m1, 6),
            "R2_OOS_M2_Elo_Move_Inter": round(r2_oos_m2, 6),
            "R2_OOS_M3_Movement_only": round(r2_oos_m3, 6),
            "Delta_R2_M1_minus_M0": round(delta_r2_m1_m0, 6),
            "Delta_R2_M2_minus_M0": round(delta_r2_m2_m0, 6),
        }])
        inc_df.to_excel(w, sheet_name="Incremental_R2", index=False)

        pd.DataFrame(katsayi_train_rows).to_excel(w, sheet_name="Katsayi_Train", index=False)
        pd.DataFrame(katsayi_val_rows).to_excel(w, sheet_name="Katsayi_Validation", index=False)
        pd.DataFrame(katsayi_oos_rows).to_excel(w, sheet_name="Katsayi_OOS", index=False)

        stab_rows = []
        for kod in ["M0", "M1", "M2", "M3"]:
            m = train_models[kod]
            for name in m.params.index:
                stab_rows.append({
                    "Model": kod,
                    "Katsayi": name,
                    "Deger_Train": round(float(m.params[name]), 6),
                    "SE_Train": round(float(m.bse[name]), 6),
                    "p_Train": round(float(m.pvalues[name]), 6),
                    "Not": "Validation/OOS'ta refit YOK; katsayı aynıdır.",
                })
        pd.DataFrame(stab_rows).to_excel(w, sheet_name="Katsayi_Stabilite", index=False)

        pd.DataFrame([{
            "Not": "Validation ve OOS katsayıları Train'de fit edilmiştir; "
                   "refit YAPILMAMIŞTIR. Katsayı stabilitesi bu nedenle yalnızca "
                   "Train değerleri üzerinden raporlanır. Bu, protokolün bilinçli "
                   "tercihidir — Train→OOS genellenebilirliği testinin temeli.",
        }]).to_excel(w, sheet_name="Katsayi_Anlam", index=False)

        pd.DataFrame([{
            "Protokol": "H2 EXPLORE v1.0 (FROZEN)",
            "Tarih": "2026-10-08",
            "Outcome": "GoalDiff = FTHG - FTAG",
            "Degiskenler": "Movement = AHCh - AHh (surekli); Elo_Diff (surekli)",
            "Modeller": "M0 Elo-only, M1 Elo+Movement, M2 +Interaction, M3 Movement-only",
            "Ana_kriter": "Delta R2_OOS = R2_OOS(M1) - R2_OOS(M0)",
            "Destek": "MAE, RMSE, katsayi stabilitesi",
            "Train": "2019/20 - 2022/23",
            "Validation": "2023/24 (Train katsayilariyla tahmin)",
            "OOS": "2024/25 (Train katsayilariyla tahmin)",
            "2026_27": "KILITLI - okunmadi",
            "Train_refit": "YOK - katsayilar Validation/OOS'ta yeniden fit edilmedi",
            "Eksik_mac_filtresi": "envanter_v2.xlsx Eksik_Maclar AH ailesi",
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} başarıyla yazıldı.")

    print("\n--- Dosya okuma sayacı ---")
    for ad in sorted(_okuma_sayaci.keys()):
        print(f"  {ad}: READ {_okuma_sayaci[ad]} TIMES")
    for ad in YASAKLI_DOSYALAR:
        print(f"  [YASAKLI] {ad}: READ {_okuma_sayaci.get(ad, 0)} TIMES")


if __name__ == "__main__":
    main()