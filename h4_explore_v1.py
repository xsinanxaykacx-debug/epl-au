# -*- coding: utf-8 -*-
"""
h4_explore_v1.py
H4 FROZEN PROTOCOL v1.0 — birebir implementasyon
Tarih: 2026-10-08

DÜZELTME (bu sürüm):
- Kriter 4 (K4): M1 ve M2'deki aynı dummy katsayılarının işaret karşılaştırması
  üzerinden deterministik yapıldı.
- Karar ağacı önceliği protokole sadık hale getirildi:
  ΔR²≤0 VEYA MAE kötüleşiyor VEYA K4 fail → NOT SUPPORTED.

Protokol değişmedi. Sadece kod sadık hale getirildi.
"""

import os
import sys
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
CIKTI = "h4_explore_v1_results.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

MARKET_DOSYALARI = {
    "2019/20": "E0 (4).csv",
    "2020/21": "E0 (5).csv",
    "2021/22": "E0 (6).csv",
    "2022/23": "E0 (7).csv",
    "2023/24": "E0 (8).csv",
    "2024/25": "E0 (9).csv",
}
MARKET_SUTUNLARI = ["Season", "Date", "HomeTeam", "AwayTeam",
                    "AHh", "AHCh", "B365AHH", "B365AHA", "B365CAHH", "B365CAHA",
                    "B365H", "B365D", "B365A",
                    "FTHG", "FTAG"]

TRAIN_SEZONLAR = ["2019/20", "2020/21", "2021/22", "2022/23"]
VALIDATION_SEZON = "2023/24"
OOS_SEZON = "2024/25"

BEKLENEN_ILK = 2280
BEKLENEN_SON = 2278

DELTA_R2_ESIK = 0.005

_okuma_sayaci = defaultdict(int)


# ==================================================================
# 1. YARDIMCILAR
# ==================================================================
def _guvenli_oku(dosya):
    ad = os.path.basename(dosya)
    if ad in YASAKLI_DOSYALAR:
        raise RuntimeError(f"YASAKLI DOSYA OKUNAMAZ: {ad}")
    _okuma_sayaci[ad] += 1
    return pd.read_csv(dosya, encoding="utf-8-sig")


def _mac_anahtari(df):
    tarih = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
    return (df["Season"].astype(str).str.strip()
            + "|" + tarih.fillna("NA")
            + "|" + df["HomeTeam"].astype(str).str.strip()
            + "|" + df["AwayTeam"].astype(str).str.strip())


def ah_yon(ahh):
    if pd.isna(ahh): return None
    if ahh < 0: return "HOME"
    elif ahh > 0: return "AWAY"
    else: return "NOTR"


def x12_yon(h, a):
    if pd.isna(h) or pd.isna(a): return None
    if h < a: return "HOME"
    elif a < h: return "AWAY"
    else: return "NOTR"


def cross_market(ah_y, x12_y):
    if ah_y is None or x12_y is None: return None
    if ah_y == "NOTR" or x12_y == "NOTR": return "NEUTRAL"
    if ah_y == x12_y: return "AGREEMENT"
    else: return "DISAGREEMENT"


# ==================================================================
# 2. VERİ YÜKLEME
# ==================================================================
def veri_yukle():
    if not os.path.exists(H1_FEATURES):
        raise RuntimeError(f"{H1_FEATURES} bulunamadı.")
    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")

    eksik = pd.read_excel(ENVANTER, sheet_name=ENVANTER_SHEET)
    eksik["_key"] = _mac_anahtari(eksik)
    eksik_ah = set(eksik.loc[eksik["AH_Eksik"] == True, "_key"].dropna().astype(str))

    feat["_key"] = _mac_anahtari(feat)

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

    df = feat.merge(market[["_key", "AHh", "AHCh", "B365AHH", "B365AHA",
                             "B365CAHH", "B365CAHA", "B365H", "B365D", "B365A"]],
                    on="_key", how="inner")

    ilk_n = len(df)
    if ilk_n != BEKLENEN_ILK:
        print(f"  [UYARI] İlk birleşik satır sayısı: {ilk_n} (beklenen {BEKLENEN_ILK})")

    df["_eksik_ah"] = df["_key"].isin(eksik_ah)
    df = df[~df["_eksik_ah"]].copy().reset_index(drop=True)
    son_n = len(df)

    print(f"  AH ailesi eksik maç filtresi: {ilk_n} → {son_n} (dışlanan: {ilk_n - son_n})")

    if ilk_n != BEKLENEN_ILK:
        raise RuntimeError(f"BEKLENMEYEN ilk satır sayısı: {ilk_n} != {BEKLENEN_ILK}")
    if son_n != BEKLENEN_SON:
        raise RuntimeError(f"BEKLENMEYEN filtre sonrası satır sayısı: {son_n} != {BEKLENEN_SON}")

    dup = df["_key"].duplicated().sum()
    if dup > 0:
        raise RuntimeError(f"Duplicate key bulundu: {dup}")

    df["GoalDiff"] = df["FTHG"] - df["FTAG"]
    df["Elo_Diff"] = df["Home_Elo_Pre"] - df["Away_Elo_Pre"]
    df["AH_yon"] = df["AHh"].apply(ah_yon)
    df["X12_yon"] = df.apply(lambda r: x12_yon(r["B365H"], r["B365A"]), axis=1)
    df["CrossMarket"] = df.apply(lambda r: cross_market(r["AH_yon"], r["X12_yon"]), axis=1)

    df["AGREEMENT"] = (df["CrossMarket"] == "AGREEMENT").astype(int)
    df["DISAGREEMENT"] = (df["CrossMarket"] == "DISAGREEMENT").astype(int)

    df["Elo_x_AGREEMENT"] = df["Elo_Diff"] * df["AGREEMENT"]
    df["Elo_x_DISAGREEMENT"] = df["Elo_Diff"] * df["DISAGREEMENT"]

    kritik = ["GoalDiff", "Elo_Diff", "AH_yon", "X12_yon", "CrossMarket",
              "AGREEMENT", "DISAGREEMENT", "Elo_x_AGREEMENT", "Elo_x_DISAGREEMENT"]
    df = df.dropna(subset=kritik).reset_index(drop=True)
    final_n = len(df)

    if final_n != BEKLENEN_SON:
        raise RuntimeError(f"NaN temizliği sonrası satır sayısı: {final_n} != {BEKLENEN_SON}")

    return df


# ==================================================================
# 3. MODEL SPESİFİKASYONLARI
# ==================================================================
def model_X(df, model_kod):
    if model_kod == "H4-M0":
        return df[["Elo_Diff"]]
    elif model_kod == "H4-M1":
        return df[["Elo_Diff", "AGREEMENT", "DISAGREEMENT"]]
    elif model_kod == "H4-M2":
        return df[["Elo_Diff", "AGREEMENT", "DISAGREEMENT",
                    "Elo_x_AGREEMENT", "Elo_x_DISAGREEMENT"]]
    else:
        raise ValueError(f"Bilinmeyen model: {model_kod}")


MODEL_META = {
    "H4-M0": "Elo-only (baseline)",
    "H4-M1": "Elo + CrossMarket (AGREEMENT/DISAGREEMENT)",
    "H4-M2": "Elo + CrossMarket + Interaction",
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
        "N": n, "R2": round(r2, 6), "Adj_R2": round(adj_r2, 6),
        "MAE": round(mae, 6), "RMSE": round(rmse, 6),
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


def katsayi_tablosu(model, model_kod, bolme, fit_kaynagi, se_kaynagi):
    params = model.params
    bse = model.bse
    tvals = model.tvalues
    pvals = model.pvalues
    ci = model.conf_int()
    ci.columns = ["CI_lo", "CI_hi"]

    rows = []
    for name in params.index:
        rows.append({
            "Bolme": bolme, "Model": model_kod, "Katsayi": name,
            "Deger": round(float(params[name]), 6),
            "SE": round(float(bse[name]), 6),
            "t": round(float(tvals[name]), 6),
            "p": round(float(pvals[name]), 6),
            "CI_lo": round(float(ci.loc[name, "CI_lo"]), 6),
            "CI_hi": round(float(ci.loc[name, "CI_hi"]), 6),
            "Fit_kaynagi": fit_kaynagi,
            "SE_p_CI_kaynagi": se_kaynagi,
        })
    return rows


# ==================================================================
# 6. GRUP BETİMSEL ANALİZİ
# ==================================================================
def grup_betimsel(df, bolme):
    rows = []
    for grp in ["AGREEMENT", "DISAGREEMENT", "NEUTRAL"]:
        alt = df[df["CrossMarket"] == grp]
        if len(alt) == 0:
            rows.append({"Bolme": bolme, "Grup": grp, "N": 0,
                          "GoalDiff_Ort": None, "GoalDiff_Med": None,
                          "GoalDiff_Std": None, "Elo_Diff_Ort": None})
            continue
        rows.append({
            "Bolme": bolme, "Grup": grp, "N": len(alt),
            "GoalDiff_Ort": round(float(alt["GoalDiff"].mean()), 6),
            "GoalDiff_Med": round(float(alt["GoalDiff"].median()), 6),
            "GoalDiff_Std": round(float(alt["GoalDiff"].std()), 6),
            "Elo_Diff_Ort": round(float(alt["Elo_Diff"].mean()), 4),
        })
    return rows


# ==================================================================
# 7. ANA AKIŞ
# ==================================================================
def main():
    print("=" * 80)
    print("H4 EXPLORE v1.0 — FROZEN (kod düzeltmeli)")
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

    print(f"\n--- CrossMarket dağılımı ---")
    for etiket, d in [("Train", train), ("Validation", val), ("OOS", oos)]:
        sayim = d["CrossMarket"].value_counts()
        print(f"  {etiket:12s}: " + ", ".join(f"{k}={v}" for k, v in sayim.items()))

    # -------- TRAIN: 3 model fit --------
    print("\n--- TRAIN: 3 model fit ediliyor ---")
    train_models = {}
    for kod in ["H4-M0", "H4-M1", "H4-M2"]:
        m = train_fit(train, kod)
        train_models[kod] = m
        print(f"  {kod} ({MODEL_META[kod]}): R²={m.rsquared:.6f}, AdjR²={m.rsquared_adj:.6f}")

    # -------- ÖZET --------
    ozet_rows = []
    katsayi_train_rows = []
    katsayi_val_rows = []
    katsayi_oos_rows = []
    grup_rows = []

    for kod in ["H4-M0", "H4-M1", "H4-M2"]:
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

        katsayi_train_rows.extend(katsayi_tablosu(m, kod, "Train",
            "Train (yeni fit)", "Train fit (SE/p/CI Train'de hesaplandı)"))
        katsayi_val_rows.extend(katsayi_tablosu(m, kod, "Validation",
            "Train (katsayı uygulandı, refit YOK)",
            "Train fit (SE/p/CI Train'den taşındı)"))
        katsayi_oos_rows.extend(katsayi_tablosu(m, kod, "OOS",
            "Train (katsayı uygulandı, refit YOK)",
            "Train fit (SE/p/CI Train'den taşındı)"))

    for etiket, d in [("Train", train), ("Validation", val), ("OOS", oos)]:
        grup_rows.extend(grup_betimsel(d, etiket))

    ozet_df = pd.DataFrame(ozet_rows)

    print("\n--- ÖZET TABLO ---")
    print(ozet_df[["Model", "Train_R2", "Val_R2", "OOS_R2",
                    "Train_MAE", "Val_MAE", "OOS_MAE"]].to_string(index=False))

    # -------- ΔR²_OOS --------
    def r2_oos(kod):
        return float(ozet_df.loc[ozet_df["Model"] == kod, "OOS_R2"].values[0])
    def mae_oos(kod):
        return float(ozet_df.loc[ozet_df["Model"] == kod, "OOS_MAE"].values[0])
    def r2_val(kod):
        return float(ozet_df.loc[ozet_df["Model"] == kod, "Val_R2"].values[0])

    r2_m0 = r2_oos("H4-M0"); r2_m1 = r2_oos("H4-M1"); r2_m2 = r2_oos("H4-M2")
    mae_m0 = mae_oos("H4-M0"); mae_m1 = mae_oos("H4-M1"); mae_m2 = mae_oos("H4-M2")

    delta_m1_m0 = r2_m1 - r2_m0
    delta_m2_m0 = r2_m2 - r2_m0
    delta_m2_m1 = r2_m2 - r2_m1

    print(f"\n--- INCREMENTAL OOS R² ---")
    print(f"  R²_OOS(M0) Elo                     : {r2_m0:.6f}   (MAE: {mae_m0:.6f})")
    print(f"  R²_OOS(M1) Elo+CrossMarket         : {r2_m1:.6f}   (MAE: {mae_m1:.6f})")
    print(f"  R²_OOS(M2) +Interaction            : {r2_m2:.6f}   (MAE: {mae_m2:.6f})")
    print(f"")
    print(f"  ΔR²_OOS(M1 − M0)  CrossMarket katkısı   : {delta_m1_m0:+.6f}   ← ANA KRİTER")
    print(f"  ΔR²_OOS(M2 − M0)  Interaction toplam    : {delta_m2_m0:+.6f}")
    print(f"  ΔR²_OOS(M2 − M1)  Interaction marjinal  : {delta_m2_m1:+.6f}")
    print(f"  ΔMAE(M1 − M0)                            : {mae_m1 - mae_m0:+.6f}")

    # -------- Katsayı özeti --------
    print("\n--- KATSAYILAR (Train) ---")
    for kod in ["H4-M1", "H4-M2"]:
        m = train_models[kod]
        for name in m.params.index:
            print(f"  {kod} {name:25s}: {float(m.params[name]):+.6f}  (p={float(m.pvalues[name]):.4f})")

    # -------- KARAR SINIFI --------
    print("\n--- KARAR SINIFI ---")

    m1 = train_models["H4-M1"]
    m2 = train_models["H4-M2"]

    # K1: ΔR²(M1-M0) >= eşik
    k1 = delta_m1_m0 >= DELTA_R2_ESIK
    # K2: MAE(M1) < MAE(M0)
    k2 = mae_m1 < mae_m0
    # K3: En az bir CrossMarket katsayısı Train'de p < 0.05
    k3 = False
    for c in ["AGREEMENT", "DISAGREEMENT"]:
        if c in m1.params.index and float(m1.pvalues[c]) < 0.05:
            k3 = True
    # K4: M2 dummy katsayıları M1'deki ana etkiyi tersine çevirmiyor
    k4 = True
    for c in ["AGREEMENT", "DISAGREEMENT"]:
        if c in m1.params.index and c in m2.params.index:
            b1 = float(m1.params[c])
            b2 = float(m2.params[c])
            if b1 != 0 and b2 != 0 and np.sign(b1) != np.sign(b2):
                k4 = False
    # K5: Val ve OOS aynı yönde pozitif
    val_delta = r2_val("H4-M1") - r2_val("H4-M0")
    k5 = (val_delta > 0) and (delta_m1_m0 > 0)

    print(f"  Kriter 1: ΔR²(M1-M0) ≥ {DELTA_R2_ESIK}: {delta_m1_m0:.6f} → {'PASS' if k1 else 'FAIL'}")
    print(f"  Kriter 2: MAE(M1) < MAE(M0): {mae_m1:.6f} < {mae_m0:.6f} → {'PASS' if k2 else 'FAIL'}")
    print(f"  Kriter 3: Katsayı anlamlı (p<0.05): {'PASS' if k3 else 'FAIL'}")
    print(f"  Kriter 4: Interaction ana bulguyu tersine çevirmiyor: {'PASS' if k4 else 'FAIL'}")
    print(f"  Kriter 5: Val Δ={val_delta:+.6f}, OOS Δ={delta_m1_m0:+.6f} → {'PASS' if k5 else 'FAIL'}")

    # KARAR AĞACI (protokol önceliği)
    if all([k1, k2, k3, k4, k5]):
        karar = "SUPPORTED"
    elif (delta_m1_m0 <= 0) or (mae_m1 > mae_m0) or (not k4):
        karar = "NOT SUPPORTED"
    else:
        karar = "WEAK / UNCERTAIN"

    print(f"\n  KARAR: {karar}")

    # -------- Kırmızı bayrak --------
    kirmizi = False
    if val_delta > 0 and delta_m1_m0 < 0:
        print("  ⚠ KIRMIZI BAYRAK: Validation pozitif, OOS negatif")
        kirmizi = True
    if val_delta < 0 and delta_m1_m0 > 0:
        print("  ⚠ KIRMIZI BAYRAK: Validation negatif, OOS pozitif")
        kirmizi = True

    # -------- Excel --------
    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        ozet_df.to_excel(w, sheet_name="Ozet_Ana_Tablo", index=False)

        inc_df = pd.DataFrame([{
            "R2_OOS_M0_Elo": round(r2_m0, 6),
            "R2_OOS_M1_Elo_Cross": round(r2_m1, 6),
            "R2_OOS_M2_Interaction": round(r2_m2, 6),
            "Delta_R2_M1_minus_M0": round(delta_m1_m0, 6),
            "Delta_R2_M2_minus_M0": round(delta_m2_m0, 6),
            "Delta_R2_M2_minus_M1": round(delta_m2_m1, 6),
            "MAE_OOS_M0": round(mae_m0, 6),
            "MAE_OOS_M1": round(mae_m1, 6),
            "MAE_OOS_M2": round(mae_m2, 6),
            "Delta_MAE_M1_minus_M0": round(mae_m1 - mae_m0, 6),
            "Val_Delta_R2_M1_M0": round(val_delta, 6),
            "K1_delta_r2": k1, "K2_mae": k2, "K3_katsayi": k3,
            "K4_interaction": k4, "K5_val_oos_yon": k5,
            "Karar": karar, "Kirmizi_bayrak": kirmizi,
        }])
        inc_df.to_excel(w, sheet_name="Incremental_R2", index=False)

        pd.DataFrame(katsayi_train_rows).to_excel(w, sheet_name="Katsayi_Train", index=False)
        pd.DataFrame(katsayi_val_rows).to_excel(w, sheet_name="Katsayi_Validation", index=False)
        pd.DataFrame(katsayi_oos_rows).to_excel(w, sheet_name="Katsayi_OOS", index=False)

        stab_rows = []
        for kod in ["H4-M0", "H4-M1", "H4-M2"]:
            m = train_models[kod]
            for name in m.params.index:
                stab_rows.append({
                    "Model": kod, "Katsayi": name,
                    "Deger_Train": round(float(m.params[name]), 6),
                    "SE_Train": round(float(m.bse[name]), 6),
                    "p_Train": round(float(m.pvalues[name]), 6),
                    "Not": "Validation/OOS'ta refit YOK.",
                })
        pd.DataFrame(stab_rows).to_excel(w, sheet_name="Katsayi_Stabilite", index=False)

        pd.DataFrame([{
            "Not": "Validation ve OOS katsayıları Train'de fit edilmiştir; refit YOK.",
            "Referans_sinif": "NEUTRAL",
            "Delta_R2_esik": DELTA_R2_ESIK,
            "K1_delta_r2": k1, "K2_mae": k2, "K3_katsayi": k3,
            "K4_interaction": k4, "K5_val_oos_yon": k5,
            "Karar": karar,
        }]).to_excel(w, sheet_name="Katsayi_Anlam", index=False)

        pd.DataFrame(grup_rows).to_excel(w, sheet_name="Grup_Betimsel", index=False)

        pd.DataFrame([{
            "Protokol": "H4 FROZEN PROTOCOL v1.0",
            "Tarih": "2026-10-08",
            "Outcome": "GoalDiff = FTHG - FTAG",
            "CrossMarket": "AGREEMENT / DISAGREEMENT / NEUTRAL (ref)",
            "Modeller": "H4-M0, H4-M1, H4-M2",
            "Ana_kriter": f"Delta R2_OOS(M1-M0) >= {DELTA_R2_ESIK} + MAE + katsayi + interaction + val/oos yon",
            "Train": "2019/20 - 2022/23",
            "Validation": "2023/24",
            "OOS": "2024/25",
            "2026_27": "KILITLI",
            "Karar": karar,
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} başarıyla yazıldı.")

    # -------- Kanıt --------
    print("\n" + "=" * 80)
    print("VERİ BÜTÜNLÜĞÜ VE KANIT")
    print("=" * 80)
    print(f"  Toplam satır: {len(df)}")
    print(f"  Train N: {len(train)}")
    print(f"  Validation N: {len(val)}")
    print(f"  OOS N: {len(oos)}")
    print(f"  CrossMarket dağılımı (tüm):")
    for k, v in df["CrossMarket"].value_counts().items():
        print(f"    {k}: {v}")

    print("\n--- Dosya okuma sayacı ---")
    for ad in sorted(_okuma_sayaci.keys()):
        print(f"  {ad}: READ {_okuma_sayaci[ad]} TIMES")
    for ad in YASAKLI_DOSYALAR:
        print(f"  [YASAKLI] {ad}: READ {_okuma_sayaci.get(ad, 0)} TIMES")


if __name__ == "__main__":
    main()