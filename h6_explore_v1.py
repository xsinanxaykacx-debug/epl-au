# -*- coding: utf-8 -*-
"""
h6_explore_v1.py
H6 FROZEN PROTOCOL v1.0 — birebir implementasyon
Tarih: 2026-10-08

Kapsam:
- Train:       2019/20 + 2020/21 + 2021/22 + 2022/23
- Validation:  2023/24
- Blind OOS:   2024/25 (tek seferlik)

KİLİTLER:
- E0 (10).csv ASLA OKUNMAZ — sayaç ile kanıtlanır.
- Yalnızca E0 (4) – E0 (9) okunur.
- h1_features.csv ile deterministik merge (Season|Date|HomeTeam|AwayTeam).
- AH ailesi filtresi: AH_Eksik == True dışlanır → 2280 → 2278.
- ZERO filtresi GEREKMEZ (inventory'de sorun yok).
- Train'de fit edilen katsayılar Validation/OOS'ta YENİDEN FIT EDİLMEZ.
- 3 modelin tamamı OOS'a girer.
- Post-hoc model seçimi, ek feature, threshold YOK.

MODELLER:
- H6-M0: GoalDiff ~ Elo_Diff
- H6-M1: GoalDiff ~ Elo_Diff + AH_BookmakerSpread
- H6-M2: GoalDiff ~ Elo_Diff + AH_BookmakerSpread + Elo×Spread

ANA DEĞİŞKEN:
- GoalDiff = FTHG - FTAG
- AH_BookmakerSpread = MaxAHH - AvgAHH

ANA KRİTER:
- ΔR²_OOS(M1 - M0) >= 0.005

RAPORLAMA NOTU (bu sürümde eklenmiştir):
- Kırmızı bayrak: "Katsayı yönü Validation ↔ OOS karşılaştırması"
  TEST EDİLEMEZ — refit yok. Bu nedenle bu kriter uygulanmıyor.
- Kalan kırmızı bayrak kriterleri (Val/OOS ΔR² işaret uyumsuzluğu)
  aktiftir ve doğru şekilde çalışmaktadır.

ÇIKTI: h6_explore_v1_results.xlsx
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
CIKTI = "h6_explore_v1_results.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

MARKET_DOSYALARI = {
    "2019/20": "E0 (4).csv",
    "2020/21": "E0 (5).csv",
    "2021/22": "E0 (6).csv",
    "2022/23": "E0 (7).csv",
    "2023/24": "E0 (8).csv",
    "2024/25": "E0 (9).csv",
}
MARKET_SUTUNLARI = ["Season", "Date", "HomeTeam", "AwayTeam", "MaxAHH", "AvgAHH"]

TRAIN_SEZONLAR = ["2019/20", "2020/21", "2021/22", "2022/23"]
VALIDATION_SEZON = "2023/24"
OOS_SEZON = "2024/25"

DELTA_R2_ESIK = 0.005

BEKLENEN_TOP = 2278
BEKLENEN_TRAIN = 1519
BEKLENEN_VAL = 379
BEKLENEN_OOS = 380

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


# ==================================================================
# 2. VERİ YÜKLEME
# ==================================================================
def veri_yukle():
    if not os.path.exists(H1_FEATURES):
        raise RuntimeError(f"{H1_FEATURES} bulunamadı.")
    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")

    for c in ["FTHG", "FTAG"]:
        if c not in feat.columns:
            raise RuntimeError(f"{H1_FEATURES} içinde {c} yok.")

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

    df = feat.merge(market[["_key", "MaxAHH", "AvgAHH"]], on="_key", how="inner")

    print(f"\n  Merge öncesi birleşik satır: {len(df)}")

    dup = df["_key"].duplicated().sum()
    if dup > 0:
        raise RuntimeError(f"Duplicate key bulundu: {dup}")

    df["_eksik_ah"] = df["_key"].isin(eksik_ah)
    onceki = len(df)
    df = df[~df["_eksik_ah"]].copy().reset_index(drop=True)
    sonraki = len(df)
    print(f"  AH ailesi eksik filtresi: {onceki} → {sonraki} "
          f"(dışlanan: {onceki - sonraki})")

    df["GoalDiff"] = df["FTHG"] - df["FTAG"]
    df["Elo_Diff"] = df["Home_Elo_Pre"] - df["Away_Elo_Pre"]
    df["AH_BookmakerSpread"] = df["MaxAHH"] - df["AvgAHH"]
    df["Elo_x_Spread"] = df["Elo_Diff"] * df["AH_BookmakerSpread"]

    kritik = ["GoalDiff", "Elo_Diff", "AH_BookmakerSpread", "Elo_x_Spread"]
    df = df.dropna(subset=kritik).reset_index(drop=True)
    print(f"  NaN temizliği sonrası satır: {len(df)}")

    return df


# ==================================================================
# 3. MODEL SPESİFİKASYONLARI
# ==================================================================
def model_X(df, model_kod):
    if model_kod == "H6-M0":
        return df[["Elo_Diff"]]
    elif model_kod == "H6-M1":
        return df[["Elo_Diff", "AH_BookmakerSpread"]]
    elif model_kod == "H6-M2":
        return df[["Elo_Diff", "AH_BookmakerSpread", "Elo_x_Spread"]]
    else:
        raise ValueError(f"Bilinmeyen model: {model_kod}")


MODEL_META = {
    "H6-M0": "Elo-only (baseline)",
    "H6-M1": "Elo + AH_BookmakerSpread (ana)",
    "H6-M2": "Elo + AH_BookmakerSpread + Interaction",
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
# 6. ANA AKIŞ
# ==================================================================
def main():
    print("=" * 80)
    print("H6 EXPLORE v1.0 — FROZEN")
    print("E0 (10).csv KESİNLİKLE OKUNMAYACAK.")
    print("Train katsayıları Validation/OOS'ta YENİDEN FIT EDİLMEZ.")
    print("=" * 80)

    print("\n--- Veri yükleniyor ---")
    df = veri_yukle()
    print(f"  Toplam satır (tüm filtreler sonrası): {len(df)}")
    print(f"  Sezonlar: {sorted(df['Season'].unique())}")

    print("\n--- SANITY CHECK ---")

    if len(df) != BEKLENEN_TOP:
        raise RuntimeError(f"Toplam satır beklenenden farklı: {len(df)} != {BEKLENEN_TOP}")
    print(f"  Toplam satır = {BEKLENEN_TOP} ✓")

    train = df[df["Season"].isin(TRAIN_SEZONLAR)].copy().reset_index(drop=True)
    val   = df[df["Season"] == VALIDATION_SEZON].copy().reset_index(drop=True)
    oos   = df[df["Season"] == OOS_SEZON].copy().reset_index(drop=True)

    if len(train) != BEKLENEN_TRAIN:
        raise RuntimeError(f"Train N beklenenden farklı: {len(train)} != {BEKLENEN_TRAIN}")
    print(f"  Train N = {BEKLENEN_TRAIN} ✓")

    if len(val) != BEKLENEN_VAL:
        raise RuntimeError(f"Validation N beklenenden farklı: {len(val)} != {BEKLENEN_VAL}")
    print(f"  Validation N = {BEKLENEN_VAL} ✓")

    if len(oos) != BEKLENEN_OOS:
        raise RuntimeError(f"OOS N beklenenden farklı: {len(oos)} != {BEKLENEN_OOS}")
    print(f"  OOS N = {BEKLENEN_OOS} ✓")

    print(f"  → TÜM SANITY CHECK'LER PASS")

    print("\n--- TRAIN: 3 model fit ediliyor (yalnızca bu bölmede) ---")
    train_models = {}
    for kod in ["H6-M0", "H6-M1", "H6-M2"]:
        m = train_fit(train, kod)
        train_models[kod] = m
        print(f"  {kod} ({MODEL_META[kod]}): R²={m.rsquared:.6f}, AdjR²={m.rsquared_adj:.6f}")

    ozet_rows = []
    katsayi_train_rows = []
    katsayi_val_rows = []
    katsayi_oos_rows = []

    for kod in ["H6-M0", "H6-M1", "H6-M2"]:
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

    ozet_df = pd.DataFrame(ozet_rows)

    print("\n--- ÖZET TABLO ---")
    print(ozet_df[["Model", "Train_R2", "Val_R2", "OOS_R2",
                    "Train_MAE", "Val_MAE", "OOS_MAE"]].to_string(index=False))

    def r2_oos(kod):
        return float(ozet_df.loc[ozet_df["Model"] == kod, "OOS_R2"].values[0])
    def mae_oos(kod):
        return float(ozet_df.loc[ozet_df["Model"] == kod, "OOS_MAE"].values[0])
    def r2_val(kod):
        return float(ozet_df.loc[ozet_df["Model"] == kod, "Val_R2"].values[0])

    r2_m0 = r2_oos("H6-M0"); r2_m1 = r2_oos("H6-M1"); r2_m2 = r2_oos("H6-M2")
    mae_m0 = mae_oos("H6-M0"); mae_m1 = mae_oos("H6-M1"); mae_m2 = mae_oos("H6-M2")

    delta_m1_m0 = r2_m1 - r2_m0
    delta_m2_m0 = r2_m2 - r2_m0
    delta_m2_m1 = r2_m2 - r2_m1
    val_delta = r2_val("H6-M1") - r2_val("H6-M0")

    print(f"\n--- INCREMENTAL OOS R² ---")
    print(f"  R²_OOS(M0) Elo                     : {r2_m0:.6f}   (MAE: {mae_m0:.6f})")
    print(f"  R²_OOS(M1) Elo+Spread              : {r2_m1:.6f}   (MAE: {mae_m1:.6f})")
    print(f"  R²_OOS(M2) +Interaction            : {r2_m2:.6f}   (MAE: {mae_m2:.6f})")
    print(f"")
    print(f"  ΔR²_OOS(M1 − M0)  Spread katkısı        : {delta_m1_m0:+.6f}   ← ANA KRİTER")
    print(f"  ΔR²_OOS(M2 − M0)  Interaction toplam    : {delta_m2_m0:+.6f}")
    print(f"  ΔR²_OOS(M2 − M1)  Interaction marjinal  : {delta_m2_m1:+.6f}")
    print(f"  ΔMAE(M1 − M0)                            : {mae_m1 - mae_m0:+.6f}")
    print(f"  Val ΔR²(M1 − M0)                         : {val_delta:+.6f}")

    print("\n--- KATSAYILAR (Train) ---")
    for kod in ["H6-M1", "H6-M2"]:
        m = train_models[kod]
        for name in m.params.index:
            print(f"  {kod} {name:25s}: {float(m.params[name]):+.6f}  "
                  f"(p={float(m.pvalues[name]):.4f})")

    print("\n--- KARAR SINIFI ---")

    m1 = train_models["H6-M1"]
    m2 = train_models["H6-M2"]

    k1 = delta_m1_m0 >= DELTA_R2_ESIK
    k2 = mae_m1 < mae_m0
    k3 = False
    if "AH_BookmakerSpread" in m1.params.index:
        if float(m1.pvalues["AH_BookmakerSpread"]) < 0.05:
            k3 = True
    k4 = True
    if "AH_BookmakerSpread" in m1.params.index and "AH_BookmakerSpread" in m2.params.index:
        b1 = float(m1.params["AH_BookmakerSpread"])
        b2 = float(m2.params["AH_BookmakerSpread"])
        if b1 != 0 and b2 != 0 and np.sign(b1) != np.sign(b2):
            k4 = False
    k5 = (val_delta > 0) and (delta_m1_m0 > 0)

    print(f"  Kriter 1: ΔR²(M1-M0) ≥ {DELTA_R2_ESIK}: {delta_m1_m0:.6f} → {'PASS' if k1 else 'FAIL'}")
    print(f"  Kriter 2: MAE(M1) < MAE(M0): {mae_m1:.6f} < {mae_m0:.6f} → {'PASS' if k2 else 'FAIL'}")
    print(f"  Kriter 3: AH_BookmakerSpread p<0.05 (Train): {'PASS' if k3 else 'FAIL'}")
    print(f"  Kriter 4: Interaction ana bulguyu tersine çevirmiyor: {'PASS' if k4 else 'FAIL'}")
    print(f"  Kriter 5: Val Δ={val_delta:+.6f}, OOS Δ={delta_m1_m0:+.6f} → {'PASS' if k5 else 'FAIL'}")

    if all([k1, k2, k3, k4, k5]):
        karar = "SUPPORTED"
    elif (delta_m1_m0 <= 0) or (mae_m1 > mae_m0) or (not k4):
        karar = "NOT SUPPORTED"
    else:
        karar = "WEAK / UNCERTAIN"

    print(f"\n  KARAR: {karar}")

    # ================================================================
    # KIRMIZI BAYRAK
    # ================================================================
    print("\n--- KIRMIZI BAYRAK KONTROLÜ ---")
    print("  Not: 'Katsayı yönü Validation ↔ OOS karşılaştırması' TEST EDİLEMEZ")
    print("       (refit yok — Validation/OOS'ta yeni katsayı yok).")
    print("  Kalan kırmızı bayrak kriterleri (Val/OOS ΔR² işaret uyumsuzluğu) aktif:")

    kirmizi = False
    if val_delta > 0 and delta_m1_m0 < 0:
        print("  ⚠ KIRMIZI BAYRAK: Validation pozitif, OOS negatif")
        kirmizi = True
    if val_delta < 0 and delta_m1_m0 > 0:
        print("  ⚠ KIRMIZI BAYRAK: Validation negatif, OOS pozitif")
        kirmizi = True
    if not kirmizi:
        print("  ✓ Val/OOS ΔR² işaret uyumsuzluğu YOK")

    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        ozet_df.to_excel(w, sheet_name="Ozet_Ana_Tablo", index=False)

        inc_df = pd.DataFrame([{
            "R2_OOS_M0_Elo": round(r2_m0, 6),
            "R2_OOS_M1_Elo_Spread": round(r2_m1, 6),
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
        for kod in ["H6-M0", "H6-M1", "H6-M2"]:
            m = train_models[kod]
            for name in m.params.index:
                stab_rows.append({
                    "Model": kod, "Katsayi": name,
                    "Deger_Train": round(float(m.params[name]), 6),
                    "SE_Train": round(float(m.bse[name]), 6),
                    "p_Train": round(float(m.pvalues[name]), 6),
                    "Not": "Validation/OOS'ta refit YOK; katsayı ve SE/p/CI Train'den.",
                })
        pd.DataFrame(stab_rows).to_excel(w, sheet_name="Katsayi_Stabilite", index=False)

        pd.DataFrame([{
            "Not": "Validation ve OOS katsayıları Train'de fit edilmiştir; refit YOK. "
                   "Kırmızı bayrak: 'Katsayı yönü Val↔OOS' TEST EDİLEMEZ (refit yok). "
                   "Aktif kırmızı bayrak: Val/OOS ΔR² işaret uyumsuzluğu.",
            "Delta_R2_esik": DELTA_R2_ESIK,
            "Zero_kurali": "GEREKMEZ (inventory'de sorun yok)",
            "Beklenen_top": BEKLENEN_TOP,
            "Beklenen_train": BEKLENEN_TRAIN,
            "Beklenen_val": BEKLENEN_VAL,
            "Beklenen_oos": BEKLENEN_OOS,
            "K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5,
            "Karar": karar,
            "Kirmizi_bayrak_test_edilemez": "Katsayi yonu Val↔OOS (refit yok)",
            "Kirmizi_bayrak_aktif": "Val/OOS ΔR² işaret uyumsuzluğu",
        }]).to_excel(w, sheet_name="Katsayi_Anlam", index=False)

        pd.DataFrame([{
            "Protokol": "H6 FROZEN PROTOCOL v1.0",
            "Tarih": "2026-10-08",
            "Outcome": "GoalDiff = FTHG - FTAG",
            "Ana_degisken": "AH_BookmakerSpread = MaxAHH - AvgAHH",
            "Threshold": "YOK",
            "Modeller": "H6-M0 (Elo), H6-M1 (+Spread), H6-M2 (+Interaction)",
            "Ana_kriter": f"Delta R2_OOS(M1-M0) >= {DELTA_R2_ESIK} + MAE + katsayi p<0.05 + interaction + Val/OOS yon",
            "Train": "2019/20 - 2022/23",
            "Validation": "2023/24",
            "OOS": "2024/25",
            "2026_27": "KILITLI - okunmadi",
            "Train_refit": "YOK",
            "Karar": karar,
            "Kirmizi_bayrak": kirmizi,
            "Raporlama_notu": "Katsayi yonu Val↔OOS karsilastirmasi TEST EDILEMEZ (refit yok)",
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} başarıyla yazıldı.")

    print("\n" + "=" * 80)
    print("VERİ BÜTÜNLÜĞÜ VE KANIT")
    print("=" * 80)
    print(f"  Toplam satır (filtreler sonrası): {len(df)}")
    print(f"  Train N: {len(train)}")
    print(f"  Validation N: {len(val)}")
    print(f"  OOS N: {len(oos)}")
    print(f"  NaN kontrolü:")
    for c in ["GoalDiff", "Elo_Diff", "AH_BookmakerSpread", "Elo_x_Spread"]:
        print(f"    {c}: NaN={df[c].isna().sum()}")

    print("\n--- Dosya okuma sayacı ---")
    for ad in sorted(_okuma_sayaci.keys()):
        print(f"  {ad}: READ {_okuma_sayaci[ad]} TIMES")
    for ad in YASAKLI_DOSYALAR:
        print(f"  [YASAKLI] {ad}: READ {_okuma_sayaci.get(ad, 0)} TIMES")


if __name__ == "__main__":
    main()