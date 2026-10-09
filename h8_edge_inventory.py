# -*- coding: utf-8 -*-
"""
h8_edge_inventory.py
H8-EDGE v1.0 Inventory — Veri Uygunluğu ve Sinyal Yoğunluğu

AMAÇ:
- B365H, FTR, Elo_Diff sütunlarının varlığını, tutarlılığını kontrol et.
- FTR ∈ {H,D,A} kontrolü.
- B365H > 0 kontrolü.
- Sinyal yoğunluğunu (Elo_Diff > 100 AND B365H < 2.50) BİLGİ AMAÇLI raporla.

YAPMAZ:
- ROI hesaplamaz
- Bootstrap yapmaz
- Permutation yapmaz
- MDD hesaplamaz
- Sinyal performansına göre karar vermez
- Eşik değiştirmez
- 2026/27'yi okumaz

DÜZELTMELER (bu sürüm):
1. karar_matrisi() artık nan değerlerini döndürüyor.
2. Market CSV'leri tek seferde okunuyor.
3. FTR market'ten alınmıyor; h1_features.csv'de zaten var.
   Bu, merge sırasındaki FTR_x / FTR_y çakışmasını çözer.
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

# ==================================================================
# 0. SABİTLER
# ==================================================================
H1_FEATURES = "h1_features.csv"
ENVANTER = "envanter_v2.xlsx"
ENVANTER_SHEET = "Eksik_Maclar"
CIKTI = "h8_edge_inventory.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

MARKET_DOSYALARI = {
    "2019/20": "E0 (4).csv",
    "2020/21": "E0 (5).csv",
    "2021/22": "E0 (6).csv",
    "2022/23": "E0 (7).csv",
    "2023/24": "E0 (8).csv",
    "2024/25": "E0 (9).csv",
}

# DÜZELTME 3: FTR kaldırıldı
MARKET_SUTUNLAR = ["Season", "Date", "HomeTeam", "AwayTeam", "B365H"]

ELO_ESIK = 100
FIYAT_ESIK = 2.50

GECERLI_FTR = {"H", "D", "A"}

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
# 2. MARKET VERİLERİNİ YÜKLE
# ==================================================================
def market_verilerini_yukle():
    print("\n--- Market CSV'leri yükleniyor (tek seferde) ---")
    frames = []
    per_sezon_meta = {}
    for sezon, dosya in MARKET_DOSYALARI.items():
        ad = os.path.basename(dosya)
        if ad in YASAKLI_DOSYALAR:
            raise RuntimeError(f"YASAKLI DOSYA: {ad}")
        df = _guvenli_oku(dosya)
        df["Season"] = sezon
        cols = set(df.columns)
        per_sezon_meta[sezon] = {
            "Dosya": dosya,
            "N_satir": len(df),
            "B365H": "B365H" in cols,
            "FTR": "FTR" in cols,
        }
        eksik = [s for s in MARKET_SUTUNLAR if s not in df.columns]
        if eksik:
            raise RuntimeError(f"{ad} içinde eksik sütunlar: {eksik}")
        frames.append(df[MARKET_SUTUNLAR].copy())
    market = pd.concat(frames, ignore_index=True)
    return market, per_sezon_meta


# ==================================================================
# 3. SÜTUN VARLIK KONTROLÜ
# ==================================================================
def sutun_varlik(per_sezon_meta):
    print("\n" + "=" * 80)
    print("1. SÜTUN VARLIK KONTROLÜ")
    print("=" * 80)

    print("\n  --- Market CSV'leri: B365H, FTR ---")
    for sezon, meta in per_sezon_meta.items():
        print(f"  {meta['Dosya']:15s} → {sezon}  ({meta['N_satir']} satır)")
        print(f"    B365H: {'VAR' if meta['B365H'] else 'YOK'}")
        print(f"    FTR  : {'VAR' if meta['FTR'] else 'YOK'}")

    print("\n  --- h1_features.csv: FTR + Elo sütunları ---")
    if os.path.exists(H1_FEATURES):
        feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig", nrows=5)
        cols = set(feat.columns)
        feat_ftr_ok = "FTR" in cols
        feat_elo_ok = ("Home_Elo_Pre" in cols) and ("Away_Elo_Pre" in cols)
        print(f"    FTR           : {'VAR' if feat_ftr_ok else 'YOK'}")
        print(f"    Home_Elo_Pre  : {'VAR' if 'Home_Elo_Pre' in cols else 'YOK'}")
        print(f"    Away_Elo_Pre  : {'VAR' if 'Away_Elo_Pre' in cols else 'YOK'}")
    else:
        feat_ftr_ok = False
        feat_elo_ok = False
        print(f"    [YOK] {H1_FEATURES}")

    tum_b365h = all(per_sezon_meta[s]["B365H"] for s in per_sezon_meta)
    tum_ftr_market = all(per_sezon_meta[s]["FTR"] for s in per_sezon_meta)

    print(f"\n  --- KRİTİK KONTROL ---")
    print(f"    B365H tüm sezonlarda       : {'✅' if tum_b365h else '❌'}")
    print(f"    FTR tüm market CSV'lerinde : {'✅' if tum_ftr_market else '❌'}")
    print(f"    FTR h1_features.csv'de     : {'✅' if feat_ftr_ok else '❌'}")
    print(f"    Elo sütunları h1_features'te: {'✅' if feat_elo_ok else '❌'}")

    return tum_b365h, tum_ftr_market, feat_ftr_ok, feat_elo_ok


# ==================================================================
# 4. VERİ YÜKLEME VE FİLTRELEME
# ==================================================================
def veri_yukle(market):
    print("\n" + "=" * 80)
    print("2. VERİ YÜKLEME VE FİLTRELEME")
    print("=" * 80)

    print(f"\n  Market satırı (ham): {len(market)}")

    if not os.path.exists(H1_FEATURES):
        raise RuntimeError(f"{H1_FEATURES} bulunamadı.")
    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")

    # FTR ve Elo'nun feat'te olduğunu doğrula
    for c in ["FTR", "Home_Elo_Pre", "Away_Elo_Pre"]:
        if c not in feat.columns:
            raise RuntimeError(f"{H1_FEATURES} içinde {c} yok.")

    feat["_key"] = _mac_anahtari(feat)
    market["_key"] = _mac_anahtari(market)

    # DÜZELTME 3: FTR market'ten alınmıyor
    df = feat.merge(market[["_key", "B365H"]], on="_key", how="inner")
    print(f"  Merge sonrası: {len(df)}")

    dup = df["_key"].duplicated().sum()
    print(f"  Duplicate key: {dup}")
    if dup > 0:
        raise RuntimeError(f"Duplicate bulundu: {dup}")

    eksik = pd.read_excel(ENVANTER, sheet_name=ENVANTER_SHEET)
    eksik["_key"] = _mac_anahtari(eksik)
    eksik_ah = set(eksik.loc[eksik["AH_Eksik"] == True, "_key"].dropna().astype(str))

    df["_eksik_ah"] = df["_key"].isin(eksik_ah)
    onceki = len(df)
    df = df[~df["_eksik_ah"]].copy().reset_index(drop=True)
    sonraki = len(df)
    print(f"  AH ailesi eksik filtresi: {onceki} → {sonraki} (dışlanan: {onceki - sonraki})")

    return df


# ==================================================================
# 5. DEĞİŞKEN KONTROLLERİ
# ==================================================================
def degisken_kontrol(df):
    print("\n" + "=" * 80)
    print("3. DEĞİŞKEN KONTROLLERİ")
    print("=" * 80)

    print("\n  --- NaN kontrolü ---")
    for c in ["B365H", "FTR", "Home_Elo_Pre", "Away_Elo_Pre"]:
        nn = int(df[c].isna().sum())
        print(f"    {c:15s}: NaN={nn} / {len(df)}")

    print("\n  --- B365H sıfır/negatif kontrolü ---")
    sifir = int((df["B365H"] == 0).sum())
    negatif = int((df["B365H"] < 0).sum())
    print(f"    sıfır={sifir}, negatif={negatif}")

    print("\n  --- FTR değer kontrolü ---")
    ftr_degerler = df["FTR"].dropna().unique()
    gecersiz = [v for v in ftr_degerler if v not in GECERLI_FTR]
    print(f"    Geçerli değerler: {GECERLI_FTR}")
    print(f"    Veride bulunanlar: {sorted(ftr_degerler)}")
    print(f"    Geçersiz değerler: {gecersiz if gecersiz else 'YOK'}")

    print("\n  --- FTR dağılımı ---")
    ftr_sayim = df["FTR"].value_counts()
    for k, v in ftr_sayim.items():
        print(f"    {k}: {v} (%{v/len(df)*100:.2f})")

    return sifir, negatif, gecersiz


# ==================================================================
# 6. ELO_DIFF VE SİNYAL YOĞUNLUĞU
# ==================================================================
def sinyal_yogunluk(df):
    print("\n" + "=" * 80)
    print("4. ELO_DIFF VE SİNYAL YOĞUNLUĞU (bilgi amaçlı)")
    print("=" * 80)

    df["Elo_Diff"] = df["Home_Elo_Pre"] - df["Away_Elo_Pre"]

    print("\n  --- Elo_Diff dağılımı ---")
    print(f"    NaN: {df['Elo_Diff'].isna().sum()}")
    s = df["Elo_Diff"].dropna()
    print(f"    N     : {len(s)}")
    print(f"    Min   : {s.min():.4f}")
    print(f"    P25   : {s.quantile(0.25):.4f}")
    print(f"    P50   : {s.quantile(0.50):.4f}")
    print(f"    P75   : {s.quantile(0.75):.4f}")
    print(f"    Max   : {s.max():.4f}")
    print(f"    Mean  : {s.mean():.4f}")
    print(f"    Std   : {s.std():.4f}")

    df["signal"] = (df["Elo_Diff"] > ELO_ESIK) & (df["B365H"] < FIYAT_ESIK)

    print(f"\n  --- Sinyal: Elo_Diff > {ELO_ESIK} AND B365H < {FIYAT_ESIK} ---")
    print(f"  Bu bilgi amaçlıdır. Karar mekanizmasına GİRMEZ.")
    print(f"  Performans (ROI, p, MDD) HESAPLANMAMIŞTIR.")

    toplam_sinyal = int(df["signal"].sum())
    print(f"\n  Toplam: {toplam_sinyal} / {len(df)} (%{toplam_sinyal/len(df)*100:.2f})")

    print(f"\n  --- Sezon bazında sinyal sayısı ---")
    for sezon in sorted(df["Season"].unique()):
        alt = df[df["Season"] == sezon]
        n = len(alt)
        s = int(alt["signal"].sum())
        print(f"    {sezon}: {s} / {n} (%{s/n*100:.2f})")

    print(f"\n  --- Bölme bazında sinyal sayısı ---")
    for etiket, sezonlar in [
        ("Train", ["2019/20", "2020/21", "2021/22", "2022/23"]),
        ("Validation", ["2023/24"]),
        ("OOS", ["2024/25"]),
    ]:
        alt = df[df["Season"].isin(sezonlar)]
        n = len(alt)
        s = int(alt["signal"].sum())
        if n > 0:
            print(f"    {etiket:12s}: {s} / {n} (%{s/n*100:.2f})")
        else:
            print(f"    {etiket:12s}: 0")

    if toplam_sinyal < 100:
        print(f"\n  ⚠ BİLGİ: Toplam sinyal {toplam_sinyal} < 100.")
        print(f"     Bu bir FAIL sebebi DEĞİLDİR (protokol).")
    else:
        print(f"\n  ℹ Toplam sinyal {toplam_sinyal} ≥ 100.")

    return df, toplam_sinyal


# ==================================================================
# 7. VERİ BÜTÜNLÜĞÜ VE KANIT
# ==================================================================
def veri_butunlugu(df):
    print("\n" + "=" * 80)
    print("5. VERİ BÜTÜNLÜĞÜ VE KANIT")
    print("=" * 80)

    print(f"\n  Toplam satır (AH filtresi sonrası): {len(df)}")
    print(f"\n  Sezon dağılımı:")
    for s in sorted(df["Season"].unique()):
        n = len(df[df["Season"] == s])
        print(f"    {s}: {n}")

    print("\n--- Dosya okuma sayacı ---")
    for ad in sorted(_okuma_sayaci.keys()):
        print(f"  {ad}: READ {_okuma_sayaci[ad]} TIMES")
    for ad in YASAKLI_DOSYALAR:
        print(f"  [YASAKLI] {ad}: READ {_okuma_sayaci.get(ad, 0)} TIMES")


# ==================================================================
# 8. KARAR MATRİSİ
# ==================================================================
def karar_matrisi(df, tum_b365h, feat_ftr_ok, feat_elo_ok, gecersiz_ftr):
    print("\n" + "=" * 80)
    print("6. KARAR MATRİSİ (H8-EDGE FROZEN PROTOKOL)")
    print("=" * 80)

    print("\n  Kontroller:")
    print(f"    B365H tüm sezonlarda     : {'✅' if tum_b365h else '❌'}")
    print(f"    FTR h1_features'te       : {'✅' if feat_ftr_ok else '❌'}")
    print(f"    Elo sütunları mevcut     : {'✅' if feat_elo_ok else '❌'}")

    nan_b365h = int(df["B365H"].isna().sum())
    nan_ftr = int(df["FTR"].isna().sum())
    nan_elo = int((df["Home_Elo_Pre"].isna() | df["Away_Elo_Pre"].isna()).sum())
    nan_ok = (nan_b365h == 0) and (nan_ftr == 0) and (nan_elo == 0)
    print(f"    Kritik NaN yok           : {'✅' if nan_ok else '❌'} "
          f"(B365H={nan_b365h}, FTR={nan_ftr}, Elo={nan_elo})")

    ftr_ok = len(gecersiz_ftr) == 0
    print(f"    FTR yalnızca H/D/A       : {'✅' if ftr_ok else '❌'}")

    sifir = int((df["B365H"] == 0).sum())
    neg = int((df["B365H"] < 0).sum())
    fiyat_ok = (sifir == 0) and (neg == 0)
    print(f"    B365H > 0                : {'✅' if fiyat_ok else '❌'}")

    if tum_b365h and feat_ftr_ok and feat_elo_ok and nan_ok and ftr_ok and fiyat_ok:
        karar = "PASS → Explore"
        print(f"\n  ✅ KARAR: {karar}")
    else:
        karar = "FAIL → H8-EDGE KAPANIR"
        print(f"\n  ❌ KARAR: {karar}")
        print(f"     Protokol DEĞİŞMEZ.")
        print(f"     Eşik değiştirilmez.")

    return karar, nan_b365h, nan_ftr, nan_elo


# ==================================================================
# ANA
# ==================================================================
def main():
    print("=" * 80)
    print("H8-EDGE INVENTORY — Veri Uygunluğu ve Sinyal Yoğunluğu")
    print("2026/27 (E0 (10).csv) KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    # 1. Market verilerini TEK SEFERDE yükle
    market, per_sezon_meta = market_verilerini_yukle()

    # 2. Sütun varlık
    tum_b365h, tum_ftr_market, feat_ftr_ok, feat_elo_ok = sutun_varlik(per_sezon_meta)

    # 3. Veri yükleme + filtre
    df = veri_yukle(market)

    # 4. Değişken kontrolleri
    sifir, negatif, gecersiz_ftr = degisken_kontrol(df)

    # 5. Sinyal yoğunluğu
    df, toplam_sinyal = sinyal_yogunluk(df)

    # 6. Bütünlük
    veri_butunlugu(df)

    # 7. Karar matrisi
    karar, nan_b365h, nan_ftr, nan_elo = karar_matrisi(
        df, tum_b365h, feat_ftr_ok, feat_elo_ok, gecersiz_ftr
    )

    # -------- EXCEL --------
    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:

        # Sutun_Varlik — per_sezon_meta'dan
        rows = []
        for sezon, meta in per_sezon_meta.items():
            rows.append({
                "Sezon": sezon,
                "Dosya": meta["Dosya"],
                "N_satir": meta["N_satir"],
                "B365H": "VAR" if meta["B365H"] else "YOK",
                "FTR": "VAR" if meta["FTR"] else "YOK",
            })
        pd.DataFrame(rows).to_excel(w, sheet_name="Sutun_Varlik", index=False)

        # FTR dağılımı
        ftr_rows = []
        for sezon in sorted(df["Season"].unique()):
            alt = df[df["Season"] == sezon]
            sayim = alt["FTR"].value_counts()
            ftr_rows.append({
                "Sezon": sezon,
                "H": int(sayim.get("H", 0)),
                "D": int(sayim.get("D", 0)),
                "A": int(sayim.get("A", 0)),
                "Toplam": len(alt),
            })
        pd.DataFrame(ftr_rows).to_excel(w, sheet_name="FTR_Dagilimi", index=False)

        # Sinyal yoğunluğu
        sinyal_rows = []
        for sezon in sorted(df["Season"].unique()):
            alt = df[df["Season"] == sezon]
            sinyal_rows.append({
                "Sezon": sezon,
                "N_toplam": len(alt),
                "N_sinyal": int(alt["signal"].sum()),
                "Sinyal_oran_%": round(alt["signal"].sum() / len(alt) * 100, 4) if len(alt) > 0 else 0,
            })
        pd.DataFrame(sinyal_rows).to_excel(w, sheet_name="Sinyal_Sezon", index=False)

        # Bölme bazında sinyal
        bolme_rows = []
        for etiket, sezonlar in [
            ("Train", ["2019/20", "2020/21", "2021/22", "2022/23"]),
            ("Validation", ["2023/24"]),
            ("OOS", ["2024/25"]),
        ]:
            alt = df[df["Season"].isin(sezonlar)]
            n = len(alt)
            s = int(alt["signal"].sum())
            bolme_rows.append({
                "Bolme": etiket,
                "N_toplam": n,
                "N_sinyal": s,
                "Sinyal_oran_%": round(s / n * 100, 4) if n > 0 else 0,
            })
        pd.DataFrame(bolme_rows).to_excel(w, sheet_name="Sinyal_Bolme", index=False)

        # Elo dağılımı
        elo_rows = []
        s = df["Elo_Diff"].dropna()
        for k, v in [("N", len(s)), ("Min", float(s.min())), ("P25", float(s.quantile(0.25))),
                      ("P50", float(s.quantile(0.50))), ("P75", float(s.quantile(0.75))),
                      ("Max", float(s.max())), ("Mean", float(s.mean())), ("Std", float(s.std()))]:
            elo_rows.append({"Metrik": k, "Deger": v})
        pd.DataFrame(elo_rows).to_excel(w, sheet_name="Elo_Diff_Dagilimi", index=False)

        # Notlar
        pd.DataFrame([{
            "Protokol": "H8-EDGE v1.0 INVENTORY",
            "Tarih": "2026-10-08",
            "Amaç": "Veri uygunluğu ve sinyal yoğunluğu (bilgi amaçlı)",
            "Kapsam": "E0 (4) - E0 (9)",
            "E0_10": "KILITLI - okunmadi",
            "Sinyal": f"Elo_Diff > {ELO_ESIK} AND B365H < {FIYAT_ESIK}",
            "Performans_hesaplandi_mi": "HAYIR (ROI/p/MDD hesaplanmadi)",
            "Esik_degistirildi_mi": "HAYIR",
            "KARAR": karar,
            "B365H_tum_sezonlarda": tum_b365h,
            "FTR_h1_features_te": feat_ftr_ok,
            "Elo_sutunlari_mevcut": feat_elo_ok,
            "Kritik_NaN": int(nan_b365h + nan_ftr + nan_elo),
            "Toplam_sinyal": toplam_sinyal,
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} yazıldı.")
    print(f"\n{'='*80}")
    print(f"SON KARAR: {karar}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()