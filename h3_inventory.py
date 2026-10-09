# -*- coding: utf-8 -*-
"""
h3_inventory.py
H3 Veri Envanteri — Mevcut veri durumu tespiti

YAPMAZ:
- Model kurmaz
- Eşik aramaz
- ROI hesaplamaz
- 2026/27'yi okumaz
- H3 hipotezini değiştirmez

SADECE:
- Fiyat dağılımı
- Movement dağılımı
- Elo dağılımı
- Gözlem yoğunluğu
- Eksik veri kontrolü
- Mevcut sütunlar
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
CIKTI = "h3_inventory.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

MARKET_DOSYALARI = {
    "2019/20": "E0 (4).csv",
    "2020/21": "E0 (5).csv",
    "2021/22": "E0 (6).csv",
    "2022/23": "E0 (7).csv",
    "2023/24": "E0 (8).csv",
    "2024/25": "E0 (9).csv",
}

_okuma_sayaci = defaultdict(int)


# ==================================================================
# 1. YARDIMCI
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
# 2. MARKET SÜTUNLARINI KONTROL ET
# ==================================================================
def market_sutunlarini_tara():
    print("\n" + "=" * 80)
    print("1. MARKET SÜTUNLARI TARAMASI")
    print("=" * 80)

    tum_sutunlar = {}
    for sezon, dosya in MARKET_DOSYALARI.items():
        if not os.path.exists(dosya):
            print(f"  [YOK] {dosya}")
            continue
        df = _guvenli_oku(dosya)
        print(f"  {dosya:15s} → {sezon}  ({len(df)} satır, {len(df.columns)} sütun)")
        tum_sutunlar[sezon] = set(df.columns)

    if tum_sutunlar:
        ortak = set.intersection(*tum_sutunlar.values())
        print(f"\n  Tüm sezonlarda ortak sütun sayısı: {len(ortak)}")

        kritik = {
            "AH açılış handikap": "AHh",
            "AH kapanış handikap": "AHCh",
            "B365 ev açılış": "B365AHH",
            "B365 ev kapanış": "B365CAHH",
            "B365 dep açılış": "B365AHA",
            "B365 dep kapanış": "B365CAHA",
            "Pinnacle ev açılış": "PAHH",
            "Pinnacle ev kapanış": "PCAHH",
            "Pinnacle dep açılış": "PAHA",
            "Pinnacle dep kapanış": "PCAHA",
            "1X2 ev açılış (B365H)": "B365H",
            "1X2 beraberlik (B365D)": "B365D",
            "1X2 dep (B365A)": "B365A",
            "O/U Over açılış": "B365>2.5",
            "O/U Under açılış": "B365<2.5",
            "FTHG": "FTHG", "FTAG": "FTAG", "FTR": "FTR",
        }
        print("\n  H3 için kritik sütunların varlığı (tüm sezonlarda ortak):")
        for ad, sutun in kritik.items():
            durum = "[VAR]" if sutun in ortak else "[YOK]"
            print(f"    {durum} {ad:30s} ({sutun})")

    return tum_sutunlar


# ==================================================================
# 3. FİYAT DAĞILIMI
# ==================================================================
def fiyat_dagilimi():
    print("\n" + "=" * 80)
    print("2. AH FİYAT DAĞILIMI (açılış / kapanış)")
    print("=" * 80)

    frames = []
    for sezon, dosya in MARKET_DOSYALARI.items():
        df = _guvenli_oku(dosya)
        df["Season"] = sezon
        gerekli = ["Season", "Date", "HomeTeam", "AwayTeam",
                    "AHh", "AHCh", "B365AHH", "B365CAHH", "B365AHA", "B365CAHA",
                    "FTHG", "FTAG"]
        eksik = [c for c in gerekli if c not in df.columns]
        if eksik:
            print(f"  [UYARI] {dosya} eksik sütunlar: {eksik}")
            continue
        frames.append(df[gerekli].copy())
    market = pd.concat(frames, ignore_index=True)

    eksik = pd.read_excel(ENVANTER, sheet_name=ENVANTER_SHEET)
    eksik["_key"] = _mac_anahtari(eksik)
    eksik_ah = set(eksik.loc[eksik["AH_Eksik"] == True, "_key"].dropna().astype(str))
    market["_key"] = _mac_anahtari(market)
    market = market[~market["_key"].isin(eksik_ah)].copy()

    print(f"\n  Toplam market satırı (AH eksik filtresi sonrası): {len(market)}")

    for col, ad in [("B365AHH", "B365 ev açılış"),
                     ("B365CAHH", "B365 ev kapanış"),
                     ("B365AHA", "B365 dep açılış"),
                     ("B365CAHA", "B365 dep kapanış")]:
        seri = market[col].dropna()
        print(f"\n  {ad} ({col}):")
        print(f"    N: {len(seri)} / {len(market)} (eksik: {market[col].isna().sum()})")
        if len(seri) > 0:
            print(f"    min: {seri.min():.2f}")
            print(f"    %5: {seri.quantile(0.05):.2f}")
            print(f"    %25: {seri.quantile(0.25):.2f}")
            print(f"    %50: {seri.quantile(0.50):.2f}")
            print(f"    %75: {seri.quantile(0.75):.2f}")
            print(f"    %95: {seri.quantile(0.95):.2f}")
            print(f"    max: {seri.max():.2f}")

    return market


# ==================================================================
# 4. FİYAT KOVALARININ DOLULUĞU
# ==================================================================
def fiyat_kovalari(market):
    print("\n" + "=" * 80)
    print("3. FİYAT KOVALARININ DOLULUĞU (bilgi amaçlı — protokol değil)")
    print("=" * 80)

    def kova(f):
        if pd.isna(f): return None
        if f < 1.50: return "P1 <1.50"
        elif f < 1.80: return "P2 1.50-1.79"
        elif f < 2.10: return "P3 1.80-2.09"
        elif f < 2.50: return "P4 2.10-2.49"
        else: return "P5 >=2.50"

    for col, ad in [("B365AHH", "B365 ev açılış"),
                     ("B365CAHH", "B365 ev kapanış")]:
        market["_kova"] = market[col].apply(kova)
        print(f"\n  {ad} ({col}) kovaları:")
        sayim = market["_kova"].value_counts(dropna=False).sort_index()
        for k, n in sayim.items():
            print(f"    {k}: {n}")

    print("\n  B365 ev açılış kovaları — sezon bazında:")
    market["_kova"] = market["B365AHH"].apply(kova)
    for sezon in sorted(market["Season"].unique()):
        alt = market[market["Season"] == sezon]
        print(f"\n    {sezon} (N={len(alt)}):")
        sayim = alt["_kova"].value_counts(dropna=False).sort_index()
        for k, n in sayim.items():
            print(f"      {k}: {n}")

    market = market.drop(columns=["_kova"], errors="ignore")
    return market


# ==================================================================
# 5. MOVEMENT VE ELO DAĞILIMI
# ==================================================================
def movement_elo_dagilimi(market):
    print("\n" + "=" * 80)
    print("4. MOVEMENT VE ELO DAĞILIMI")
    print("=" * 80)

    market["Movement"] = market["AHCh"] - market["AHh"]
    print("\n  Movement = AHCh - AHh:")
    mv = market["Movement"].dropna()
    print(f"    N: {len(mv)}")
    print(f"    min: {mv.min():.2f}")
    print(f"    %5: {mv.quantile(0.05):.2f}")
    print(f"    %25: {mv.quantile(0.25):.2f}")
    print(f"    %50: {mv.quantile(0.50):.2f}")
    print(f"    %75: {mv.quantile(0.75):.2f}")
    print(f"    %95: {mv.quantile(0.95):.2f}")
    print(f"    max: {mv.max():.2f}")

    print("\n  Movement dağılımı (tam değerler):")
    sayim = market["Movement"].value_counts().sort_index()
    for k, n in sayim.items():
        print(f"    {k:+.2f}: {n}")

    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")
    feat["_key"] = _mac_anahtari(feat)
    market["_key"] = _mac_anahtari(market)
    birlesik = market.merge(feat[["_key", "Home_Elo_Pre", "Away_Elo_Pre"]], on="_key", how="inner")
    birlesik["Elo_Diff"] = birlesik["Home_Elo_Pre"] - birlesik["Away_Elo_Pre"]

    print("\n  Elo_Diff = Home_Elo_Pre - Away_Elo_Pre:")
    elo = birlesik["Elo_Diff"].dropna()
    print(f"    N: {len(elo)}")
    print(f"    min: {elo.min():.2f}")
    print(f"    %25: {elo.quantile(0.25):.2f}")
    print(f"    %50: {elo.quantile(0.50):.2f}")
    print(f"    %75: {elo.quantile(0.75):.2f}")
    print(f"    max: {elo.max():.2f}")

    return market, birlesik


# ==================================================================
# 6. ÇAPRAZ GÖZLEM YOĞUNLUĞU
# ==================================================================
def capraz_yogunluk(birlesik):
    print("\n" + "=" * 80)
    print("5. FİYAT x MOVEMENT GÖZLEM YOĞUNLUĞU")
    print("=" * 80)

    def kova(f):
        if pd.isna(f): return None
        if f < 1.50: return "P1"
        elif f < 1.80: return "P2"
        elif f < 2.10: return "P3"
        elif f < 2.50: return "P4"
        else: return "P5"

    def m_kov(m):
        if pd.isna(m): return None
        if m < -0.25: return "M-"
        elif m > 0.25: return "M+"
        else: return "M0"

    birlesik["_p"] = birlesik["B365AHH"].apply(kova)
    birlesik["_m"] = birlesik["Movement"].apply(m_kov)

    ct = pd.crosstab(birlesik["_p"], birlesik["_m"], margins=True)
    print("\n  Fiyat x Movement çapraz tablosu (tüm sezonlar):")
    print(ct.to_string())

    print("\n  Sezon bazında B365 ev açılış kovaları:")
    for sezon in sorted(birlesik["Season"].unique()):
        alt = birlesik[birlesik["Season"] == sezon]
        sayim = alt["_p"].value_counts(dropna=False).sort_index()
        print(f"\n    {sezon} (N={len(alt)}):")
        for k, n in sayim.items():
            print(f"      {k}: {n}")

    birlesik = birlesik.drop(columns=["_p", "_m"], errors="ignore")
    return birlesik


# ==================================================================
# 7. 1X2 VERİSİ KONTROLÜ
# ==================================================================
def kontrol_1x2():
    print("\n" + "=" * 80)
    print("6. 1X2 VERİSİ KONTROLÜ (gelecek araştırma için)")
    print("=" * 80)

    for sezon, dosya in MARKET_DOSYALARI.items():
        df = _guvenli_oku(dosya)
        cols = set(df.columns)
        adaylar = ["B365H", "B365D", "B365A",
                    "PSH", "PSD", "PSA",
                    "WHH", "WHD", "WHA",
                    "BbAvH", "BbAvD", "BbAvA",
                    "BbMxH", "BbMxD", "BbMxA"]
        bulunan = [c for c in adaylar if c in cols]
        print(f"\n  {sezon} ({dosya}):")
        if bulunan:
            print(f"    Bulunan 1X2 sütunları: {bulunan}")
        else:
            print(f"    1X2 sütunu YOK")


# ==================================================================
# 8. EKSİK VERİ ÖZETİ
# ==================================================================
def eksik_veri_ozeti(market):
    print("\n" + "=" * 80)
    print("7. EKSİK VERİ ÖZETİ")
    print("=" * 80)

    kritik = ["AHh", "AHCh", "B365AHH", "B365CAHH", "B365AHA", "B365CAHA",
              "FTHG", "FTAG"]
    print("\n  Eksik sütun sayıları:")
    for c in kritik:
        if c in market.columns:
            nn = market[c].isna().sum()
            oran = nn / len(market) * 100
            print(f"    {c:12s}: {nn:5d} / {len(market)} (%{oran:.2f})")


# ==================================================================
# 9. H1_FEATURES İLE EŞLEŞME
# ==================================================================
def eslesme_kontrol(market):
    print("\n" + "=" * 80)
    print("8. H1_FEATURES İLE EŞLEŞME KONTROLÜ")
    print("=" * 80)

    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")
    feat["_key"] = _mac_anahtari(feat)
    market["_key"] = _mac_anahtari(market)

    feat_ilgili = feat[feat["Season"].isin(MARKET_DOSYALARI.keys())]

    feat_keys = set(feat_ilgili["_key"])
    market_keys = set(market["_key"])

    print(f"  Feature (2019/20-2024/25) unique key: {len(feat_keys)}")
    print(f"  Market unique key: {len(market_keys)}")
    print(f"  Kesişim: {len(feat_keys & market_keys)}")
    print(f"  Yalnız feature: {len(feat_keys - market_keys)}")
    print(f"  Yalnız market: {len(market_keys - feat_keys)}")


# ==================================================================
# ANA
# ==================================================================
def main():
    print("=" * 80)
    print("H3 VERİ ENVANTERİ")
    print("2026/27 (E0 (10).csv) KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    market_sutunlarini_tara()
    market = fiyat_dagilimi()
    market = fiyat_kovalari(market)
    market, birlesik = movement_elo_dagilimi(market)
    birlesik = capraz_yogunluk(birlesik)
    kontrol_1x2()
    eksik_veri_ozeti(market)
    eslesme_kontrol(market)

    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        pd.DataFrame([{
            "Sutun": "B365AHH",
            "N": int(market["B365AHH"].notna().sum()),
            "Min": float(market["B365AHH"].min()),
            "P50": float(market["B365AHH"].median()),
            "Max": float(market["B365AHH"].max()),
        }, {
            "Sutun": "B365CAHH",
            "N": int(market["B365CAHH"].notna().sum()),
            "Min": float(market["B365CAHH"].min()),
            "P50": float(market["B365CAHH"].median()),
            "Max": float(market["B365CAHH"].max()),
        }]).to_excel(w, sheet_name="Fiyat_Dagilimi", index=False)

        pd.DataFrame(market["Movement"].value_counts().sort_index()).rename(
            columns={"count": "N"}).to_excel(w, sheet_name="Movement_Dagilimi")

        def kova(f):
            if pd.isna(f): return "NaN"
            if f < 1.50: return "P1"
            elif f < 1.80: return "P2"
            elif f < 2.10: return "P3"
            elif f < 2.50: return "P4"
            else: return "P5"
        market["_kova"] = market["B365AHH"].apply(kova)
        pd.crosstab(market["Season"], market["_kova"]).to_excel(w, sheet_name="Kova_Sezon")

        pd.DataFrame([{
            "Protokol": "H3 VERİ ENVANTERİ (protokol değil)",
            "Tarih": "2026-10-08",
            "Amaç": "Mevcut veri durumu tespiti - model kurulmadı, eşik seçilmedi",
            "2026_27": "KILITLI - okunmadı",
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} yazıldı.")

    print("\n--- Dosya okuma sayacı ---")
    for ad in sorted(_okuma_sayaci.keys()):
        print(f"  {ad}: READ {_okuma_sayaci[ad]} TIMES")
    for ad in YASAKLI_DOSYALAR:
        print(f"  [YASAKLI] {ad}: READ {_okuma_sayaci.get(ad, 0)} TIMES")


if __name__ == "__main__":
    main()