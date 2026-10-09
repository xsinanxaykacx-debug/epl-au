# -*- coding: utf-8 -*-
"""
h4_inventory.py
H4 Cross-Market (AH ↔ 1X2) Veri Envanteri

AMAÇ:
- AH ve 1X2 B365 sütunlarının varlığını, tutarlılığını, dağılımını ve
  birleşme uygunluğunu belgelemek.

YAPMAZ:
- Cross-market spread hesaplamaz
- İmplied probability normalizasyonu seçmez
- Model kurmaz, eşik aramaz, ROI hesaplamaz
- Outcome seçmez
- 2026/27'yi okumaz
- H2/H3 sonuçlarını kullanmaz

KİLİTLER:
- Yalnızca E0 (4) – E0 (9) okunur
- E0 (10).csv için güvenlik + okuma sayacı
- Sütun tutarlılığı, eksik veri, ham dağılım
- h1_features.csv ile eşleşme
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
CIKTI = "h4_inventory.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

MARKET_DOSYALARI = {
    "2019/20": "E0 (4).csv",
    "2020/21": "E0 (5).csv",
    "2021/22": "E0 (6).csv",
    "2022/23": "E0 (7).csv",
    "2023/24": "E0 (8).csv",
    "2024/25": "E0 (9).csv",
}

# H4 için hedef sütunlar
AH_SUTUNLAR = ["AHh", "AHCh", "B365AHH", "B365AHA", "B365CAHH", "B365CAHA"]
X12_SUTUNLAR = ["B365H", "B365D", "B365A"]
FIYAT_SUTUNLAR = ["B365AHH", "B365AHA", "B365CAHH", "B365CAHA", "B365H", "B365D", "B365A"]
KIMLIK_SUTUNLAR = ["Season", "Date", "HomeTeam", "AwayTeam"]

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


def _describe(seri):
    s = seri.dropna()
    if len(s) == 0:
        return {"N": 0, "Min": None, "P5": None, "P25": None, "P50": None,
                "P75": None, "P95": None, "Max": None, "Mean": None}
    return {
        "N": int(len(s)),
        "Min": round(float(s.min()), 4),
        "P5": round(float(s.quantile(0.05)), 4),
        "P25": round(float(s.quantile(0.25)), 4),
        "P50": round(float(s.quantile(0.50)), 4),
        "P75": round(float(s.quantile(0.75)), 4),
        "P95": round(float(s.quantile(0.95)), 4),
        "Max": round(float(s.max()), 4),
        "Mean": round(float(s.mean()), 4),
    }


# ==================================================================
# 2. SÜTUN VARLIK KONTROLÜ
# ==================================================================
def sutun_varlik_taramasi():
    print("\n" + "=" * 80)
    print("1. SÜTUN VARLIK TARAMASI (E0 (4) – E0 (9))")
    print("=" * 80)

    per_sezon = {}
    for sezon, dosya in MARKET_DOSYALARI.items():
        if not os.path.exists(dosya):
            print(f"  [YOK] {dosya}")
            continue
        df = _guvenli_oku(dosya)
        cols = set(df.columns)
        per_sezon[sezon] = {
            "n_satir": len(df),
            "n_sutun": len(df.columns),
            "AH": {c: (c in cols) for c in AH_SUTUNLAR},
            "X12": {c: (c in cols) for c in X12_SUTUNLAR},
            "Fiyat": {c: (c in cols) for c in FIYAT_SUTUNLAR},
            "Kimlik": {c: (c in cols) for c in KIMLIK_SUTUNLAR},
        }
        print(f"\n  {dosya:15s} → {sezon}  ({len(df)} satır × {len(df.columns)} sütun)")

    print("\n  --- AH sütunları ---")
    for c in AH_SUTUNLAR:
        satir = "  ".join(f"{s}:{'V' if per_sezon[s]['AH'][c] else 'X'}"
                          for s in per_sezon)
        print(f"    {c:12s}  {satir}")

    print("\n  --- 1X2 sütunları ---")
    for c in X12_SUTUNLAR:
        satir = "  ".join(f"{s}:{'V' if per_sezon[s]['X12'][c] else 'X'}"
                          for s in per_sezon)
        print(f"    {c:12s}  {satir}")

    print("\n  --- Fiyat sütunları (ortak) ---")
    for c in FIYAT_SUTUNLAR:
        sayi = sum(1 for s in per_sezon if per_sezon[s]["Fiyat"][c])
        durum = "[TÜM SEZONLARDA VAR]" if sayi == len(per_sezon) else f"[{sayi}/{len(per_sezon)}]"
        print(f"    {c:12s} {durum}")

    return per_sezon


# ==================================================================
# 3. EKSİK VERİ
# ==================================================================
def eksik_veri_analizi():
    print("\n" + "=" * 80)
    print("2. EKSİK VERİ ANALİZİ")
    print("=" * 80)

    # Birleşik market tablosu
    frames = []
    for sezon, dosya in MARKET_DOSYALARI.items():
        df = _guvenli_oku(dosya)
        df["Season"] = sezon
        gerekli = KIMLIK_SUTUNLAR + AH_SUTUNLAR + X12_SUTUNLAR
        eksik = [c for c in gerekli if c not in df.columns]
        if eksik:
            print(f"  [UYARI] {dosya} eksik sütunlar: {eksik}")
        # Var olan sütunları al
        mevcut = [c for c in gerekli if c in df.columns]
        frames.append(df[mevcut].copy())

    market = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\n  Toplam market satırı (ham): {len(market)}")

    print(f"\n  Sütun bazında eksik oran:")
    for c in FIYAT_SUTUNLAR:
        if c in market.columns:
            nn = market[c].isna().sum()
            oran = nn / len(market) * 100
            print(f"    {c:12s}: {nn:5d} / {len(market)} (%{oran:.2f})")
        else:
            print(f"    {c:12s}: SÜTUN YOK")

    # Sezon bazında eksik
    print(f"\n  Sezon bazında eksik oran (kritik sütunlar):")
    kritik = ["B365AHH", "B365CAHH", "B365H", "B365A"]
    for sezon in sorted(market["Season"].unique()):
        alt = market[market["Season"] == sezon]
        print(f"\n    {sezon} (N={len(alt)}):")
        for c in kritik:
            if c in alt.columns:
                nn = alt[c].isna().sum()
                print(f"      {c:12s}: {nn} (%{nn/len(alt)*100:.2f})")

    return market


# ==================================================================
# 4. AH EKSİK FİLTRESİ (H1/H2/H3 ile uyumlu)
# ==================================================================
def ah_eksik_filtresi(market):
    print("\n" + "=" * 80)
    print("3. AH EKSİK FİLTRE UYGULAMASI")
    print("=" * 80)

    eksik = pd.read_excel(ENVANTER, sheet_name=ENVANTER_SHEET)
    eksik["_key"] = _mac_anahtari(eksik)
    eksik_ah = set(eksik.loc[eksik["AH_Eksik"] == True, "_key"].dropna().astype(str))

    market["_key"] = _mac_anahtari(market)
    onceki = len(market)
    market = market[~market["_key"].isin(eksik_ah)].copy().reset_index(drop=True)
    sonraki = len(market)
    print(f"  {onceki} → {sonraki}  (dışlanan: {onceki - sonraki})")
    return market


# ==================================================================
# 5. HAM FİYAT DAĞILIMLARI
# ==================================================================
def fiyat_dagilimi(market):
    print("\n" + "=" * 80)
    print("4. HAM FİYAT DAĞILIMLARI (özet)")
    print("=" * 80)

    print("\n  --- AH B365 ---")
    for c in ["B365AHH", "B365AHA", "B365CAHH", "B365CAHA"]:
        if c not in market.columns:
            print(f"    {c:12s}: SÜTUN YOK")
            continue
        s = _describe(market[c])
        print(f"    {c:12s}: N={s['N']}, min={s['Min']}, P25={s['P25']}, "
              f"P50={s['P50']}, P75={s['P75']}, max={s['Max']}, mean={s['Mean']}")

    print("\n  --- 1X2 B365 ---")
    for c in ["B365H", "B365D", "B365A"]:
        if c not in market.columns:
            print(f"    {c:12s}: SÜTUN YOK")
            continue
        s = _describe(market[c])
        print(f"    {c:12s}: N={s['N']}, min={s['Min']}, P25={s['P25']}, "
              f"P50={s['P50']}, P75={s['P75']}, max={s['Max']}, mean={s['Mean']}")


# ==================================================================
# 6. HAM 1/ODDS İSTATİSTİKLERİ (normalizasyon YOK)
# ==================================================================
def ham_implied_istatistikleri(market):
    print("\n" + "=" * 80)
    print("5. HAM 1/ODDS İSTATİSTİKLERİ (normalizasyon YOK)")
    print("=" * 80)
    print("  Not: Bu bölüm yalnızca ham istatistiktir. Overround düzeltmesi,")
    print("       normalize etme yöntemi H4 protokolüne bırakılmıştır.")

    # AH tarafı — Home / Away
    if "B365AHH" in market.columns:
        market["_inv_AHH"] = 1.0 / market["B365AHH"]
    if "B365AHA" in market.columns:
        market["_inv_AHA"] = 1.0 / market["B365AHA"]
    if "B365CAHH" in market.columns:
        market["_inv_CAHH"] = 1.0 / market["B365CAHH"]
    if "B365CAHA" in market.columns:
        market["_inv_CAHA"] = 1.0 / market["B365CAHA"]

    # 1X2 tarafı
    if "B365H" in market.columns:
        market["_inv_1H"] = 1.0 / market["B365H"]
    if "B365D" in market.columns:
        market["_inv_1D"] = 1.0 / market["B365D"]
    if "B365A" in market.columns:
        market["_inv_1A"] = 1.0 / market["B365A"]

    print("\n  --- AH B365 (ham 1/odds) ---")
    for c in ["_inv_AHH", "_inv_AHA", "_inv_CAHH", "_inv_CAHA"]:
        if c in market.columns:
            s = _describe(market[c])
            print(f"    {c:12s}: N={s['N']}, min={s['Min']}, P50={s['P50']}, max={s['Max']}, mean={s['Mean']}")

    print("\n  --- 1X2 B365 (ham 1/odds) ---")
    for c in ["_inv_1H", "_inv_1D", "_inv_1A"]:
        if c in market.columns:
            s = _describe(market[c])
            print(f"    {c:12s}: N={s['N']}, min={s['Min']}, P50={s['P50']}, max={s['Max']}, mean={s['Mean']}")

    # 1X2 overround (ham toplam — normalizasyon değil, sadece toplam)
    if all(c in market.columns for c in ["_inv_1H", "_inv_1D", "_inv_1A"]):
        market["_1X2_overround"] = market["_inv_1H"] + market["_inv_1D"] + market["_inv_1A"]
        s = _describe(market["_1X2_overround"])
        print(f"\n  --- 1X2 overround (ham toplam, sadece istatistik) ---")
        print(f"    N={s['N']}, min={s['Min']}, P25={s['P25']}, P50={s['P50']}, "
              f"P75={s['P75']}, max={s['Max']}, mean={s['Mean']}")

    return market


# ==================================================================
# 7. H1_FEATURES İLE EŞLEŞME
# ==================================================================
def eslesme_kontrol(market):
    print("\n" + "=" * 80)
    print("6. H1_FEATURES İLE EŞLEŞME KONTROLÜ")
    print("=" * 80)

    if not os.path.exists(H1_FEATURES):
        print(f"  HATA: {H1_FEATURES} bulunamadı.")
        return

    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")
    feat["_key"] = _mac_anahtari(feat)

    feat_ilgili = feat[feat["Season"].isin(MARKET_DOSYALARI.keys())]

    feat_keys = set(feat_ilgili["_key"])
    market_keys = set(market["_key"])

    print(f"  Feature (2019/20–2024/25) unique key: {len(feat_keys)}")
    print(f"  Market unique key: {len(market_keys)}")
    print(f"  Kesişim: {len(feat_keys & market_keys)}")
    print(f"  Yalnız feature: {len(feat_keys - market_keys)}")
    print(f"  Yalnız market: {len(market_keys - feat_keys)}")

    print(f"\n  Feature duplicate key: {feat_ilgili['_key'].duplicated().sum()}")
    print(f"  Market duplicate key: {market['_key'].duplicated().sum()}")


# ==================================================================
# 8. VERİ BÜTÜNLÜĞÜ KANITI
# ==================================================================
def veri_butunlugu(market):
    print("\n" + "=" * 80)
    print("7. VERİ BÜTÜNLÜĞÜ VE KANIT")
    print("=" * 80)

    print(f"  Toplam satır (AH filtresi sonrası): {len(market)}")
    print(f"  Sezon dağılımı:")
    for s in sorted(market["Season"].unique()):
        n = len(market[market["Season"] == s])
        print(f"    {s}: {n}")

    print(f"\n  Kritik sütunlarda NaN kontrolü:")
    for c in ["B365AHH", "B365AHA", "B365CAHH", "B365CAHA", "B365H", "B365D", "B365A"]:
        if c in market.columns:
            nn = market[c].isna().sum()
            print(f"    {c:12s}: NaN={nn} / {len(market)}")

    print(f"\n  --- Dosya okuma sayacı ---")
    for ad in sorted(_okuma_sayaci.keys()):
        print(f"    {ad}: READ {_okuma_sayaci[ad]} TIMES")
    for ad in YASAKLI_DOSYALAR:
        print(f"    [YASAKLI] {ad}: READ {_okuma_sayaci.get(ad, 0)} TIMES")


# ==================================================================
# ANA
# ==================================================================
def main():
    print("=" * 80)
    print("H4 INVENTORY — AH ↔ 1X2 CROSS-MARKET")
    print("2026/27 (E0 (10).csv) KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    # 1. Sütun varlık
    sutun_varlik_taramasi()

    # 2. Eksik veri (ham)
    market = eksik_veri_analizi()

    # 3. AH filtresi
    market = ah_eksik_filtresi(market)

    # 4. Fiyat dağılımı
    fiyat_dagilimi(market)

    # 5. Ham 1/odds
    market = ham_implied_istatistikleri(market)

    # 6. Eşleşme
    eslesme_kontrol(market)

    # 7. Bütünlük
    veri_butunlugu(market)

    # -------- EXCEL --------
    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:

        # Sütun varlık
        per_sezon = {}
        for sezon, dosya in MARKET_DOSYALARI.items():
            df = _guvenli_oku(dosya)
            per_sezon[sezon] = set(df.columns)
        rows = []
        for c in FIYAT_SUTUNLAR:
            row = {"Sutun": c}
            for sezon in MARKET_DOSYALARI:
                row[sezon] = "VAR" if c in per_sezon.get(sezon, set()) else "YOK"
            rows.append(row)
        pd.DataFrame(rows).to_excel(w, sheet_name="Sutun_Varlik", index=False)

        # Fiyat dağılımı
        fiyat_rows = []
        for c in FIYAT_SUTUNLAR:
            if c in market.columns:
                d = _describe(market[c])
                d["Sutun"] = c
                fiyat_rows.append(d)
        pd.DataFrame(fiyat_rows).to_excel(w, sheet_name="Fiyat_Dagilimi", index=False)

        # Ham 1/odds dağılımı
        inv_rows = []
        for c in ["_inv_AHH", "_inv_AHA", "_inv_CAHH", "_inv_CAHA",
                  "_inv_1H", "_inv_1D", "_inv_1A", "_1X2_overround"]:
            if c in market.columns:
                d = _describe(market[c])
                d["Sutun"] = c
                inv_rows.append(d)
        pd.DataFrame(inv_rows).to_excel(w, sheet_name="Ham_Implied", index=False)

        # Eksik veri
        eksik_rows = []
        for c in FIYAT_SUTUNLAR:
            if c in market.columns:
                eksik_rows.append({
                    "Sutun": c,
                    "N": int(market[c].notna().sum()),
                    "Eksik": int(market[c].isna().sum()),
                    "Eksik_oran_%": round(market[c].isna().sum() / len(market) * 100, 4),
                })
        pd.DataFrame(eksik_rows).to_excel(w, sheet_name="Eksik_Veri", index=False)

        # Sezon dağılımı
        sezon_rows = []
        for s in sorted(market["Season"].unique()):
            alt = market[market["Season"] == s]
            sezon_rows.append({"Sezon": s, "N": len(alt)})
        pd.DataFrame(sezon_rows).to_excel(w, sheet_name="Sezon_Dagilimi", index=False)

        # Notlar
        pd.DataFrame([{
            "Protokol": "H4 INVENTORY (protokol değil)",
            "Tarih": "2026-10-08",
            "Kapsam": "AH B365 + 1X2 B365",
            "Amaç": "Veri yapısı tespiti - model/hipotez/eşik YOK",
            "Normalizasyon": "YAPILMADI - sadece ham 1/odds istatistikleri",
            "Spread": "HESAPLANMADI - H4 protokolüne bırakıldı",
            "E0_10": "KILITLI - okunmadi",
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} yazıldı.")


if __name__ == "__main__":
    main()