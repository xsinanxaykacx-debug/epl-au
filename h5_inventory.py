# -*- coding: utf-8 -*-
"""
h5_inventory.py
H5 O/U Piyasa Yapısı Envanteri

AMAÇ:
- E0 (4) – E0 (9) dosyalarında O/U ile ilgili sütunları otomatik listeler.
- O/U line, price, opening/closing ayrımını tespit eder.
- Missingness, dağılım, sezon tutarlılığı raporlanır.

YAPMAZ:
- Model kurmaz
- Eşik aramaz
- OU_Movement tanımı seçmez
- ROI hesaplamaz
- Outcome seçmez
- 2026/27'yi okumaz
"""

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

# ==================================================================
# 0. SABİTLER
# ==================================================================
H1_FEATURES = "h1_features.csv"
ENVANTER = "envanter_v2.xlsx"
ENVANTER_SHEET = "Eksik_Maclar"
CIKTI = "h5_inventory.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

MARKET_DOSYALARI = {
    "2019/20": "E0 (4).csv",
    "2020/21": "E0 (5).csv",
    "2021/22": "E0 (6).csv",
    "2022/23": "E0 (7).csv",
    "2023/24": "E0 (8).csv",
    "2024/25": "E0 (9).csv",
}

# O/U ile ilgili sütunları otomatik tespit etmek için regex
# Football-Data formatı: B365>2.5, B365<2.5, B365C>2.5, B365C<2.5, P>2.5, PC>2.5, ...
OU_DESENLERI = [
    r"^B365[<>C]+2\.5$",
    r"^P[<>C]+2\.5$",
    r"^PC[<>C]+2\.5$",
    r"^Max[<>C]+2\.5$",
    r"^MaxC[<>C]+2\.5$",
    r"^Avg[<>C]+2\.5$",
    r"^AvgC[<>C]+2\.5$",
    r"^BbAv[<>C]+2\.5$",
    r"^BbMx[<>C]+2\.5$",
    r"^B365[<>C]+[0-9]+\.[0-9]+$",  # diğer çizgiler (1.5, 3.5 vb.)
    r"^P[<>C]+[0-9]+\.[0-9]+$",
    r"^OU[0-9]*",
    r"^Over",
    r"^Under",
]

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


def ou_sutun_mu(c):
    for d in OU_DESENLERI:
        if re.match(d, c):
            return True
    return False


# ==================================================================
# 2. O/U SÜTUNLARINI OTOMATİK TESPİT
# ==================================================================
def ou_sutunlari_tara():
    print("\n" + "=" * 80)
    print("1. O/U SÜTUNLARINI OTOMATİK TESPİT (E0 (4) – E0 (9))")
    print("=" * 80)

    per_sezon = {}
    tum_ou_sutunlar = set()

    for sezon, dosya in MARKET_DOSYALARI.items():
        if not os.path.exists(dosya):
            print(f"  [YOK] {dosya}")
            continue
        df = _guvenli_oku(dosya)
        ou_cols = [c for c in df.columns if ou_sutun_mu(c)]
        per_sezon[sezon] = {
            "n_satir": len(df),
            "n_sutun": len(df.columns),
            "ou_sutunlar": ou_cols,
        }
        tum_ou_sutunlar.update(ou_cols)
        print(f"\n  {dosya:15s} → {sezon}  ({len(df)} satır × {len(df.columns)} sütun)")
        print(f"    O/U ile ilgili sütunlar ({len(ou_cols)}):")
        for c in sorted(ou_cols):
            print(f"      {c}")

    # Tüm sezonlarda ortak O/U sütunları
    print(f"\n  Toplam farklı O/U sütunu (tüm sezonlar): {len(tum_ou_sutunlar)}")

    # Sezon bazında varlık matrisi
    print(f"\n  --- Sezon bazında O/U sütun varlık matrisi ---")
    for c in sorted(tum_ou_sutunlar):
        satir = "  ".join(
            f"{s}:{'V' if c in per_sezon[s]['ou_sutunlar'] else 'X'}"
            for s in per_sezon
        )
        # Kaç sezonda var?
        sayi = sum(1 for s in per_sezon if c in per_sezon[s]["ou_sutunlar"])
        durum = "[TÜMÜ]" if sayi == len(per_sezon) else f"[{sayi}/{len(per_sezon)}]"
        print(f"    {c:20s} {durum}  {satir}")

    return per_sezon, tum_ou_sutunlar


# ==================================================================
# 3. O/U SÜTUNLARINI SINIFLANDIR
# ==================================================================
def ou_siniflandir(sutunlar):
    print("\n" + "=" * 80)
    print("2. O/U SÜTUNLARINI SINIFLANDIR")
    print("=" * 80)

    siniflar = {
        "Line_OU": [],          # O/U çizgisi (örn. 2.5 sabit — Football-Data'da yok)
        "Over_Price_Ac": [],    # Over açılış fiyatı
        "Over_Price_Kp": [],    # Over kapanış fiyatı
        "Under_Price_Ac": [],   # Under açılış fiyatı
        "Under_Price_Kp": [],   # Under kapanış fiyatı
        "Diger": [],
    }

    for c in sorted(sutunlar):
        c_upper = c.upper()
        # Over/Under ve opening/closing ayrımı
        if "C" in c_upper and (">" in c or "<" in c):
            if ">" in c:
                siniflar["Over_Price_Kp"].append(c)
            elif "<" in c:
                siniflar["Under_Price_Kp"].append(c)
        elif (">" in c or "<" in c):
            if ">" in c:
                siniflar["Over_Price_Ac"].append(c)
            elif "<" in c:
                siniflar["Under_Price_Ac"].append(c)
        elif "OU" in c_upper or "LINE" in c_upper:
            siniflar["Line_OU"].append(c)
        else:
            siniflar["Diger"].append(c)

    for sinif, sutunlar_liste in siniflar.items():
        print(f"\n  {sinif} ({len(sutunlar_liste)}):")
        if sutunlar_liste:
            for c in sutunlar_liste:
                print(f"    {c}")
        else:
            print("    [YOK]")

    return siniflar


# ==================================================================
# 4. O/U FİYAT DAĞILIMI
# ==================================================================
def ou_fiyat_dagilimi(siniflar):
    print("\n" + "=" * 80)
    print("3. O/U FİYAT DAĞILIMI")
    print("=" * 80)

    frames = []
    for sezon, dosya in MARKET_DOSYALARI.items():
        df = _guvenli_oku(dosya)
        df["Season"] = sezon
        frames.append(df)
    market = pd.concat(frames, ignore_index=True)
    print(f"\n  Toplam market satırı: {len(market)}")

    # Her sınıftaki sütunların dağılımı
    for sinif in ["Over_Price_Ac", "Over_Price_Kp", "Under_Price_Ac", "Under_Price_Kp"]:
        sutunlar_liste = siniflar.get(sinif, [])
        if not sutunlar_liste:
            continue
        print(f"\n  --- {sinif} ---")
        for c in sutunlar_liste:
            if c in market.columns:
                s = _describe(market[c])
                nn = market[c].isna().sum()
                print(f"    {c:20s}: N={s['N']}, NaN={nn}, "
                      f"min={s['Min']}, P50={s['P50']}, max={s['Max']}, mean={s['Mean']}")

    return market


# ==================================================================
# 5. SEZON BAZINDA EKSİK VERİ
# ==================================================================
def sezon_eksik_veri(market, siniflar):
    print("\n" + "=" * 80)
    print("4. SEZON BAZINDA EKSİK VERİ (O/U)")
    print("=" * 80)

    # En kritik sütunlar: Over_Price_Ac, Over_Price_Kp
    kritik = []
    for sinif in ["Over_Price_Ac", "Over_Price_Kp", "Under_Price_Ac", "Under_Price_Kp"]:
        kritik.extend(siniflar.get(sinif, [])[:2])  # ilk 2 tanesini al (B365 >2.5 gibi)

    print(f"\n  Kritik O/U sütunları: {kritik}")

    for c in kritik:
        if c not in market.columns:
            continue
        print(f"\n  {c}:")
        for sezon in sorted(market["Season"].unique()):
            alt = market[market["Season"] == sezon]
            nn = alt[c].isna().sum()
            oran = nn / len(alt) * 100
            print(f"    {sezon}: NaN={nn} / {len(alt)} (%{oran:.2f})")


# ==================================================================
# 6. H1_FEATURES İLE EŞLEŞME
# ==================================================================
def eslesme_kontrol(market):
    print("\n" + "=" * 80)
    print("5. H1_FEATURES İLE EŞLEŞME KONTROLÜ")
    print("=" * 80)

    if not os.path.exists(H1_FEATURES):
        print(f"  HATA: {H1_FEATURES} bulunamadı.")
        return

    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")
    feat["_key"] = _mac_anahtari(feat)
    market["_key"] = _mac_anahtari(market)

    # AH filtresi
    eksik = pd.read_excel(ENVANTER, sheet_name=ENVANTER_SHEET)
    eksik["_key"] = _mac_anahtari(eksik)
    eksik_ah = set(eksik.loc[eksik["AH_Eksik"] == True, "_key"].dropna().astype(str))

    market["_eksik_ah"] = market["_key"].isin(eksik_ah)
    market_f = market[~market["_eksik_ah"]].copy()

    feat_ilgili = feat[feat["Season"].isin(MARKET_DOSYALARI.keys())]

    feat_keys = set(feat_ilgili["_key"])
    market_keys = set(market_f["_key"])

    print(f"  Feature (2019/20–2024/25) unique key: {len(feat_keys)}")
    print(f"  Market unique key (AH filtresi sonrası): {len(market_keys)}")
    print(f"  Kesişim: {len(feat_keys & market_keys)}")
    print(f"  Yalnız feature: {len(feat_keys - market_keys)}")
    print(f"  Yalnız market: {len(market_keys - feat_keys)}")
    print(f"  Duplicate (feature): {feat_ilgili['_key'].duplicated().sum()}")
    print(f"  Duplicate (market): {market_f['_key'].duplicated().sum()}")


# ==================================================================
# 7. VERİ BÜTÜNLÜĞÜ
# ==================================================================
def veri_butunlugu(market):
    print("\n" + "=" * 80)
    print("6. VERİ BÜTÜNLÜĞÜ VE KANIT")
    print("=" * 80)

    print(f"  Toplam market satırı (ham): {len(market)}")
    print(f"  Sezon dağılımı:")
    for s in sorted(market["Season"].unique()):
        n = len(market[market["Season"] == s])
        print(f"    {s}: {n}")

    print("\n--- Dosya okuma sayacı ---")
    for ad in sorted(_okuma_sayaci.keys()):
        print(f"  {ad}: READ {_okuma_sayaci[ad]} TIMES")
    for ad in YASAKLI_DOSYALAR:
        print(f"  [YASAKLI] {ad}: READ {_okuma_sayaci.get(ad, 0)} TIMES")


# ==================================================================
# ANA
# ==================================================================
def main():
    print("=" * 80)
    print("H5 INVENTORY — O/U PİYASA YAPISI TESPİTİ")
    print("2026/27 (E0 (10).csv) KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    # 1. O/U sütunlarını otomatik tespit
    per_sezon, tum_ou_sutunlar = ou_sutunlari_tara()

    # 2. Sınıflandır
    siniflar = ou_siniflandir(tum_ou_sutunlar)

    # 3. Fiyat dağılımı
    market = ou_fiyat_dagilimi(siniflar)

    # 4. Sezon bazında eksik veri
    sezon_eksik_veri(market, siniflar)

    # 5. H1 eşleşme
    eslesme_kontrol(market)

    # 6. Bütünlük
    veri_butunlugu(market)

    # -------- EXCEL --------
    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:

        # Sütun varlık
        rows = []
        for c in sorted(tum_ou_sutunlar):
            row = {"Sutun": c}
            for sezon in MARKET_DOSYALARI:
                row[sezon] = "VAR" if c in per_sezon.get(sezon, {}).get("ou_sutunlar", []) else "YOK"
            rows.append(row)
        pd.DataFrame(rows).to_excel(w, sheet_name="OU_Sutun_Varlik", index=False)

        # Sınıflandırma
        sinif_rows = []
        for sinif, sutunlar_liste in siniflar.items():
            for c in sutunlar_liste:
                sinif_rows.append({"Sinif": sinif, "Sutun": c})
        pd.DataFrame(sinif_rows).to_excel(w, sheet_name="OU_Siniflandirma", index=False)

        # Fiyat dağılımı
        fiyat_rows = []
        for sinif in ["Over_Price_Ac", "Over_Price_Kp", "Under_Price_Ac", "Under_Price_Kp"]:
            for c in siniflar.get(sinif, []):
                if c in market.columns:
                    d = _describe(market[c])
                    d["Sutun"] = c
                    d["Sinif"] = sinif
                    fiyat_rows.append(d)
        pd.DataFrame(fiyat_rows).to_excel(w, sheet_name="OU_Fiyat_Dagilimi", index=False)

        # Eksik veri
        eksik_rows = []
        for c in sorted(tum_ou_sutunlar):
            if c in market.columns:
                eksik_rows.append({
                    "Sutun": c,
                    "N": int(market[c].notna().sum()),
                    "Eksik": int(market[c].isna().sum()),
                    "Eksik_oran_%": round(market[c].isna().sum() / len(market) * 100, 4),
                })
        pd.DataFrame(eksik_rows).to_excel(w, sheet_name="OU_Eksik_Veri", index=False)

        # Notlar
        pd.DataFrame([{
            "Protokol": "H5 INVENTORY (protokol değil)",
            "Tarih": "2026-10-08",
            "Amaç": "O/U piyasa yapısı tespiti - model/eşik/OU_Movement tanımı YOK",
            "Kapsam": "E0 (4) - E0 (9)",
            "E0_10": "KILITLI - okunmadi",
            "Toplam_OU_sutun": len(tum_ou_sutunlar),
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} yazıldı.")


if __name__ == "__main__":
    main()