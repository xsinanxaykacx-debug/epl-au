# -*- coding: utf-8 -*-
"""
h6_inventory.py
H6 Bookmaker Disagreement (AH) — Veri Envanteri

AMAÇ:
- MaxAHH ve AvgAHH sütunlarının varlığını, tutarlılığını, dağılımını belgele.
- AH_BookmakerSpread = MaxAHH - AvgAHH değişkeninin veri olarak
  uygulanabilir olup olmadığını doğrula.

YAPMAZ:
- Model kurmaz
- Eşik aramaz
- Değişken tanımı değiştirmez
- Sonuç/edge aramaz
- H6'yı FROZEN etmez
- 2026/27'yi okumaz

KİLİTLER:
- Yalnızca E0 (4) – E0 (9) okunur
- E0 (10).csv için güvenlik + sayaç
- Sadece H6 değişkenleri taranır: MaxAHH, AvgAHH, AH_BookmakerSpread
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
CIKTI = "h6_inventory.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

MARKET_DOSYALARI = {
    "2019/20": "E0 (4).csv",
    "2020/21": "E0 (5).csv",
    "2021/22": "E0 (6).csv",
    "2022/23": "E0 (7).csv",
    "2023/24": "E0 (8).csv",
    "2024/25": "E0 (9).csv",
}

# H6 için gerekli sütunlar
H6_SUTUNLAR = ["Season", "Date", "HomeTeam", "AwayTeam", "MaxAHH", "AvgAHH"]

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
        return {"N": 0, "Min": None, "P1": None, "P5": None, "P25": None,
                "P50": None, "P75": None, "P95": None, "P99": None,
                "Max": None, "Mean": None, "Std": None}
    return {
        "N": int(len(s)),
        "Min": round(float(s.min()), 4),
        "P1": round(float(s.quantile(0.01)), 4),
        "P5": round(float(s.quantile(0.05)), 4),
        "P25": round(float(s.quantile(0.25)), 4),
        "P50": round(float(s.quantile(0.50)), 4),
        "P75": round(float(s.quantile(0.75)), 4),
        "P95": round(float(s.quantile(0.95)), 4),
        "P99": round(float(s.quantile(0.99)), 4),
        "Max": round(float(s.max()), 4),
        "Mean": round(float(s.mean()), 4),
        "Std": round(float(s.std()), 4),
    }


# ==================================================================
# 2. SÜTUN VARLIK KONTROLÜ
# ==================================================================
def sutun_varlik():
    print("\n" + "=" * 80)
    print("1. SÜTUN VARLIK KONTROLÜ")
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
            "MaxAHH": "MaxAHH" in cols,
            "AvgAHH": "AvgAHH" in cols,
            "MinAHH": "MinAHH" in cols,
            "PAHH": "PAHH" in cols,
        }
        print(f"  {dosya:15s} → {sezon}  ({len(df)} satır × {len(df.columns)} sütun)")
        print(f"    MaxAHH: {'VAR' if per_sezon[sezon]['MaxAHH'] else 'YOK'}")
        print(f"    AvgAHH: {'VAR' if per_sezon[sezon]['AvgAHH'] else 'YOK'}")

    print(f"\n  --- Özet: H6 kritik sütunlar ---")
    for s in ["MaxAHH", "AvgAHH"]:
        satir = "  ".join(f"{sezon}:{'V' if per_sezon[sezon][s] else 'X'}"
                          for sezon in per_sezon)
        sayi = sum(1 for sezon in per_sezon if per_sezon[sezon][s])
        durum = "[TÜMÜ]" if sayi == len(per_sezon) else f"[{sayi}/{len(per_sezon)}]"
        print(f"    {s:10s} {durum}  {satir}")

    return per_sezon


# ==================================================================
# 3. VERİ YÜKLEME VE FİLTRELEME
# ==================================================================
def veri_yukle():
    print("\n" + "=" * 80)
    print("2. VERİ YÜKLEME VE FİLTRELEME")
    print("=" * 80)

    frames = []
    for sezon, dosya in MARKET_DOSYALARI.items():
        ad = os.path.basename(dosya)
        if ad in YASAKLI_DOSYALAR:
            raise RuntimeError(f"YASAKLI DOSYA: {ad}")
        df = _guvenli_oku(dosya)
        df["Season"] = sezon
        eksik = [s for s in H6_SUTUNLAR if s not in df.columns]
        if eksik:
            raise RuntimeError(f"{ad} içinde eksik sütunlar: {eksik}")
        frames.append(df[H6_SUTUNLAR].copy())

    market = pd.concat(frames, ignore_index=True)
    print(f"\n  Market satırı (ham): {len(market)}")

    # h1_features merge (Elo_Diff için)
    if not os.path.exists(H1_FEATURES):
        raise RuntimeError(f"{H1_FEATURES} bulunamadı.")
    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")
    feat["_key"] = _mac_anahtari(feat)
    market["_key"] = _mac_anahtari(market)

    df = feat.merge(market[["_key", "MaxAHH", "AvgAHH"]], on="_key", how="inner")
    print(f"  Merge sonrası: {len(df)}")

    # Duplicate kontrolü
    dup = df["_key"].duplicated().sum()
    print(f"  Duplicate key: {dup}")
    if dup > 0:
        raise RuntimeError(f"Duplicate bulundu: {dup}")

    # AH ailesi eksik filtresi
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
# 4. H6 DEĞİŞKENİNİ OLUŞTUR VE İNCELE
# ==================================================================
def h6_degisken_analiz(df):
    print("\n" + "=" * 80)
    print("3. H6 DEĞİŞKENİ: AH_BookmakerSpread = MaxAHH - AvgAHH")
    print("=" * 80)

    # NaN kontrolü
    print("\n  --- NaN kontrolü ---")
    for c in ["MaxAHH", "AvgAHH"]:
        nn = int(df[c].isna().sum())
        print(f"    {c:10s}: NaN={nn} / {len(df)} (%{nn/len(df)*100:.4f})")

    # Sıfır kontrolü
    print("\n  --- Sıfır ve negatif kontrolü ---")
    for c in ["MaxAHH", "AvgAHH"]:
        sifir = int((df[c] == 0).sum())
        negatif = int((df[c] < 0).sum())
        print(f"    {c:10s}: sıfır={sifir}, negatif={negatif}")

    # Spread oluştur
    df["AH_BookmakerSpread"] = df["MaxAHH"] - df["AvgAHH"]

    # Invariant: MaxAHH >= AvgAHH olmalı
    invariant_ihlal = (df["AH_BookmakerSpread"] < 0).sum()
    print(f"\n  --- Invariant kontrolü: MaxAHH >= AvgAHH ---")
    print(f"    İhlal sayısı: {int(invariant_ihlal)}")
    if invariant_ihlal > 0:
        # İhlal eden satırları göster
        ihlal = df[df["AH_BookmakerSpread"] < 0][["Season", "Date", "HomeTeam", "AwayTeam",
                                                    "MaxAHH", "AvgAHH", "AH_BookmakerSpread"]]
        print(f"    --- İhlal eden maçlar ---")
        print(ihlal.to_string(index=False))

    # Spread dağılımı
    print(f"\n  --- AH_BookmakerSpread dağılımı ---")
    d = _describe(df["AH_BookmakerSpread"])
    for k, v in d.items():
        print(f"    {k:6s}: {v}")

    # Sezon bazında spread dağılımı
    print(f"\n  --- Sezon bazında AH_BookmakerSpread ---")
    for sezon in sorted(df["Season"].unique()):
        alt = df[df["Season"] == sezon]
        d = _describe(alt["AH_BookmakerSpread"])
        print(f"    {sezon} (N={d['N']}): min={d['Min']}, P50={d['P50']}, "
              f"max={d['Max']}, mean={d['Mean']}")

    # Spread = 0 (MaxAHH = AvgAHH) kaç maçta?
    sifir_spread = int((df["AH_BookmakerSpread"] == 0).sum())
    print(f"\n  --- Spread = 0 (MaxAHH = AvgAHH) ---")
    print(f"    Sayı: {sifir_spread} / {len(df)} (%{sifir_spread/len(df)*100:.2f})")

    return df


# ==================================================================
# 5. ELO_DIFF KONTROLÜ
# ==================================================================
def elo_kontrol(df):
    print("\n" + "=" * 80)
    print("4. ELO_DIFF KONTROLÜ")
    print("=" * 80)

    if "Elo_Diff" not in df.columns:
        print("  HATA: Elo_Diff yok!")
        return

    print(f"\n  Elo_Diff:")
    print(f"    NaN: {df['Elo_Diff'].isna().sum()}")
    d = _describe(df["Elo_Diff"])
    for k, v in d.items():
        print(f"    {k:6s}: {v}")


# ==================================================================
# 6. VERİ BÜTÜNLÜĞÜ VE KANIT
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

    print(f"\n  Kritik sütunlarda NaN:")
    for c in ["MaxAHH", "AvgAHH", "AH_BookmakerSpread", "Elo_Diff"]:
        if c in df.columns:
            nn = int(df[c].isna().sum())
            print(f"    {c:22s}: NaN={nn}")

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
    print("H6 INVENTORY — BOOKMAKER DISAGREEMENT (AH)")
    print("2026/27 (E0 (10).csv) KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    # 1. Sütun varlık
    sutun_varlik()

    # 2. Veri yükleme + filtre
    df = veri_yukle()

    # 3. H6 değişkeni
    df = h6_degisken_analiz(df)

    # 4. Elo kontrolü
    elo_kontrol(df)

    # 5. Bütünlük
    veri_butunlugu(df)

    # -------- EXCEL --------
    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:

        # Sütun varlık
        rows = []
        for sezon, dosya in MARKET_DOSYALARI.items():
            if not os.path.exists(dosya):
                continue
            df_check = _guvenli_oku(dosya)
            cols = set(df_check.columns)
            rows.append({
                "Sezon": sezon,
                "Dosya": dosya,
                "N_satir": len(df_check),
                "MaxAHH": "VAR" if "MaxAHH" in cols else "YOK",
                "AvgAHH": "VAR" if "AvgAHH" in cols else "YOK",
            })
        pd.DataFrame(rows).to_excel(w, sheet_name="Sutun_Varlik", index=False)

        # Spread dağılımı
        d = _describe(df["AH_BookmakerSpread"])
        pd.DataFrame([{"Metrik": k, "Deger": v} for k, v in d.items()]).to_excel(
            w, sheet_name="Spread_Dagilimi", index=False)

        # Sezon bazında spread
        sezon_rows = []
        for sezon in sorted(df["Season"].unique()):
            alt = df[df["Season"] == sezon]
            d = _describe(alt["AH_BookmakerSpread"])
            sezon_rows.append({"Sezon": sezon, **d})
        pd.DataFrame(sezon_rows).to_excel(w, sheet_name="Spread_Sezon", index=False)

        # MaxAHH ve AvgAHH dağılımı
        dagilim_rows = []
        for c in ["MaxAHH", "AvgAHH", "AH_BookmakerSpread", "Elo_Diff"]:
            d = _describe(df[c])
            for k, v in d.items():
                dagilim_rows.append({"Sutun": c, "Metrik": k, "Deger": v})
        pd.DataFrame(dagilim_rows).to_excel(w, sheet_name="Tum_Dagilim", index=False)

        # Notlar
        pd.DataFrame([{
            "Protokol": "H6 INVENTORY (protokol değil)",
            "Tarih": "2026-10-08",
            "Amaç": "MaxAHH - AvgAHH veri uygulanabilirliği kontrolü",
            "Kapsam": "E0 (4) - E0 (9)",
            "E0_10": "KILITLI - okunmadi",
            "Model_kurulmadi": "EVET",
            "Esik_aranmadi": "EVET",
            "H6_FROZEN": "HAYIR - inventory sonrası karar verilecek",
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} yazıldı.")


if __name__ == "__main__":
    main()