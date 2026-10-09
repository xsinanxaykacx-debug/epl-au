# -*- coding: utf-8 -*-
"""
h8_inventory.py
H8 1X2 Home Price Level — Veri Envanteri

AMAÇ:
- B365H sütununun varlığını, tutarlılığını, dağılımını belgele.
- X12_HomePrice = B365H değişkeninin veri olarak
  uygulanabilir olup olmadığını doğrula.

YAPMAZ:
- Model kurmaz
- Eşik aramaz
- Değişken tanımı değiştirmez
- Sonuç/edge aramaz
- H8'i FROZEN etmez
- 2026/27'yi okumaz
- Kısmi PASS veya "yeni filtre kuralı ekle" seçeneği YOK

KARAR MATRİSİ:
- B365H tüm 6 sezonda var + NaN yok + tüm fiyatlar > 0 → PASS
- B365H herhangi bir sezonda yok → FAIL
- Kritik NaN varsa → FAIL
- Sıfır/negatif fiyat varsa → FAIL
- Düşük varyasyon → FAIL değil

DÜZELTME (bu sürüm):
- karar_matrisi() artık nan_h, sifir, neg değerlerini döndürüyor.
- main() içinde bu değerler ayrı değişkenlere atanıyor.
- Notlar sheet'inde kullanılıyor.
- Başka hiçbir satır değişmedi.
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
CIKTI = "h8_inventory.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

MARKET_DOSYALARI = {
    "2019/20": "E0 (4).csv",
    "2020/21": "E0 (5).csv",
    "2021/22": "E0 (6).csv",
    "2022/23": "E0 (7).csv",
    "2023/24": "E0 (8).csv",
    "2024/25": "E0 (9).csv",
}

H8_SUTUNLAR = ["Season", "Date", "HomeTeam", "AwayTeam", "B365H"]

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
    print("1. SÜTUN VARLIK KONTROLÜ (B365H)")
    print("=" * 80)

    per_sezon = {}
    for sezon, dosya in MARKET_DOSYALARI.items():
        if not os.path.exists(dosya):
            print(f"  [YOK] {dosya}")
            per_sezon[sezon] = {"B365H": False, "n_satir": 0}
            continue
        df = _guvenli_oku(dosya)
        cols = set(df.columns)
        per_sezon[sezon] = {
            "n_satir": len(df),
            "n_sutun": len(df.columns),
            "B365H": "B365H" in cols,
        }
        print(f"  {dosya:15s} → {sezon}  ({len(df)} satır × {len(df.columns)} sütun)")
        print(f"    B365H: {'VAR' if per_sezon[sezon]['B365H'] else 'YOK'}")

    print(f"\n  --- Özet: H8 kritik sütun ---")
    satir = "  ".join(f"{sezon}:{'V' if per_sezon[sezon]['B365H'] else 'X'}"
                      for sezon in per_sezon)
    sayi = sum(1 for sezon in per_sezon if per_sezon[sezon]["B365H"])
    durum = "[TÜMÜ]" if sayi == len(per_sezon) else f"[{sayi}/{len(per_sezon)}]"
    print(f"    {'B365H':10s} {durum}  {satir}")

    tum_b365h = all(per_sezon[s]["B365H"] for s in per_sezon)
    print(f"\n  --- KRİTİK KONTROL ---")
    if not tum_b365h:
        print(f"    ❌ B365H bazı sezonlarda YOK → H8 FAIL")
        print(f"       Protokol DEĞİŞMEZ. Alternatif 1X2 sütunu ARANMAZ.")
    else:
        print(f"    ✅ B365H tüm sezonlarda VAR")

    return per_sezon, tum_b365h


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
        eksik = [s for s in H8_SUTUNLAR if s not in df.columns]
        if eksik:
            raise RuntimeError(f"{ad} içinde eksik sütunlar: {eksik}")
        frames.append(df[H8_SUTUNLAR].copy())

    market = pd.concat(frames, ignore_index=True)
    print(f"\n  Market satırı (ham): {len(market)}")

    if not os.path.exists(H1_FEATURES):
        raise RuntimeError(f"{H1_FEATURES} bulunamadı.")
    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")
    feat["_key"] = _mac_anahtari(feat)
    market["_key"] = _mac_anahtari(market)

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
# 4. H8 DEĞİŞKEN ANALİZİ
# ==================================================================
def h8_degisken_analiz(df):
    print("\n" + "=" * 80)
    print("3. H8 DEĞİŞKENİ: X12_HomePrice = B365H")
    print("=" * 80)

    print("\n  --- NaN kontrolü ---")
    nan_h = int(df["B365H"].isna().sum())
    print(f"    B365H: NaN={nan_h} / {len(df)} (%{nan_h/len(df)*100:.4f})")

    print("\n  --- Sıfır ve negatif kontrolü ---")
    sifir = int((df["B365H"] == 0).sum())
    negatif = int((df["B365H"] < 0).sum())
    cok_dusuk = int(((df["B365H"] > 0) & (df["B365H"] < 1.0)).sum())
    print(f"    B365H: sıfır={sifir}, negatif={negatif}, 0<x<1={cok_dusuk}")

    print(f"\n  --- X12_HomePrice dağılımı ---")
    d = _describe(df["B365H"])
    for k, v in d.items():
        print(f"    {k:6s}: {v}")

    print(f"\n  --- Sezon bazında X12_HomePrice ---")
    for sezon in sorted(df["Season"].unique()):
        alt = df[df["Season"] == sezon]
        d = _describe(alt["B365H"])
        print(f"    {sezon} (N={d['N']}): min={d['Min']}, P50={d['P50']}, "
              f"max={d['Max']}, mean={d['Mean']}")

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
    for c in ["B365H", "Elo_Diff"]:
        if c in df.columns:
            nn = int(df[c].isna().sum())
            print(f"    {c:22s}: NaN={nn}")

    print("\n--- Dosya okuma sayacı ---")
    for ad in sorted(_okuma_sayaci.keys()):
        print(f"  {ad}: READ {_okuma_sayaci[ad]} TIMES")
    for ad in YASAKLI_DOSYALAR:
        print(f"  [YASAKLI] {ad}: READ {_okuma_sayaci.get(ad, 0)} TIMES")


# ==================================================================
# 7. KARAR MATRİSİ
# ==================================================================
def karar_matrisi(df, tum_b365h):
    print("\n" + "=" * 80)
    print("6. KARAR MATRİSİ (H8 FROZEN PROTOKOL)")
    print("=" * 80)

    print("\n  Kontroller:")
    print(f"    B365H tüm sezonlarda     : {'✅' if tum_b365h else '❌'}")

    nan_h = df["B365H"].isna().sum()
    nan_ok = nan_h == 0
    print(f"    Kritik NaN yok           : {'✅' if nan_ok else '❌'} (B365H NaN={nan_h})")

    sifir = (df["B365H"] == 0).sum()
    neg = (df["B365H"] < 0).sum()
    fiyat_ok = (sifir == 0) and (neg == 0)
    print(f"    Sıfır/negatif fiyat yok  : {'✅' if fiyat_ok else '❌'}")

    if tum_b365h and nan_ok and fiyat_ok:
        karar = "PASS → Explore"
        print(f"\n  ✅ KARAR: {karar}")
    else:
        karar = "FAIL → H8 KAPANIR"
        print(f"\n  ❌ KARAR: {karar}")
        print(f"     Protokol DEĞİŞMEZ.")
        print(f"     Yeni filtre kuralı EKLENMEZ.")
        print(f"     Alternatif 1X2 sütunu ARANMAZ.")

    # DÜZELTME: Değerleri de döndür
    return karar, int(nan_h), int(sifir), int(neg)


# ==================================================================
# ANA
# ==================================================================
def main():
    print("=" * 80)
    print("H8 INVENTORY — 1X2 HOME PRICE LEVEL (B365H)")
    print("2026/27 (E0 (10).csv) KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    # 1. Sütun varlık
    per_sezon, tum_b365h = sutun_varlik()

    # 2. Veri yükleme + filtre
    df = veri_yukle()

    # 3. H8 değişkeni
    df = h8_degisken_analiz(df)

    # 4. Elo kontrolü
    elo_kontrol(df)

    # 5. Bütünlük
    veri_butunlugu(df)

    # 6. Karar matrisi
    # DÜZELTME: 4 değer döndürülüyor
    karar, nan_h_val, sifir_val, neg_val = karar_matrisi(df, tum_b365h)

    # -------- EXCEL --------
    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:

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
                "B365H": "VAR" if "B365H" in cols else "YOK",
            })
        pd.DataFrame(rows).to_excel(w, sheet_name="Sutun_Varlik", index=False)

        d = _describe(df["B365H"])
        pd.DataFrame([{"Metrik": k, "Deger": v} for k, v in d.items()]).to_excel(
            w, sheet_name="B365H_Dagilimi", index=False)

        sezon_rows = []
        for sezon in sorted(df["Season"].unique()):
            alt = df[df["Season"] == sezon]
            d = _describe(alt["B365H"])
            sezon_rows.append({"Sezon": sezon, **d})
        pd.DataFrame(sezon_rows).to_excel(w, sheet_name="B365H_Sezon", index=False)

        d_elo = _describe(df["Elo_Diff"])
        pd.DataFrame([{"Metrik": k, "Deger": v} for k, v in d_elo.items()]).to_excel(
            w, sheet_name="Elo_Diff_Dagilimi", index=False)

        # DÜZELTME: nan_h_val, sifir_val, neg_val kullanılıyor
        pd.DataFrame([{
            "Protokol": "H8 INVENTORY",
            "Tarih": "2026-10-08",
            "Amaç": "B365H veri uygulanabilirliği",
            "Kapsam": "E0 (4) - E0 (9)",
            "E0_10": "KILITLI - okunmadi",
            "Model_kurulmadi": "EVET",
            "Esik_aranmadi": "EVET",
            "Yeni_filtre_kurali": "EKLENMEDI",
            "KARAR": karar,
            "B365H_tum_sezonlarda": tum_b365h,
            "Kritik_NaN": nan_h_val,
            "Sifir_veya_negatif": sifir_val + neg_val,
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} yazıldı.")
    print(f"\n{'='*80}")
    print(f"SON KARAR: {karar}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()