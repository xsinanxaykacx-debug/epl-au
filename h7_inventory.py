# -*- coding: utf-8 -*-
"""
h7_inventory.py
H7 1X2 Home Movement — Veri Envanteri

AMAÇ:
- B365H ve B365HC sütunlarının varlığını, tutarlılığını, dağılımını belgele.
- X12_HomeMovement = B365HC - B365H değişkeninin veri olarak
  uygulanabilir olup olmadığını doğrula.

YAPMAZ:
- Model kurmaz
- Eşik aramaz
- Değişken tanımı değiştirmez
- Sonuç/edge aramaz
- H7'yi FROZEN etmez
- 2026/27'yi okumaz
- Kısmi PASS veya "yeni filtre kuralı ekle" seçeneği YOK

KARAR MATRİSİ (protokol önceden belirlemiştir):
- B365H ve B365HC tüm 6 sezonda var + NaN yok + tüm fiyatlar > 0 → PASS
- B365H herhangi bir sezonda yok → FAIL
- B365HC herhangi bir sezonda yok → FAIL
- Kritik NaN varsa → FAIL
- Sıfır/negatif fiyat varsa → FAIL
- Movement = 0 olan maçlar → FAIL değil, geçerli gözlem

KİLİTLER:
- Yalnızca E0 (4) – E0 (9) okunur
- E0 (10).csv için güvenlik + sayaç
- Sadece H7 değişkenleri taranır: B365H, B365HC
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
CIKTI = "h7_inventory.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

MARKET_DOSYALARI = {
    "2019/20": "E0 (4).csv",
    "2020/21": "E0 (5).csv",
    "2021/22": "E0 (6).csv",
    "2022/23": "E0 (7).csv",
    "2023/24": "E0 (8).csv",
    "2024/25": "E0 (9).csv",
}

H7_SUTUNLAR = ["Season", "Date", "HomeTeam", "AwayTeam", "B365H", "B365HC"]

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
    print("1. SÜTUN VARLIK KONTROLÜ (B365H, B365HC)")
    print("=" * 80)

    per_sezon = {}
    for sezon, dosya in MARKET_DOSYALARI.items():
        if not os.path.exists(dosya):
            print(f"  [YOK] {dosya}")
            per_sezon[sezon] = {"B365H": False, "B365HC": False, "n_satir": 0}
            continue
        df = _guvenli_oku(dosya)
        cols = set(df.columns)
        per_sezon[sezon] = {
            "n_satir": len(df),
            "n_sutun": len(df.columns),
            "B365H": "B365H" in cols,
            "B365HC": "B365HC" in cols,
            "B365A": "B365A" in cols,
            "B365AC": "B365AC" in cols,
            "B365D": "B365D" in cols,
            "B365DC": "B365DC" in cols,
        }
        print(f"  {dosya:15s} → {sezon}  ({len(df)} satır × {len(df.columns)} sütun)")
        print(f"    B365H : {'VAR' if per_sezon[sezon]['B365H'] else 'YOK'}")
        print(f"    B365HC: {'VAR' if per_sezon[sezon]['B365HC'] else 'YOK'}")

    print(f"\n  --- Özet: H7 kritik sütunlar ---")
    for s in ["B365H", "B365HC"]:
        satir = "  ".join(f"{sezon}:{'V' if per_sezon[sezon][s] else 'X'}"
                          for sezon in per_sezon)
        sayi = sum(1 for sezon in per_sezon if per_sezon[sezon][s])
        durum = "[TÜMÜ]" if sayi == len(per_sezon) else f"[{sayi}/{len(per_sezon)}]"
        print(f"    {s:10s} {durum}  {satir}")

    # KRİTİK KARAR: B365H ve B365HC tüm sezonlarda var mı?
    tum_b365h = all(per_sezon[s]["B365H"] for s in per_sezon)
    tum_b365hc = all(per_sezon[s]["B365HC"] for s in per_sezon)

    print(f"\n  --- KRİTİK KONTROL ---")
    if not tum_b365h:
        print(f"    ❌ B365H bazı sezonlarda YOK → H7 FAIL")
    else:
        print(f"    ✅ B365H tüm sezonlarda VAR")

    if not tum_b365hc:
        print(f"    ❌ B365HC bazı sezonlarda YOK → H7 FAIL")
        print(f"       Protokol DEĞİŞMEZ. Başka sütun aranmaz.")
    else:
        print(f"    ✅ B365HC tüm sezonlarda VAR")

    # Alternatif sütunlar bilgi amaçlı
    print(f"\n  --- Bilgi amaçlı: Alternatif 1X2 sütunları (KULLANILMAYACAK) ---")
    for s in ["B365A", "B365AC", "B365D", "B365DC"]:
        satir = "  ".join(f"{sezon}:{'V' if per_sezon[sezon].get(s, False) else 'X'}"
                          for sezon in per_sezon)
        print(f"    {s:10s}  {satir}")

    return per_sezon, tum_b365h, tum_b365hc


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
        eksik = [s for s in H7_SUTUNLAR if s not in df.columns]
        if eksik:
            raise RuntimeError(f"{ad} içinde eksik sütunlar: {eksik}")
        frames.append(df[H7_SUTUNLAR].copy())

    market = pd.concat(frames, ignore_index=True)
    print(f"\n  Market satırı (ham): {len(market)}")

    if not os.path.exists(H1_FEATURES):
        raise RuntimeError(f"{H1_FEATURES} bulunamadı.")
    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")
    feat["_key"] = _mac_anahtari(feat)
    market["_key"] = _mac_anahtari(market)

    df = feat.merge(market[["_key", "B365H", "B365HC"]], on="_key", how="inner")
    print(f"  Merge sonrası: {len(df)}")

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
# 4. H7 DEĞİŞKEN ANALİZİ
# ==================================================================
def h7_degisken_analiz(df):
    print("\n" + "=" * 80)
    print("3. H7 DEĞİŞKENİ: X12_HomeMovement = B365HC - B365H")
    print("=" * 80)

    # NaN kontrolü
    print("\n  --- NaN kontrolü ---")
    nan_h = int(df["B365H"].isna().sum())
    nan_hc = int(df["B365HC"].isna().sum())
    print(f"    B365H  : NaN={nan_h} / {len(df)}")
    print(f"    B365HC : NaN={nan_hc} / {len(df)}")

    # Sıfır/negatif kontrolü
    print("\n  --- Sıfır ve negatif kontrolü ---")
    for c in ["B365H", "B365HC"]:
        sifir = int((df[c] == 0).sum())
        negatif = int((df[c] < 0).sum())
        cok_dusuk = int(((df[c] > 0) & (df[c] < 1.0)).sum())
        print(f"    {c:10s}: sıfır={sifir}, negatif={negatif}, 0<x<1={cok_dusuk}")

    # Movement
    df["X12_HomeMovement"] = df["B365HC"] - df["B365H"]

    print(f"\n  --- X12_HomeMovement dağılımı ---")
    d = _describe(df["X12_HomeMovement"])
    for k, v in d.items():
        print(f"    {k:6s}: {v}")

    print(f"\n  --- Sezon bazında X12_HomeMovement ---")
    for sezon in sorted(df["Season"].unique()):
        alt = df[df["Season"] == sezon]
        d = _describe(alt["X12_HomeMovement"])
        print(f"    {sezon} (N={d['N']}): min={d['Min']}, P50={d['P50']}, "
              f"max={d['Max']}, mean={d['Mean']}")

    # Movement = 0
    sifir_movement = int((df["X12_HomeMovement"] == 0).sum())
    print(f"\n  --- Movement = 0 (B365HC = B365H) ---")
    print(f"    Sayı: {sifir_movement} / {len(df)} (%{sifir_movement/len(df)*100:.2f})")
    print(f"    NOT: Bu gözlemler GEÇERLİDİR. Filtrelenmez.")

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
    for c in ["B365H", "B365HC", "X12_HomeMovement", "Elo_Diff"]:
        if c in df.columns:
            nn = int(df[c].isna().sum())
            print(f"    {c:22s}: NaN={nn}")

    print("\n--- Dosya okuma sayacı ---")
    for ad in sorted(_okuma_sayaci.keys()):
        print(f"  {ad}: READ {_okuma_sayaci[ad]} TIMES")
    for ad in YASAKLI_DOSYALAR:
        print(f"  [YASAKLI] {ad}: READ {_okuma_sayaci.get(ad, 0)} TIMES")


# ==================================================================
# 7. KARAR MATRİSİ (PROTOKOL ÖNCEDEN BELİRLEMİŞTİR)
# ==================================================================
def karar_matrisi(df, tum_b365h, tum_b365hc):
    print("\n" + "=" * 80)
    print("6. KARAR MATRİSİ (H7 FROZEN PROTOKOL)")
    print("=" * 80)

    print("\n  Kriterler:")
    print("  - B365H tüm sezonlarda var mı?")
    print("  - B365HC tüm sezonlarda var mı?")
    print("  - Kritik NaN var mı?")
    print("  - Sıfır/negatif fiyat var mı?")
    print("  - (Movement = 0 FAIL değildir)")

    print("\n  --- Kontroller ---")
    print(f"    B365H tüm sezonlarda     : {'✅' if tum_b365h else '❌'}")
    print(f"    B365HC tüm sezonlarda    : {'✅' if tum_b365hc else '❌'}")

    nan_h = df["B365H"].isna().sum()
    nan_hc = df["B365HC"].isna().sum()
    nan_ok = (nan_h == 0) and (nan_hc == 0)
    print(f"    Kritik NaN yok           : {'✅' if nan_ok else '❌'} "
          f"(B365H NaN={nan_h}, B365HC NaN={nan_hc})")

    sifir_h = (df["B365H"] == 0).sum()
    sifir_hc = (df["B365HC"] == 0).sum()
    neg_h = (df["B365H"] < 0).sum()
    neg_hc = (df["B365HC"] < 0).sum()
    fiyat_ok = (sifir_h == 0) and (sifir_hc == 0) and (neg_h == 0) and (neg_hc == 0)
    print(f"    Sıfır/negatif fiyat yok  : {'✅' if fiyat_ok else '❌'}")

    # KARAR
    if tum_b365h and tum_b365hc and nan_ok and fiyat_ok:
        karar = "PASS → Explore"
        print(f"\n  ✅ KARAR: {karar}")
    else:
        karar = "FAIL → H7 KAPANIR"
        print(f"\n  ❌ KARAR: {karar}")
        print(f"     Protokol DEĞİŞMEZ.")
        print(f"     Yeni filtre kuralı EKLENMEZ.")
        print(f"     Başka sütun ARANMAZ.")

    return karar


# ==================================================================
# ANA
# ==================================================================
def main():
    print("=" * 80)
    print("H7 INVENTORY — 1X2 HOME MOVEMENT (B365)")
    print("2026/27 (E0 (10).csv) KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    # 1. Sütun varlık
    per_sezon, tum_b365h, tum_b365hc = sutun_varlik()

    # 2. Veri yükleme + filtre
    df = veri_yukle()

    # 3. H7 değişkeni
    df = h7_degisken_analiz(df)

    # 4. Elo kontrolü
    elo_kontrol(df)

    # 5. Bütünlük
    veri_butunlugu(df)

    # 6. Karar matrisi
    karar = karar_matrisi(df, tum_b365h, tum_b365hc)

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
                "B365H": "VAR" if "B365H" in cols else "YOK",
                "B365HC": "VAR" if "B365HC" in cols else "YOK",
            })
        pd.DataFrame(rows).to_excel(w, sheet_name="Sutun_Varlik", index=False)

        # Movement dağılımı
        d = _describe(df["X12_HomeMovement"])
        pd.DataFrame([{"Metrik": k, "Deger": v} for k, v in d.items()]).to_excel(
            w, sheet_name="Movement_Dagilimi", index=False)

        # Sezon bazında
        sezon_rows = []
        for sezon in sorted(df["Season"].unique()):
            alt = df[df["Season"] == sezon]
            d = _describe(alt["X12_HomeMovement"])
            sezon_rows.append({"Sezon": sezon, **d})
        pd.DataFrame(sezon_rows).to_excel(w, sheet_name="Movement_Sezon", index=False)

        # Tüm dağılımlar
        dagilim_rows = []
        for c in ["B365H", "B365HC", "X12_HomeMovement", "Elo_Diff"]:
            d = _describe(df[c])
            for k, v in d.items():
                dagilim_rows.append({"Sutun": c, "Metrik": k, "Deger": v})
        pd.DataFrame(dagilim_rows).to_excel(w, sheet_name="Tum_Dagilim", index=False)

        # Notlar
        pd.DataFrame([{
            "Protokol": "H7 INVENTORY",
            "Tarih": "2026-10-08",
            "Amaç": "B365HC - B365H veri uygulanabilirliği",
            "Kapsam": "E0 (4) - E0 (9)",
            "E0_10": "KILITLI - okunmadi",
            "Model_kurulmadi": "EVET",
            "Esik_aranmadi": "EVET",
            "Yeni_filtre_kurali": "EKLENMEDI",
            "KARAR": karar,
            "B365H_tum_sezonlarda": tum_b365h,
            "B365HC_tum_sezonlarda": tum_b365hc,
            "Kritik_NaN": int(nan_h + nan_hc),
            "Sifir_veya_negatif": int(sifir_h + sifir_hc + neg_h + neg_hc),
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} yazıldı.")
    print(f"\n{'='*80}")
    print(f"SON KARAR: {karar}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()