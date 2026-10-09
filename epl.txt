# -*- coding: utf-8 -*-
"""
csv_to_xlsx.py
EPL CSV → XLSX dönüştürücü

Girdi:  E0.csv … E0 (9).csv (10 sezon, 2015/16 – 2024/25)
Çıktı:  EPL_tum_sezonlar.xlsx (10 sheet, her sezon bir sheet)

KİLİTLER:
- E0 (10).csv (2026/27) ASLA OKUNMAZ
- Orijinal CSV'ler değiştirilmez
- Sadece okuma + XLSX yazma
"""

import os
import pandas as pd

# ==================================================================
# 0. SABİTLER
# ==================================================================
YASAKLI_DOSYALAR = {"E0 (10).csv"}

DOSYALAR = [
    ("E0.csv",       "2015-16"),
    ("E0 (1).csv",   "2016-17"),
    ("E0 (2).csv",   "2017-18"),
    ("E0 (3).csv",   "2018-19"),
    ("E0 (4).csv",   "2019-20"),
    ("E0 (5).csv",   "2020-21"),
    ("E0 (6).csv",   "2021-22"),
    ("E0 (7).csv",   "2022-23"),
    ("E0 (8).csv",   "2023-24"),
    ("E0 (9).csv",   "2024-25"),
]

CIKTI = "EPL_tum_sezonlar.xlsx"


# ==================================================================
# 1. YARDIMCILAR
# ==================================================================
def _guvenli_oku(dosya):
    ad = os.path.basename(dosya)
    if ad in YASAKLI_DOSYALAR:
        raise RuntimeError(f"YASAKLI DOSYA OKUNAMAZ: {ad}")
    return pd.read_csv(dosya, encoding="utf-8-sig")


# ==================================================================
# 2. ANA
# ==================================================================
def main():
    print("=" * 80)
    print("EPL CSV → XLSX DÖNÜŞTÜRÜCÜ")
    print("E0 (10).csv (2026/27) ASLA OKUNMAZ.")
    print("=" * 80)

    # Kontrol: tüm dosyalar var mı?
    print("\n--- Dosya kontrolü ---")
    eksik = []
    for dosya, sezon in DOSYALAR:
        if os.path.exists(dosya):
            print(f"  {dosya:15s} → {sezon} ✅")
        else:
            print(f"  {dosya:15s} → {sezon} ❌ EKSİK")
            eksik.append(dosya)

    if eksik:
        raise RuntimeError(f"Eksik dosyalar: {eksik}")

    # Yasaklı dosya var mı?
    print("\n--- Yasaklı dosya kontrolü ---")
    for yasak in YASAKLI_DOSYALAR:
        if os.path.exists(yasak):
            print(f"  {yasak} → mevcut ama OKUNMAYACAK ✅")
        else:
            print(f"  {yasak} → yok (sorun değil)")

    # XLSX yaz
    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        for dosya, sezon in DOSYALAR:
            df = _guvenli_oku(dosya)
            print(f"  {dosya:15s} → sheet '{sezon}' ({len(df)} satır, {len(df.columns)} sütun)")
            df.to_excel(w, sheet_name=sezon, index=False)

    # Kontrol: çıktı dosyası oluştu mu?
    if os.path.exists(CIKTI):
        boyut_mb = os.path.getsize(CIKTI) / (1024 * 1024)
        print(f"\n✅ {CIKTI} oluşturuldu ({boyut_mb:.2f} MB)")
    else:
        raise RuntimeError("XLSX dosyası oluşturulamadı.")

    # Sheet listesi
    print(f"\n--- Sheet listesi ---")
    xls = pd.ExcelFile(CIKTI)
    for s in xls.sheet_names:
        df = pd.read_excel(CIKTI, sheet_name=s)
        print(f"  {s}: {len(df)} satır, {len(df.columns)} sütun")

    print("\n" + "=" * 80)
    print("DÖNÜŞTÜRME TAMAMLANDI")
    print("=" * 80)
    print(f"  Girdi:  10 CSV dosyası")
    print(f"  Çıktı:  {CIKTI}")
    print(f"  Sheet:  {len(DOSYALAR)} (her sezon bir sheet)")
    print(f"  E0 (10).csv: OKUNMADI ✅")
    print("=" * 80)


if __name__ == "__main__":
    main()