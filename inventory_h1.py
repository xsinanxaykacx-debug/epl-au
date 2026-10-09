# -*- coding: utf-8 -*-
"""
inventory_h1.py
H1 Veri Envanteri — Mevcut E0*.csv dosyalarında ne var, H1 için ne üretilebilir?

Kurallar:
- 2026/27 (E0 (10).csv) KESİNLİKLE OKUNMAZ.
- Hiçbir sinyal üretilmez, ROI hesaplanmaz.
- Sadece envanter + look-ahead kontrolü + H1 uygunluk raporu.
"""

import glob
import os
import numpy as np
import pandas as pd

# ==================================================================
# 0. SABİTLER
# ==================================================================
YASAKLI_DOSYALAR = {"E0 (10).csv"}   # 2026/27 — ASLA OKUNMAZ

# Dosya → Sezon eşlemesi (envanter_v2.py ile aynı)
DOSYA_SEZON = {
    "E0.csv":       "2015/16",
    "E0 (1).csv":   "2016/17",
    "E0 (2).csv":   "2017/18",
    "E0 (3).csv":   "2018/19",
    "E0 (4).csv":   "2019/20",
    "E0 (5).csv":   "2020/21",
    "E0 (6).csv":   "2021/22",
    "E0 (7).csv":   "2022/23",
    "E0 (8).csv":   "2023/24",
    "E0 (9).csv":   "2024/25",
    "E0 (10).csv":  "2026/27",   # YASAKLI
}

# H1 için aday değişkenler (aranacak)
H1_HAZIR_DEGISKENLER = {
    "Elo":              ["elo", "rating", "elo_home", "elo_away", "home_elo", "away_elo"],
    "Lig sırası":       ["rank", "position", "table", "standing", "home_rank", "away_rank"],
    "xG":               ["xg", "xG", "expected_goals", "home_xg", "away_xg"],
    "Şut/İsabet":       ["shots", "shots_on_target", "sot", "hs", "as_", "hst", "ast"],
    "Kadro değeri":     ["squad_value", "market_value", "transfer_value"],
    "Sakatlık/Ceza":    ["injury", "suspension", "missing", "unavailable"],
    "Sonuç/Goal":       ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"],
    "Market (AH)":      ["AHh", "AHCh", "B365AHH", "B365AHA", "B365CAHH", "B365CAHA"],
    "Market (O/U)":     ["B365>2.5", "B365<2.5", "B365C>2.5", "B365C<2.5"],
    "Fikstür":          ["Date", "HomeTeam", "AwayTeam"],
}

# H1 için türetilebilir değişkenler ve gereken temel sütunlar
H1_TURETILEBILIR = {
    "Son 3 maç puanı":         {"gerekli": ["Date", "HomeTeam", "AwayTeam", "FTR"]},
    "Son 5 maç puanı":         {"gerekli": ["Date", "HomeTeam", "AwayTeam", "FTR"]},
    "Son 10 maç puanı":        {"gerekli": ["Date", "HomeTeam", "AwayTeam", "FTR"]},
    "Ev/deplasman formu":      {"gerekli": ["Date", "HomeTeam", "AwayTeam", "FTR"]},
    "Gol atma ortalaması":     {"gerekli": ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]},
    "Gol yeme ortalaması":     {"gerekli": ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]},
    "Averaj trendi":           {"gerekli": ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]},
    "Rakip gücü (dolaylı)":    {"gerekli": ["Date", "HomeTeam", "AwayTeam", "FTR"]},
    "Maç arası gün":           {"gerekli": ["Date", "HomeTeam", "AwayTeam"]},
    "Basit Elo (sonuçlardan)": {"gerekli": ["Date", "HomeTeam", "AwayTeam", "FTR"]},
}


# ==================================================================
# 1. DOSYA TARAMA
# ==================================================================
def dosyalari_tara():
    """Dizindeki E0*.csv dosyalarını listeler; 2026/27'yi ATLAR."""
    bulunanlar = []
    for f in sorted(glob.glob("E0*.csv")):
        ad = os.path.basename(f)
        if ad in YASAKLI_DOSYALAR:
            print(f"  [ATLANDI] {ad} — yasaklı (2026/27)")
            continue
        sezon = DOSYA_SEZON.get(ad, "?")
        bulunanlar.append((ad, sezon))
    return bulunanlar


# ==================================================================
# 2. SÜTUN VARLIK KONTROLÜ
# ==================================================================
def _normalize(s):
    return s.lower().replace("_", "").replace(" ", "")


def sutun_varlik(df_cols):
    """Her H1 kategorisi için CSV'de bulunan/eşleşen sütunları döner."""
    norm_cols = {_normalize(c): c for c in df_cols}
    sonuc = {}
    for kategori, adaylar in H1_HAZIR_DEGISKENLER.items():
        bulunan = []
        for a in adaylar:
            na = _normalize(a)
            if na in norm_cols:
                bulunan.append(norm_cols[na])
        sonuc[kategori] = bulunan
    return sonuc


# ==================================================================
# 3. TÜRETİLEBİLİRLİK KONTROLÜ
# ==================================================================
def turetilebilirlik_kontrol(df_cols):
    """Her türetilebilir değişken için gerekli sütunlar var mı?"""
    sonuc = {}
    for degisken, kural in H1_TURETILEBILIR.items():
        eksik = [s for s in kural["gerekli"] if s not in df_cols]
        sonuc[degisken] = {
            "gerekli": kural["gerekli"],
            "eksik": eksik,
            "turetilebilir": len(eksik) == 0,
        }
    return sonuc


# ==================================================================
# 4. LOOK-AHEAD KONTROLÜ
# ==================================================================
def lookahead_kontrol(df, sezon):
    """
    Türetilebilir değişkenler için maç öncesi güvenlik kontrolü.
    Kritik: aynı maçın sonucu, aynı maçın değişkeni olarak kullanılmamalı.
    """
    # Tarih sırası kontrolü
    try:
        tarih = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
        sirali = tarih.is_monotonic_increasing
    except Exception:
        sirali = None

    # Aynı takım aynı gün iki maç?
    try:
        gunluk = df.groupby(["Date", "HomeTeam"]).size()
        ayni_gun = (gunluk > 1).sum()
    except Exception:
        ayni_gun = None

    return {
        "sezon": sezon,
        "tarih_sirali": sirali,
        "ayni_gun_cakismasi": int(ayni_gun) if ayni_gun is not None else None,
    }


# ==================================================================
# 5. EKSİK VERİ ORANI
# ==================================================================
def eksik_veri(df, sutunlar):
    """Verilen sütunlar için eksik veri oranı."""
    n = len(df)
    out = {}
    for s in sutunlar:
        if s in df.columns:
            out[s] = round(df[s].isna().sum() / n * 100, 2)
    return out


# ==================================================================
# 6. ANA AKIŞ
# ==================================================================
def main():
    print("=" * 80)
    print("H1 VERİ ENVANTERİ — E0*.csv taraması")
    print("2026/27 (E0 (10).csv) KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    dosyalar = dosyalari_tara()
    print(f"\nTaranacak dosya: {len(dosyalar)}")

    # Tüm dosyaların sütunlarını topla (birleşik)
    tum_sutunlar = set()
    per_sezon_sutunlar = {}
    per_sezon_eksik = {}

    print("\n--- Dosyalar okunuyor ---")
    for dosya, sezon in dosyalar:
        if dosya in YASAKLI_DOSYALAR:
            continue
        try:
            df = pd.read_csv(dosya, encoding="utf-8-sig", nrows=5)  # sadece başlık + birkaç satır
            sutunlar = set(df.columns)
            tum_sutunlar.update(sutunlar)
            per_sezon_sutunlar[sezon] = sutunlar
            print(f"  {dosya:15s} → {sezon}  ({len(sutunlar)} sütun)")
        except Exception as e:
            print(f"  {dosya} HATA: {e}")

    # 2025/26 kontrolü
    print("\n--- 2025/26 kontrolü ---")
    if "2025/26" not in per_sezon_sutunlar:
        print("  2025/26 → DOSYA YOK (beklenen)")
    else:
        print("  2025/26 → mevcut")

    # Tam veri (eksik veri oranı için) — sadece hedef sütunlar
    print("\n--- Tam veri yükleniyor (eksik oran için) ---")
    tum_df = []
    for dosya, sezon in dosyalar:
        if dosya in YASAKLI_DOSYALAR:
            continue
        try:
            df = pd.read_csv(dosya, encoding="utf-8-sig")
            df["Season"] = sezon
            tum_df.append(df)
        except Exception:
            pass
    df_all = pd.concat(tum_df, ignore_index=True) if tum_df else pd.DataFrame()
    print(f"  Toplam: {len(df_all)} satır")

    # ==========================================================
    # 7. H1 HAZIR DEĞİŞKENLER (CSV'de var mı?)
    # ==========================================================
    print("\n" + "=" * 80)
    print("BÖLÜM 1 — H1 İÇİN DOĞRUDAN DEĞİŞKENLER (CSV'de mevcut mu?)")
    print("=" * 80)

    varlik = sutun_varlik(tum_sutunlar)
    for kategori, bulunanlar in varlik.items():
        durum = "VAR" if bulunanlar else "YOK"
        print(f"\n  {kategori}: [{durum}]")
        if bulunanlar:
            for s in bulunanlar:
                print(f"      - {s}")

    # ==========================================================
    # 8. TÜRETİLEBİLİRLİK KONTROLÜ
    # ==========================================================
    print("\n" + "=" * 80)
    print("BÖLÜM 2 — H1 İÇİN TÜRETİLEBİLİR DEĞİŞKENLER")
    print("=" * 80)

    turet = turetilebilirlik_kontrol(tum_sutunlar)
    for degisken, bilgi in turet.items():
        durum = "ÜRETİLEBİLİR" if bilgi["turetilebilir"] else "ÜRETİLEMEZ"
        print(f"\n  {degisken}: [{durum}]")
        if bilgi["eksik"]:
            print(f"      Eksik sütunlar: {bilgi['eksik']}")
        else:
            print(f"      Gerekli sütunlar mevcut: {bilgi['gerekli']}")

    # ==========================================================
    # 9. LOOK-AHEAD KONTROLÜ
    # ==========================================================
    print("\n" + "=" * 80)
    print("BÖLÜM 3 — LOOK-AHEAD GÜVENLİK KONTROLÜ")
    print("=" * 80)

    print("\n  Kural: Türetilmiş değişkenler YALNIZCA maçtan ÖNCEKİ bilgiden hesaplanmalı.")
    print("  Bu, her sezon için tarih sırasının ve maç kimliklerinin doğrulanmasını gerektirir.")

    for dosya, sezon in dosyalar:
        if dosya in YASAKLI_DOSYALAR:
            continue
        try:
            df = pd.read_csv(dosya, encoding="utf-8-sig")
            la = lookahead_kontrol(df, sezon)
            print(f"\n  {sezon}:")
            print(f"      Tarih sıralı mı: {la['tarih_sirali']}")
            print(f"      Aynı gün ev sahibi çakışması: {la['ayni_gun_cakismasi']}")
        except Exception:
            pass

    # ==========================================================
    # 10. EKSİK VERİ ORANI
    # ==========================================================
    print("\n" + "=" * 80)
    print("BÖLÜM 4 — H1 İÇİN ÖNEMLİ SÜTUNLARIN EKSİK VERİ ORANI (%)")
    print("=" * 80)

    hedef_sutunlar = []
    for kategori, adaylar in H1_HAZIR_DEGISKENLER.items():
        for a in adaylar:
            if a in df_all.columns:
                hedef_sutunlar.append(a)

    if hedef_sutunlar and "Season" in df_all.columns:
        print("\n  Sezon bazında eksik oran (%):")
        for sezon in sorted(df_all["Season"].unique()):
            alt = df_all[df_all["Season"] == sezon]
            eksikler = eksik_veri(alt, hedef_sutunlar)
            print(f"\n  {sezon}:")
            for s, o in eksikler.items():
                print(f"      {s:15s} → %{o}")

    # ==========================================================
    # 11. H1 UYGUNLUK RAPORU
    # ==========================================================
    print("\n" + "=" * 80)
    print("BÖLÜM 5 — H1 UYGUNLUK RAPORU")
    print("=" * 80)

    rapor = []
    for kategori, adaylar in H1_HAZIR_DEGISKENLER.items():
        bulunan = [a for a in adaylar if a in tum_sutunlar]
        rapor.append({
            "Değişken": kategori,
            "CSV'de mevcut": "EVET" if bulunan else "HAYIR",
            "Türetilebilir": "—",  # Aşağıda doldurulacak
            "Maç öncesi güvenli": "—",
            "Sezon kapsamı": "—",
            "H1 için": "—",
        })

    for degisken, bilgi in turet.items():
        rapor.append({
            "Değişken": degisken,
            "CSV'de mevcut": "HAYIR",
            "Türetilebilir": "EVET" if bilgi["turetilebilir"] else "HAYIR",
            "Maç öncesi güvenli": "EVET" if bilgi["turetilebilir"] else "—",
            "Sezon kapsamı": "2015/16–2024/25",
            "H1 için": "UYGUN" if bilgi["turetilebilir"] else "YETERSİZ",
        })

    rapor_df = pd.DataFrame(rapor)
    print("\n" + rapor_df.to_string(index=False))

    # ==========================================================
    # 12. EXCEL ÇIKTISI
    # ==========================================================
    cikti = "inventory_h1.xlsx"
    with pd.ExcelWriter(cikti, engine="openpyxl") as w:
        rapor_df.to_excel(w, sheet_name="H1_Uygunluk", index=False)

        # Doğrudan değişkenler
        pd.DataFrame([{
            "Kategori": k, "Bulunan_sutunlar": ", ".join(v) if v else "YOK"
        } for k, v in varlik.items()]).to_excel(w, sheet_name="Dogrudan_Degiskenler", index=False)

        # Türetilebilirlik
        pd.DataFrame([{
            "Degisken": k, "Turetilebilir": v["turetilebilir"],
            "Gerekli": ", ".join(v["gerekli"]),
            "Eksik": ", ".join(v["eksik"]) if v["eksik"] else "—"
        } for k, v in turet.items()]).to_excel(w, sheet_name="Turetilebilirlik", index=False)

        # Sütun listesi (sezon bazlı)
        rows = []
        for sezon, sutunlar in per_sezon_sutunlar.items():
            for s in sorted(sutunlar):
                rows.append({"Sezon": sezon, "Sutun": s})
        pd.DataFrame(rows).to_excel(w, sheet_name="Sezon_Sutunlar", index=False)

        # Notlar
        pd.DataFrame([{
            "Protokol": "H1 Veri Envanteri (henüz FROZEN değil)",
            "Tarih": "2026-10-08",
            "Taranan": "2015/16 – 2024/25",
            "Atlanan": "E0 (10).csv (2026/27) — YASAKLI",
            "Eksik_sezon": "2025/26 — dosya yok",
            "Amaç": "H1 için mevcut veri kapsamı ve türetilebilirlik analizi",
            "Not": "Bu script H1'i test etmedi, sinyal üretmedi, ROI hesaplamadı.",
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{cikti} yazıldı.")
    print("\nÖzet:")
    print(f"  Taranan dosya: {len(dosyalar)}")
    print(f"  Toplam satır: {len(df_all)}")
    print(f"  H1 için doğrudan mevcut değişken kategorisi: {sum(1 for v in varlik.values() if v)}")
    print(f"  H1 için türetilebilir değişken: {sum(1 for v in turet.values() if v['turetilebilir'])}")


if __name__ == "__main__":
    main()