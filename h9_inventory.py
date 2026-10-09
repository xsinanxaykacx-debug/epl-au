# -*- coding: utf-8 -*-
"""
h9_inventory.py
H9 Inventory — Veri Uygunluğu Kontrolü

AMAÇ:
- h1_features_v2.csv + E0*.csv birleşik veri uygunluğunu kontrol et.
- 49 sayısal pre-match feature'ın varlığını doğrula.
- B365H, B365D, B365A kontrolü (FTR zaten h1_features_v2'de).
- AH ailesi eksik maç filtresi.
- Train/Val/OOS satır sayıları (filtre sonrası gerçek).

YAPMAZ:
- Model fit etmez
- Value hesaplamaz
- Threshold uygulamaz
- Validation/OOS sonuç incelemez
- 2026/27'yi okumaz

KATI BÜTÜNLÜK KONTROLLERİ:
1. 10 tarihsel CSV mevcut ve her biri 380 satır (yoksa RuntimeError).
2. h1_features_v2.csv 10 sezon × 380 = 3800 satır.
3. Feature sezonları tam ve doğru (başka sezon yok).
4. Merge öncesi/sonrası 3800.
5. Duplicate key = 0.
6. AH filtresi sonrası N > 0 ve Train/Val/OOS > 0.

KARAR MATRİSİ (mantıksal):
- h1_features_v2 mevcut
- Tam 49 feature
- Tüm feature'lar sayısal
- 10 tarihsel CSV mevcut (her biri 380 satır)
- h1_features_v2 toplam 3800 ve 10 sezon
- Merge öncesi/sonrası 3800
- Duplicate key = 0
- B365 geçerli (>0, NaN yok)
- FTR H/D/A, NaN yok
- AH filtresi sonrası N > 0
- Train/Val/OOS > 0
- E0 (10).csv READ = 0

KİLİTLER:
- Yalnızca E0.csv – E0 (9).csv okunur
- E0 (10).csv için güvenlik + sayaç
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

# ==================================================================
# 0. SABİTLER
# ==================================================================
H1_FEATURES = "h1_features_v2.csv"
ENVANTER = "envanter_v2.xlsx"
ENVANTER_SHEET = "Eksik_Maclar"
CIKTI = "h9_inventory.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

# Kullanılacak 10 dosya (2015/16 – 2024/25)
TUM_DOSYALAR = [
    ("E0.csv",       "2015/16"),
    ("E0 (1).csv",   "2016/17"),
    ("E0 (2).csv",   "2017/18"),
    ("E0 (3).csv",   "2018/19"),
    ("E0 (4).csv",   "2019/20"),
    ("E0 (5).csv",   "2020/21"),
    ("E0 (6).csv",   "2021/22"),
    ("E0 (7).csv",   "2022/23"),
    ("E0 (8).csv",   "2023/24"),
    ("E0 (9).csv",   "2024/25"),
]

# H9 için market sütunları (FTR hariç — h1_features_v2'den geliyor)
MARKET_SUTUNLAR = ["B365H", "B365D", "B365A"]

KIMLIK_SUTUNLAR = ["Season", "Date", "HomeTeam", "AwayTeam"]

# Feature dosyasından çıkarılacak (modele girmeyecek) sütunlar
CIKARILACAK = ["Season", "Date", "HomeTeam", "AwayTeam",
               "FTHG", "FTAG", "FTR", "_key"]

GECERLI_FTR = {"H", "D", "A"}

TRAIN_SEZONLAR = ["2015/16", "2016/17", "2017/18", "2018/19",
                  "2019/20", "2020/21", "2021/22", "2022/23"]
VALIDATION_SEZON = "2023/24"
OOS_SEZON = "2024/25"

# Beklenen ham değerler
BEKLENEN_CSV_SATIR = 380
BEKLENEN_HAM_TOP = 3800
BEKLENEN_HAM_TRAIN = 3040
BEKLENEN_HAM_VAL = 380
BEKLENEN_HAM_OOS = 380

BEKLENEN_SEZONLAR = set(["2015/16", "2016/17", "2017/18", "2018/19",
                          "2019/20", "2020/21", "2021/22", "2022/23",
                          "2023/24", "2024/25"])

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
    # Kaynaklarda hem DD/MM/YYYY hem DD/MM/YY bulunduğundan her satırı
    # bağımsız yorumla. Geçersiz tarihleri "NA" anahtarıyla eşleştirme.
    try:
        parsed = pd.to_datetime(
            df["Date"], dayfirst=True, format="mixed", errors="coerce"
        )
    except (TypeError, ValueError):
        # Eski pandas sürümleri için format bazlı güvenli geri dönüş.
        raw = df["Date"].astype("string").str.strip()
        parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            mask = parsed.isna() & raw.notna()
            if mask.any():
                parsed.loc[mask] = pd.to_datetime(raw.loc[mask], format=fmt, errors="coerce")
    if parsed.isna().any():
        sample = df.loc[parsed.isna(), ["Season", "Date", "HomeTeam", "AwayTeam"]].head(5)
        raise RuntimeError(
            f"Maç anahtarında geçersiz tarih var: {int(parsed.isna().sum())} satır. "
            f"Örnekler:\\n{sample.to_string(index=False)}"
        )
    tarih = parsed.dt.strftime("%Y-%m-%d")
    ev = df["HomeTeam"].astype("string").str.strip().str.replace(r"\\s+", " ", regex=True).str.casefold()
    dep = df["AwayTeam"].astype("string").str.strip().str.replace(r"\\s+", " ", regex=True).str.casefold()
    if (ev.isna() | dep.isna() | ev.eq("") | dep.eq("")).any():
        raise RuntimeError("Maç anahtarında boş ev/deplasman takım adı bulundu.")
    return (
        df["Season"].astype("string").str.strip()
        + "|" + tarih
        + "|" + ev
        + "|" + dep
    )


# ==================================================================
# 2. h1_features_v2.csv KONTROLÜ (KATI)
# ==================================================================
def kontrol_h1_features_v2():
    print("\n" + "=" * 80)
    print("1. h1_features_v2.csv KONTROLÜ (KATI)")
    print("=" * 80)

    if not os.path.exists(H1_FEATURES):
        print(f"  ❌ {H1_FEATURES} bulunamadı.")
        return None, None, None

    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")
    n_top = len(feat)
    print(f"  Toplam satır: {n_top}")
    print(f"  Toplam sütun: {len(feat.columns)}")

    # A) Toplam 3800
    if n_top != BEKLENEN_HAM_TOP:
        raise RuntimeError(f"h1_features_v2.csv satır sayısı {n_top} != {BEKLENEN_HAM_TOP}")

    # B) 10 sezon × 380
    print(f"\n  Sezon dağılımı:")
    sezon_sayim = feat.groupby("Season").size().to_dict()
    for s in sorted(sezon_sayim.keys()):
        n = sezon_sayim[s]
        status = "✅" if n == BEKLENEN_CSV_SATIR else "❌"
        print(f"    {status} {s}: {n}")

    # C) Beklenen sezonlar tam mı?
    feat_sezonlar = set(sezon_sayim.keys())
    eksik_sezon = BEKLENEN_SEZONLAR - feat_sezonlar
    fazla_sezon = feat_sezonlar - BEKLENEN_SEZONLAR
    if eksik_sezon:
        raise RuntimeError(f"h1_features_v2.csv'de eksik sezonlar: {eksik_sezon}")
    if fazla_sezon:
        raise RuntimeError(f"h1_features_v2.csv'de beklenmeyen sezonlar: {fazla_sezon}")

    # D) Her sezon 380 mi?
    hatali_sezon = [s for s, n in sezon_sayim.items() if n != BEKLENEN_CSV_SATIR]
    if hatali_sezon:
        raise RuntimeError(f"Satır sayısı 380 olmayan sezonlar: {hatali_sezon}")

    print(f"  ✅ 10 sezon, her biri 380 satır")

    # Kimlik sütunları
    eksik_kimlik = [c for c in KIMLIK_SUTUNLAR if c not in feat.columns]
    if eksik_kimlik:
        raise RuntimeError(f"Eksik kimlik sütunları: {eksik_kimlik}")

    # FTR kontrolü
    if "FTR" not in feat.columns:
        raise RuntimeError(f"FTR h1_features_v2.csv'de YOK!")
    else:
        print(f"  ✅ FTR h1_features_v2.csv'de VAR")

    # Modele girecek sayısal feature'lar
    feature_sutunlar = [c for c in feat.columns if c not in CIKARILACAK]
    print(f"\n  Kullanılacak sayısal feature sayısı: {len(feature_sutunlar)}")

    # Feature'ların sayısal olduğunu kontrol et
    numerik_olmayan = []
    for c in feature_sutunlar:
        if not pd.api.types.is_numeric_dtype(feat[c]):
            numerik_olmayan.append(c)
    if numerik_olmayan:
        print(f"  ⚠ Sayısal olmayan feature'lar: {numerik_olmayan}")
    else:
        print(f"  ✅ Tüm feature'lar sayısal")

    return feat, feature_sutunlar, numerik_olmayan


# ==================================================================
# 3. MARKET VERİLERİNİ YÜKLE (KATI — HER DOSYA 380 SATIR)
# ==================================================================
def market_yukle():
    print("\n" + "=" * 80)
    print("2. MARKET VERİLERİ (B365H, B365D, B365A)")
    print("=" * 80)

    frames = []
    for dosya, sezon in TUM_DOSYALAR:
        # KRİTİK: Dosya yoksa RuntimeError
        if not os.path.exists(dosya):
            raise RuntimeError(f"EKSİK DOSYA: {dosya} ({sezon})")

        df = _guvenli_oku(dosya)

        # KRİTİK: Satır sayısı 380 mi?
        if len(df) != BEKLENEN_CSV_SATIR:
            raise RuntimeError(
                f"{dosya} ({sezon}) satır sayısı {len(df)} != {BEKLENEN_CSV_SATIR}"
            )

        df["Season"] = sezon

        # Sütun kontrolü
        eksik = [s for s in MARKET_SUTUNLAR if s not in df.columns]
        if eksik:
            raise RuntimeError(f"{dosya} ({sezon}): eksik sütunlar {eksik}")

        df_alt = df[["Season", "Date", "HomeTeam", "AwayTeam"] + MARKET_SUTUNLAR].copy()
        frames.append(df_alt)
        print(f"  ✅ {dosya:15s} → {sezon}  ({len(df)} satır)")

    if not frames:
        raise RuntimeError("Hiç market verisi yüklenemedi.")

    market = pd.concat(frames, ignore_index=True)
    print(f"\n  Toplam market satırı: {len(market)}")

    if len(market) != BEKLENEN_HAM_TOP:
        raise RuntimeError(
            f"Market toplam satır {len(market)} != {BEKLENEN_HAM_TOP}"
        )

    return market


# ==================================================================
# 4. BİRLEŞTİRME VE FİLTRELEME (KATI)
# ==================================================================
def birlestir_ve_filtrele(feat, market):
    print("\n" + "=" * 80)
    print("3. BİRLEŞTİRME VE FİLTRELEME")
    print("=" * 80)

    feat["_key"] = _mac_anahtari(feat)
    market["_key"] = _mac_anahtari(market)

    # Ham N (filtre öncesi)
    ham_n = len(feat)
    print(f"  Ham feature satırı: {ham_n}")
    if ham_n != BEKLENEN_HAM_TOP:
        raise RuntimeError(f"Ham feature satırı {ham_n} != {BEKLENEN_HAM_TOP}")

    # KRİTİK: Merge öncesi kesişim kontrolü
    feat_keys = set(feat["_key"])
    market_keys = set(market["_key"])
    kesisim = feat_keys & market_keys
    print(f"  Feature unique key: {len(feat_keys)}")
    print(f"  Market unique key:  {len(market_keys)}")
    print(f"  Kesişim:            {len(kesisim)}")

    if len(kesisim) != BEKLENEN_HAM_TOP:
        raise RuntimeError(
            f"Feature/Market kesişim {len(kesisim)} != {BEKLENEN_HAM_TOP}"
        )

    # Merge
    df = feat.merge(
        market[["_key", "B365H", "B365D", "B365A"]],
        on="_key", how="inner"
    )
    merge_n = len(df)
    print(f"  Merge sonrası: {merge_n}")

    # KRİTİK: Merge sonrası 3800 olmalı
    if merge_n != BEKLENEN_HAM_TOP:
        raise RuntimeError(f"Merge sonrası {merge_n} != {BEKLENEN_HAM_TOP}")

    # Duplicate kontrolü
    dup = df["_key"].duplicated().sum()
    print(f"  Duplicate key: {dup}")
    if dup != 0:
        raise RuntimeError(f"Duplicate key bulundu: {dup}")

    # AH ailesi eksik filtresi
    eksik = pd.read_excel(ENVANTER, sheet_name=ENVANTER_SHEET)
    eksik["_key"] = _mac_anahtari(eksik)
    eksik_ah = set(eksik.loc[eksik["AH_Eksik"] == True, "_key"].dropna().astype(str))

    df["_eksik_ah"] = df["_key"].isin(eksik_ah)
    onceki = len(df)
    df = df[~df["_eksik_ah"]].copy().reset_index(drop=True)
    sonraki = len(df)
    ah_dislanan = onceki - sonraki
    print(f"  AH ailesi eksik filtresi: {onceki} → {sonraki} (dışlanan: {ah_dislanan})")

    return df, ham_n, merge_n, dup, ah_dislanan


# ==================================================================
# 5. DEĞİŞKEN KONTROLLERİ
# ==================================================================
def degisken_kontrol(df, feature_sutunlar):
    print("\n" + "=" * 80)
    print("4. DEĞİŞKEN KONTROLLERİ")
    print("=" * 80)

    print("\n  --- B365 kontrolü ---")
    for c in ["B365H", "B365D", "B365A"]:
        nn = int(df[c].isna().sum())
        sifir = int((df[c] == 0).sum())
        neg = int((df[c] < 0).sum())
        print(f"    {c:8s}: NaN={nn}, sıfır={sifir}, negatif={neg}")

    print("\n  --- FTR kontrolü ---")
    ftr_degerler = df["FTR"].dropna().unique()
    gecersiz = [v for v in ftr_degerler if v not in GECERLI_FTR]
    print(f"    Geçerli: {GECERLI_FTR}")
    print(f"    Bulunan: {sorted(ftr_degerler)}")
    print(f"    Geçersiz: {gecersiz if gecersiz else 'YOK'}")
    nn_ftr = int(df["FTR"].isna().sum())
    print(f"    FTR NaN: {nn_ftr}")

    print("\n  --- Feature NaN kontrolü ---")
    feature_nan = {}
    for c in feature_sutunlar:
        nn = int(df[c].isna().sum())
        if nn > 0:
            feature_nan[c] = nn
    print(f"    NaN içeren feature sayısı: {len(feature_nan)}")
    if feature_nan:
        print(f"    (Not: NaN'lar median ile doldurulacak — protokol gereği)")
        for c, nn in list(feature_nan.items())[:10]:
            print(f"      {c}: {nn}")

    return feature_nan, gecersiz, nn_ftr


# ==================================================================
# 6. BÖLME KONTROLÜ (FİLTRE SONRASI DİNAMİK)
# ==================================================================
def bolme_kontrol(df):
    print("\n" + "=" * 80)
    print("5. VERİ BÖLMESİ KONTROLÜ (AH filtresi sonrası)")
    print("=" * 80)

    train = df[df["Season"].isin(TRAIN_SEZONLAR)]
    val = df[df["Season"] == VALIDATION_SEZON]
    oos = df[df["Season"] == OOS_SEZON]

    print(f"\n  Train:      {len(train):5d}  (ham beklenti {BEKLENEN_HAM_TRAIN})")
    print(f"  Validation: {len(val):5d}  (ham beklenti {BEKLENEN_HAM_VAL})")
    print(f"  OOS:        {len(oos):5d}  (ham beklenti {BEKLENEN_HAM_OOS})")
    print(f"  Toplam:     {len(df):5d}  (ham beklenti {BEKLENEN_HAM_TOP})")

    print(f"\n  Not: Ham beklenti değerleri filtre ÖNCESİ içindir. AH filtresi")
    print(f"       sonrası N hafif düşebilir. Bu normal ve beklenen bir durumdur.")

    return len(train), len(val), len(oos), len(df)


# ==================================================================
# 7. VERİ BÜTÜNLÜĞÜ KANITI
# ==================================================================
def veri_butunlugu(df):
    print("\n" + "=" * 80)
    print("6. VERİ BÜTÜNLÜĞÜ VE KANIT")
    print("=" * 80)

    print(f"\n  Toplam satır (filtre sonrası): {len(df)}")
    print(f"  Sezon dağılımı:")
    for s in sorted(df["Season"].unique()):
        n = len(df[df["Season"] == s])
        print(f"    {s}: {n}")

    print("\n--- Dosya okuma sayacı ---")
    for ad in sorted(_okuma_sayaci.keys()):
        print(f"  {ad}: READ {_okuma_sayaci[ad]} TIMES")
    for ad in YASAKLI_DOSYALAR:
        print(f"  [YASAKLI] {ad}: READ {_okuma_sayaci.get(ad, 0)} TIMES")


# ==================================================================
# 8. KARAR MATRİSİ (KATI MANTIKSAL)
# ==================================================================
def karar_matrisi(df, feature_sutunlar, numerik_olmayan, gecersiz,
                   nn_ftr, dup, ham_n, merge_n, n_train, n_val, n_oos, n_top):
    print("\n" + "=" * 80)
    print("7. KARAR MATRİSİ (H9 FROZEN PROTOKOL)")
    print("=" * 80)

    print("\n  Kontroller:")

    feat_ok = os.path.exists(H1_FEATURES)
    print(f"    h1_features_v2.csv mevcut       : {'✅' if feat_ok else '❌'}")

    n_feat = len(feature_sutunlar) if feature_sutunlar else 0
    feature_ok = n_feat == 49
    print(f"    Sayısal feature sayısı = 49      : {n_feat} {'✅' if feature_ok else '❌'}")

    numerik_ok = len(numerik_olmayan) == 0
    print(f"    Tüm feature'lar sayısal         : {'✅' if numerik_ok else '❌'}")

    csv_ok = len(TUM_DOSYALAR) == 10
    print(f"    10 tarihsel CSV mevcut          : {'✅' if csv_ok else '❌'}")

    ham_ok = ham_n == BEKLENEN_HAM_TOP
    print(f"    Ham feature toplamı = 3800      : {ham_n} {'✅' if ham_ok else '❌'}")

    merge_ok = merge_n == BEKLENEN_HAM_TOP
    print(f"    Merge sonrası = 3800            : {merge_n} {'✅' if merge_ok else '❌'}")

    dup_ok = dup == 0
    print(f"    Duplicate key = 0               : {'✅' if dup_ok else '❌'}")

    b365_ok = True
    for c in ["B365H", "B365D", "B365A"]:
        if df[c].isna().sum() > 0 or (df[c] <= 0).sum() > 0:
            b365_ok = False
    print(f"    B365 NaN/sıfır/negatif yok      : {'✅' if b365_ok else '❌'}")

    ftr_ok = len(gecersiz) == 0 and nn_ftr == 0
    print(f"    FTR H/D/A, NaN yok              : {'✅' if ftr_ok else '❌'}")

    top_ok = n_top > 0
    train_ok = n_train > 0
    val_ok = n_val > 0
    oos_ok = n_oos > 0
    print(f"    AH filtresi sonrası N > 0       : {n_top} {'✅' if top_ok else '❌'}")
    print(f"    Train N > 0                     : {n_train} {'✅' if train_ok else '❌'}")
    print(f"    Validation N > 0                : {n_val} {'✅' if val_ok else '❌'}")
    print(f"    OOS N > 0                       : {n_oos} {'✅' if oos_ok else '❌'}")

    e0_10_ok = _okuma_sayaci.get("E0 (10).csv", 0) == 0
    print(f"    E0 (10).csv READ = 0            : {'✅' if e0_10_ok else '❌'}")

    # KARAR
    if (feat_ok and feature_ok and numerik_ok and csv_ok and ham_ok and merge_ok
            and dup_ok and b365_ok and ftr_ok and top_ok and train_ok
            and val_ok and oos_ok and e0_10_ok):
        karar = "PASS → Explore"
        print(f"\n  ✅ KARAR: {karar}")
    else:
        karar = "FAIL → H9 KAPANIR"
        print(f"\n  ❌ KARAR: {karar}")
        print(f"     Protokol DEĞİŞMEZ.")

    return karar


# ==================================================================
# ANA
# ==================================================================
def main():
    print("=" * 80)
    print("H9 INVENTORY — Value Betting Veri Uygunluğu")
    print("2026/27 (E0 (10).csv) KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    # 1. h1_features_v2 kontrolü (KATI — RuntimeError riski)
    feat, feature_sutunlar, numerik_olmayan = kontrol_h1_features_v2()
    if feat is None:
        raise RuntimeError("h1_features_v2.csv yüklenemedi.")
    if feature_sutunlar is None:
        raise RuntimeError("Feature listesi oluşturulamadı.")

    # 2. Market verileri (KATI — her dosya 380)
    market = market_yukle()

    # 3. Birleştirme + filtre (KATI)
    df, ham_n, merge_n, dup, ah_dislanan = birlestir_ve_filtrele(feat, market)

    # 4. Değişken kontrolü
    feature_nan, gecersiz_ftr, nn_ftr = degisken_kontrol(df, feature_sutunlar)

    # 5. Bölme kontrolü
    n_train, n_val, n_oos, n_top = bolme_kontrol(df)

    # 6. Bütünlük
    veri_butunlugu(df)

    # 7. Karar
    karar = karar_matrisi(df, feature_sutunlar, numerik_olmayan, gecersiz_ftr,
                          nn_ftr, dup, ham_n, merge_n, n_train, n_val, n_oos, n_top)

    # -------- EXCEL --------
    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:

        # Sezon dağılımı
        sezon_rows = []
        for s in sorted(df["Season"].unique()):
            n = len(df[df["Season"] == s])
            rol = ("Train" if s in TRAIN_SEZONLAR
                   else "Validation" if s == VALIDATION_SEZON
                   else "OOS" if s == OOS_SEZON else "?")
            sezon_rows.append({"Sezon": s, "N": n, "Rol": rol})
        pd.DataFrame(sezon_rows).to_excel(w, sheet_name="Sezon_Dagilimi", index=False)

        # Bölme özeti
        pd.DataFrame([
            {"Bolme": "Train", "N_ham": BEKLENEN_HAM_TRAIN, "N_filtre_sonrasi": n_train},
            {"Bolme": "Validation", "N_ham": BEKLENEN_HAM_VAL, "N_filtre_sonrasi": n_val},
            {"Bolme": "OOS", "N_ham": BEKLENEN_HAM_OOS, "N_filtre_sonrasi": n_oos},
            {"Bolme": "Toplam", "N_ham": BEKLENEN_HAM_TOP, "N_filtre_sonrasi": n_top},
        ]).to_excel(w, sheet_name="Bolme_Ozet", index=False)

        # Feature listesi
        pd.DataFrame([{"Feature": c} for c in feature_sutunlar]).to_excel(
            w, sheet_name="Feature_Listesi", index=False)

        # Feature NaN
        if feature_nan:
            pd.DataFrame([{"Feature": c, "NaN_sayisi": v}
                          for c, v in feature_nan.items()]).to_excel(
                w, sheet_name="Feature_NaN", index=False)
        else:
            pd.DataFrame([{"Not": "Hiçbir feature'da NaN yok"}]).to_excel(
                w, sheet_name="Feature_NaN", index=False)

        # B365 kontrolü
        b365_rows = []
        for c in ["B365H", "B365D", "B365A"]:
            b365_rows.append({
                "Sutun": c,
                "N": int(df[c].notna().sum()),
                "NaN": int(df[c].isna().sum()),
                "Sifir": int((df[c] == 0).sum()),
                "Negatif": int((df[c] < 0).sum()),
            })
        pd.DataFrame(b365_rows).to_excel(w, sheet_name="B365_Kontrol", index=False)

        # FTR dağılımı
        ftr_rows = []
        for s in sorted(df["Season"].unique()):
            alt = df[df["Season"] == s]
            sayim = alt["FTR"].value_counts()
            ftr_rows.append({
                "Sezon": s,
                "H": int(sayim.get("H", 0)),
                "D": int(sayim.get("D", 0)),
                "A": int(sayim.get("A", 0)),
                "Toplam": len(alt),
            })
        pd.DataFrame(ftr_rows).to_excel(w, sheet_name="FTR_Dagilimi", index=False)

        # Notlar
        pd.DataFrame([{
            "Protokol": "H9 INVENTORY",
            "Tarih": "2026-10-08",
            "Amaç": "Veri uygunluğu kontrolü (KATI)",
            "Kapsam": "E0.csv – E0 (9).csv (10 sezon)",
            "E0_10": "KILITLI - okunmadi",
            "Model_fit": "YAPILMADI",
            "Value_hesaplandi": "HAYIR",
            "Threshold_uygulandi": "HAYIR",
            "KARAR": karar,
            "Toplam_satir_ham": ham_n,
            "Merge_sonrasi": merge_n,
            "Toplam_satir_filtre_sonrasi": n_top,
            "AH_dislanan": ah_dislanan,
            "Train_N": n_train,
            "Val_N": n_val,
            "OOS_N": n_oos,
            "Feature_sayisi": len(feature_sutunlar),
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} yazıldı.")
    print(f"\n{'='*80}")
    print(f"SON KARAR: {karar}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()