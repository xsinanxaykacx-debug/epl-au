
# -*- coding: utf-8 -*-
"""
EPL O/U 2.5 — veri bütünlüğü ve maç eşleşmesi denetimi.

Kapsam:
- 2015/16 - 2024/25: 10 tam sezon
- Her sezonda beklenen maç: 380
- E0 (10).csv: kısmi 2026/27 sezonu, kapsam dışı
- Tahmin modeli veya bahis performansı hesaplamaz.
- Yinelenen anahtarları ve skor uyuşmazlıklarını raporlar.
- Çıktı: ou25_match_audit.csv

Çalıştırma:
    python audit_ou25.py
"""

from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd


# ============================================================
# AYARLAR
# ============================================================

BASE = Path(__file__).resolve().parent
FEATURE_FILE = BASE / "h1_features_v2.csv"
OUTPUT_FILE = BASE / "ou25_match_audit.csv"
DUPLICATE_FILE = BASE / "ou25_duplicate_keys.csv"
UNMATCHED_FILE = BASE / "ou25_unmatched_matches.csv"

EXPECTED_MATCHES_PER_SEASON = 380

# Dosya adları ile sezonları açıkça eşleştir.
SEASON_FILES = [
    ("E0.csv", "2015/16"),
    ("E0 (1).csv", "2016/17"),
    ("E0 (2).csv", "2017/18"),
    ("E0 (3).csv", "2018/19"),
    ("E0 (4).csv", "2019/20"),
    ("E0 (5).csv", "2020/21"),
    ("E0 (6).csv", "2021/22"),
    ("E0 (7).csv", "2022/23"),
    ("E0 (8).csv", "2023/24"),
    ("E0 (9).csv", "2024/25"),
]

KEY_COLUMNS = ["DateKey", "HomeKey", "AwayKey"]

REQUIRED_FEATURE_COLUMNS = {
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
}

REQUIRED_ODDS_COLUMNS = {
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
}


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def read_csv_file(path):
    """CSV dosyasını yaygın kodlamaları deneyerek oku."""
    errors = []

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            df.columns = [
                str(column).strip().replace("\ufeff", "")
                for column in df.columns
            ]
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError(
        f"CSV okunamadı: {path}\n" + "\n".join(errors)
    )


def normalize_team(value):
    """Takım adını eşleştirme için standartlaştır."""
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_date(series):
    """
    Futbol veri setlerindeki gün/ay/yıl tarihlerini normalize et.
    Hatalı tarihler NaT olur ve ayrıca raporlanır.
    """
    return pd.to_datetime(
        series.astype("string").str.strip(),
        dayfirst=True,
        format="mixed",
        errors="coerce",
    ).dt.normalize()


def to_numeric(series):
    """Sayısal sütunları güvenli şekilde dönüştür."""
    return pd.to_numeric(series, errors="coerce")


def prepare_keys(df):
    """Standart maç anahtarlarını üret."""
    df["DateKey"] = normalize_date(df["Date"])
    df["HomeKey"] = df["HomeTeam"].map(normalize_team)
    df["AwayKey"] = df["AwayTeam"].map(normalize_team)

    invalid = (
        df["DateKey"].isna()
        | df["HomeKey"].eq("")
        | df["AwayKey"].eq("")
    )

    return invalid


def print_section(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def find_duplicates(df, key_columns):
    """Yinelenen anahtarlara ait bütün satırları döndür."""
    return df.loc[
        df.duplicated(key_columns, keep=False)
    ].copy()


def choose_odds_columns(df, filename):
    """
    Mevcut veri şemasına uygun ortalama oran sütunlarını seç.

    Öncelik:
    1. Avg>2.5 / Avg<2.5
    2. BbAv>2.5 / BbAv<2.5

    B365 veya Max sütunlarına sessizce geçiş yapılmaz.
    Böylece sezonlar arasında farklı oran kaynakları
    fark edilmeden karıştırılmaz.
    """
    candidates = [
        ("Avg>2.5", "Avg<2.5", "Avg"),
        ("BbAv>2.5", "BbAv<2.5", "BbAv"),
    ]

    for over_col, under_col, source in candidates:
        if {over_col, under_col}.issubset(df.columns):
            return over_col, under_col, source

    available = [
        column for column in df.columns
        if any(term in column.lower() for term in (
            "2.5", "avg", "bbav", "b365"
        ))
    ]

    raise ValueError(
        f"{filename}: uygun üst/alt 2.5 oran sütunları bulunamadı.\n"
        f"İlgili sütunlar: {available}"
    )


# ============================================================
# ANA DENETİM
# ============================================================

def main():
    if not BASE.exists():
        raise FileNotFoundError(
            f"Çalışma klasörü bulunamadı: {BASE}"
        )

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Özellik dosyası bulunamadı: {FEATURE_FILE}"
        )

    # --------------------------------------------------------
    # 1. ÖZELLİK DOSYASI
    # --------------------------------------------------------

    features = read_csv_file(FEATURE_FILE)

    missing = REQUIRED_FEATURE_COLUMNS - set(features.columns)
    if missing:
        raise ValueError(
            f"Özellik dosyasında eksik sütunlar: {sorted(missing)}"
        )

    features = features.copy()
    features["FeatureRow"] = range(1, len(features) + 1)

    feature_invalid = prepare_keys(features)

    features["FTHG"] = to_numeric(features["FTHG"])
    features["FTAG"] = to_numeric(features["FTAG"])

    features["Season"] = features["Season"].astype("string").str.strip()

    print_section("EPL O/U 2.5 — VERİ DENETİMİ")
    print(f"Çalışma klasörü : {BASE}")
    print(f"Özellik dosyası : {FEATURE_FILE.name}")
    print(f"Özellik satırı  : {len(features)}")
    print(f"Geçersiz maç anahtarı: {int(feature_invalid.sum())}")

    # --------------------------------------------------------
    # 2. SEZON DOSYALARINI OKU
    # --------------------------------------------------------

    odds_frames = []
    season_file_errors = []

    print_section("SEZON VE ORAN KONTROLÜ")

    for filename, expected_season in SEASON_FILES:
        path = BASE / filename

        if not path.exists():
            season_file_errors.append(
                f"Eksik dosya: {filename} ({expected_season})"
            )
            print(
                f"{filename:14} | DOSYA EKSİK "
                f"| sezon={expected_season}"
            )
            continue

        raw = read_csv_file(path)

        missing = REQUIRED_ODDS_COLUMNS - set(raw.columns)
        if missing:
            season_file_errors.append(
                f"{filename}: eksik sütunlar {sorted(missing)}"
            )
            print(
                f"{filename:14} | SÜTUN EKSİK: "
                f"{sorted(missing)}"
            )
            continue

        over_col, under_col, odds_source = choose_odds_columns(
            raw, filename
        )

        raw = raw.copy()
        raw["SeasonFile"] = expected_season
        raw["SourceFile"] = filename
        raw["OddsSource"] = odds_source
        raw["OddsOver"] = to_numeric(raw[over_col])
        raw["OddsUnder"] = to_numeric(raw[under_col])
        raw["FTHG_Odds"] = to_numeric(raw["FTHG"])
        raw["FTAG_Odds"] = to_numeric(raw["FTAG"])

        invalid_keys = prepare_keys(raw)

        valid_odds = (
            raw["OddsOver"].gt(1.0)
            & raw["OddsUnder"].gt(1.0)
            & raw["OddsOver"].notna()
            & raw["OddsUnder"].notna()
        )

        valid_rows = (
            ~invalid_keys
            & valid_odds
        )

        expected_rows_ok = len(raw) == EXPECTED_MATCHES_PER_SEASON

        print(
            f"{filename:14} | sezon={expected_season} "
            f"| maç={len(raw):3} "
            f"| geçerli oran={int(valid_odds.sum()):3} "
            f"| hatalı anahtar={int(invalid_keys.sum()):2} "
            f"| kaynak={odds_source}"
        )

        if not expected_rows_ok:
            season_file_errors.append(
                f"{filename}: {len(raw)} satır var; "
                f"{EXPECTED_MATCHES_PER_SEASON} bekleniyordu."
            )

        raw["OddsKeyInvalid"] = invalid_keys
        raw["OddsPairValid"] = valid_odds

        odds_frames.append(
            raw[[
                "DateKey",
                "HomeKey",
                "AwayKey",
                "SeasonFile",
                "SourceFile",
                "OddsSource",
                "OddsOver",
                "OddsUnder",
                "FTHG_Odds",
                "FTAG_Odds",
                "OddsKeyInvalid",
                "OddsPairValid",
            ]]
        )

    if season_file_errors:
        print_section("DOSYA / ŞEMA UYARILARI")
        for error in season_file_errors:
            print(f"- {error}")

    if not odds_frames:
        raise RuntimeError(
            "Hiçbir sezon dosyası başarıyla okunamadı."
        )

    odds = pd.concat(odds_frames, ignore_index=True)

    # --------------------------------------------------------
    # 3. ANAHTAR VE YİNELENEN KAYIT KONTROLÜ
    # --------------------------------------------------------

    feature_duplicates = find_duplicates(features, KEY_COLUMNS)
    odds_duplicates = find_duplicates(odds, KEY_COLUMNS)

    print_section("ANAHTAR KONTROLÜ")
    print(f"Özellik satırı: {len(features)}")
    print(f"Oran satırı:    {len(odds)}")
    print(
        "Özellik dosyasında yinelenen anahtara ait satır: "
        f"{len(feature_duplicates)}"
    )
    print(
        "Oran dosyalarında yinelenen anahtara ait satır:    "
        f"{len(odds_duplicates)}"
    )

    if feature_invalid.any():
        print(
            "UYARI: Özellik dosyasında tarih/takım anahtarı "
            "oluşturulamayan satırlar var."
        )

    if odds["OddsKeyInvalid"].any():
        print(
            "UYARI: Oran dosyalarında tarih/takım anahtarı "
            "oluşturulamayan satırlar var."
        )

    # Yinelenen kayıtları tek dosyada sakla.
    duplicate_parts = []

    if not feature_duplicates.empty:
        temp = feature_duplicates.copy()
        temp["DuplicateSource"] = "FEATURES"
        duplicate_parts.append(temp)

    if not odds_duplicates.empty:
        temp = odds_duplicates.copy()
        temp["DuplicateSource"] = "ODDS"
        duplicate_parts.append(temp)

    if duplicate_parts:
        pd.concat(
            duplicate_parts,
            ignore_index=True,
            sort=False,
        ).to_csv(
            DUPLICATE_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Yinelenen kayıt raporu: {DUPLICATE_FILE}")

    # Anahtarlar benzersiz değilse many-to-many birleştirme yapma.
    if (
        not feature_duplicates.empty
        or not odds_duplicates.empty
        or feature_invalid.any()
        or odds["OddsKeyInvalid"].any()
    ):
        print()
        print("DURDURULDU: Anahtar bütünlüğü doğrulanamadı.")
        print("Yinelenen veya geçersiz anahtarlar incelenmeden")
        print("eşleşme ve bahis analizi güvenilir kabul edilmemeli.")
        return 2

    # --------------------------------------------------------
    # 4. SEZON ETİKETLERİNİ DOĞRULA
    # --------------------------------------------------------

    print_section("SEZON ETİKETİ KONTROLÜ")

    feature_seasons = set(features["Season"].dropna().astype(str))
    expected_seasons = {season for _, season in SEASON_FILES}

    unexpected_seasons = feature_seasons - expected_seasons
    missing_seasons = expected_seasons - feature_seasons

    print(f"Özellik dosyasındaki sezon sayısı: {len(feature_seasons)}")
    print(f"Beklenen sezon sayısı:             {len(expected_seasons)}")
    print(f"Beklenmeyen sezon etiketi:         {sorted(unexpected_seasons)}")
    print(f"Eksik sezon etiketi:               {sorted(missing_seasons)}")

    if unexpected_seasons or missing_seasons:
        print("UYARI: Sezon etiketleri beklenen kapsamla tam uyuşmuyor.")

    # --------------------------------------------------------
    # 5. BİRLEŞTİRME
    # --------------------------------------------------------

    # Sezon bilgisini oran dosyasından ekleyip yanlış sezonla
    # eşleşen kayıtları daha sonra ayrıca kontrol edeceğiz.
    merged = features.merge(
        odds,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched_mask = merged["_merge"].eq("both")
    unmatched_mask = merged["_merge"].eq("left_only")

    odds_valid_mask = (
        merged["OddsOver"].gt(1.0)
        & merged["OddsUnder"].gt(1.0)
        & merged["OddsOver"].notna()
        & merged["OddsUnder"].notna()
    )

    score_valid = (
        merged["FTHG"].notna()
        & merged["FTAG"].notna()
        & merged["FTHG_Odds"].notna()
        & merged["FTAG_Odds"].notna()
    )

    score_mismatch = (
        score_valid
        & (
            merged["FTHG"].ne(merged["FTHG_Odds"])
            | merged["FTAG"].ne(merged["FTAG_Odds"])
        )
    )

    season_mismatch = (
        matched_mask
        & merged["Season"].astype("string").ne(
            merged["SeasonFile"].astype("string")
        )
    )

    print_section("EŞLEŞME SONUCU")
    print(f"Özellik maçı:                    {len(merged)}")
    print(f"Eşleşen maç:                     {int(matched_mask.sum())}")
    print(f"Eşleşmeyen maç:                  {int(unmatched_mask.sum())}")
    print(f"Geçerli üst/alt oran çifti:      {int(odds_valid_mask.sum())}")
    print(f"Geçersiz/eksik oran çifti:       {int((~odds_valid_mask).sum())}")
    print(f"Skor uyuşmazlığı:                {int(score_mismatch.sum())}")
    print(f"Sezon etiketi uyuşmazlığı:       {int(season_mismatch.sum())}")

    # --------------------------------------------------------
    # 6. SEZON BAZINDA SONUÇLAR
    # --------------------------------------------------------

    print_section("SEZON BAZINDA")

    season_report = []

    for season, group in merged.groupby("Season", sort=True, dropna=False):
        count = len(group)
        matched = int(group["_merge"].eq("both").sum())

        valid = (
            group["OddsOver"].gt(1.0)
            & group["OddsUnder"].gt(1.0)
            & group["OddsOver"].notna()
            & group["OddsUnder"].notna()
        )
        valid_count = int(valid.sum())

        score_ok = (
            group["FTHG"].notna()
            & group["FTAG"].notna()
            & group["FTHG_Odds"].notna()
            & group["FTAG_Odds"].notna()
        )
        score_bad = (
            score_ok
            & (
                group["FTHG"].ne(group["FTHG_Odds"])
                | group["FTAG"].ne(group["FTAG_Odds"])
            )
        )

        source_counts = (
            group.loc[group["_merge"].eq("both"), "OddsSource"]
            .value_counts()
            .to_dict()
        )

        print(
            f"{season}: özellik={count}, "
            f"eşleşme={matched}, "
            f"geçerli oran={valid_count}, "
            f"skor uyuşmazlığı={int(score_bad.sum())}, "
            f"oran kaynakları={source_counts}"
        )

        season_report.append({
            "Season": season,
            "FeatureMatches": count,
            "MatchedMatches": matched,
            "ValidOddsPairs": valid_count,
            "ScoreMismatches": int(score_bad.sum()),
            "UnmatchedMatches": count - matched,
        })

    # --------------------------------------------------------
    # 7. EŞLEŞMEYEN MAÇLARI KAYDET
    # --------------------------------------------------------

    unmatched = merged.loc[
        unmatched_mask,
        ["Season", "Date", "HomeTeam", "AwayTeam"],
    ].copy()

    unmatched.to_csv(
        UNMATCHED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    if not unmatched.empty:
        print_section("İLK 20 EŞLEŞMEYEN MAÇ")
        print(unmatched.head(20).to_string(index=False))
        print(f"Tam liste: {UNMATCHED_FILE}")

    # --------------------------------------------------------
    # 8. SKOR VE SEZON HATALARINI RAPORLA
    # --------------------------------------------------------

    if score_mismatch.any():
        print_section("SKOR UYUŞMAZLIKLARI — İLK 20")
        cols = [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTHG_Odds",
            "FTAG_Odds",
        ]
        print(
            merged.loc[score_mismatch, cols]
            .head(20)
            .to_string(index=False)
        )

    if season_mismatch.any():
        print_section("SEZON UYUŞMAZLIKLARI — İLK 20")
        cols = [
            "Season",
            "SeasonFile",
            "Date",
            "HomeTeam",
            "AwayTeam",
        ]
        print(
            merged.loc[season_mismatch, cols]
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # 9. SONUÇ DOSYASI
    # --------------------------------------------------------

    merged.drop(columns=["_merge"]).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_section("DENETİM DOSYALARI")
    print(f"Ana çıktı:             {OUTPUT_FILE}")
    print(f"Eşleşmeyen maçlar:     {UNMATCHED_FILE}")
    if not feature_duplicates.empty or not odds_duplicates.empty:
        print(f"Yinelenen anahtarlar:  {DUPLICATE_FILE}")

    # --------------------------------------------------------
    # 10. OTOMATİK ÖN DEĞERLENDİRME
    # --------------------------------------------------------

    expected_total = len(SEASON_FILES) * EXPECTED_MATCHES_PER_SEASON
    full_seasons_ok = (
        len(features) == expected_total
        and len(odds) == expected_total
        and not season_file_errors
        and not unexpected_seasons
        and not missing_seasons
    )

    audit_ok = (
        full_seasons_ok
        and int(matched_mask.sum()) == len(features)
        and int(odds_valid_mask.sum()) == len(features)
        and int(score_mismatch.sum()) == 0
        and int(season_mismatch.sum()) == 0
    )

    print_section("ÖN DENETİM KARARI")

    if audit_ok:
        print("VERİ EŞLEŞMESİ: TEMİZ")
        print("10 tam sezonun maçları, skorları ve oran çiftleri eşleşti.")
        print("Not: Bu sonuç bahis avantajı veya kârlılık kanıtı değildir.")
        return 0

    print("VERİ EŞLEŞMESİ: İNCELEME GEREKİYOR")
    print("Bahis modeline geçmeden önce yukarıdaki farkları incele.")
    print("Bu betik model başarısı veya kârlılık hakkında karar vermez.")
    return 1


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print()
        print(f"HATA: {exc}")
        exit_code = 1

    sys.exit(exit_code)