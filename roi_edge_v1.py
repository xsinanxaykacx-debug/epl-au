# -*- coding: utf-8 -*-
"""
roi_edge_v1.py
ROI/Edge Araştırması v1.1 (FROZEN) — birebir implementasyon
Tarih: 2026-10-08

Kapsam:
- Train:      2019/20 + 2020/21    → E0 (4).csv + E0 (5).csv
- Validation: 2021/22 + 2022/23 + 2023/24 → E0 (6).csv + E0 (7).csv + E0 (8).csv
- Arşiv:      2024/25               → E0 (9).csv (yalnızca raporlama)
- 2026/27:    ASLA OKUNMAYACAK (E0 (10).csv)

Düzeltmeler (v2):
1. Eksik AH/OU maçları family-based filtrelenir (df_ah, df_ou).
2. AH-E2 nötr bölge |x| ≤ 0.10 → bahis yok.
3. OU-E3 nötr bölge |x| ≤ 0.15 → bahis yok.
4. Permutation: yalnızca sonuç etiketleri (gol_fark / toplam_gol) permüte edilir;
   oran, yön, handikap sabit.
5. Karar_Ozeti her durumda oluşturulur (boş olsa bile).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# ==================================================================
# 0. SABİTLER (KİLİTLİ)
# ==================================================================
CIKTI = "roi_edge_v1_results.xlsx"
ENVANTER = "envanter_v2.xlsx"
ENVANTER_SHEET = "Eksik_Maclar"

DOSYALAR = {
    "Train_2019_20": ("E0 (4).csv", "2019/20", "Train"),
    "Train_2020_21": ("E0 (5).csv", "2020/21", "Train"),
    "Val_2021_22":   ("E0 (6).csv", "2021/22", "Validation"),
    "Val_2022_23":   ("E0 (7).csv", "2022/23", "Validation"),
    "Val_2023_24":   ("E0 (8).csv", "2023/24", "Validation"),
    "Arsiv_2024_25": ("E0 (9).csv", "2024/25", "Arsiv"),
}

YASAKLI_DOSYALAR = {"E0 (10).csv"}

N_BOOTSTRAP = 10_000
N_PERMUTATION = 10_000
RANDOM_SEED = 20261008
BANKROLL = 100.0
MDD_ESIK = 0.20
ALPHA = 0.05
N_MIN = 100

# Kova sınırları (KİLİTLİ)
AH_HANDICAP_KOVALARI = [
    ("AHh ≤ -1.5",        lambda x: x <= -1.5),
    ("-1.5 < AHh ≤ -0.5", lambda x: -1.5 < x <= -0.5),
    ("-0.5 < AHh < 0.5",  lambda x: -0.5 < x < 0.5),
    ("0.5 ≤ AHh < 1.5",   lambda x: 0.5 <= x < 1.5),
    ("AHh ≥ 1.5",         lambda x: x >= 1.5),
]

AH_ASIMETRI_KOVALARI = [
    ("x ≤ -0.30",         lambda x: x <= -0.30),
    ("-0.30 < x < -0.10", lambda x: -0.30 < x < -0.10),
    ("-0.10 ≤ x ≤ 0.10",  lambda x: -0.10 <= x <= 0.10),
    ("0.10 < x < 0.30",   lambda x: 0.10 < x < 0.30),
    ("x ≥ 0.30",          lambda x: x >= 0.30),
]

B365_AH_FIYAT_KOVALARI = [
    ("<1.50",     lambda x: x < 1.50),
    ("1.50–1.79", lambda x: 1.50 <= x < 1.80),
    ("1.80–2.09", lambda x: 1.80 <= x < 2.10),
    ("2.10–2.49", lambda x: 2.10 <= x < 2.50),
    ("≥2.50",     lambda x: x >= 2.50),
]

OU_FIYAT_KOVALARI = [
    ("<1.80",     lambda x: x < 1.80),
    ("1.80–1.99", lambda x: 1.80 <= x < 2.00),
    ("2.00–2.19", lambda x: 2.00 <= x < 2.20),
    ("2.20–2.49", lambda x: 2.20 <= x < 2.50),
    ("≥2.50",     lambda x: x >= 2.50),
]

OU_ASIMETRI_KOVALARI = [
    ("x ≤ -0.40",         lambda x: x <= -0.40),
    ("-0.40 < x < -0.15", lambda x: -0.40 < x < -0.15),
    ("-0.15 ≤ x ≤ 0.15",  lambda x: -0.15 <= x <= 0.15),
    ("0.15 < x < 0.40",   lambda x: 0.15 < x < 0.40),
    ("x ≥ 0.40",          lambda x: x >= 0.40),
]

HAREKET_KOVALARI = [
    ("küçük", lambda x: abs(x) < 0.02),
    ("orta",  lambda x: 0.02 <= abs(x) < 0.05),
    ("büyük", lambda x: abs(x) >= 0.05),
]

SINYAL_META = {
    "S-AH-E1": {"tip": "Executable", "market": "AH", "tanim": "Açılış handikap"},
    "S-AH-E2": {"tip": "Executable", "market": "AH", "tanim": "Açılış asimetri"},
    "S-AH-E3": {"tip": "Executable", "market": "AH", "tanim": "Ev fiyat seviyesi→Away"},
    "S-AH-E4": {"tip": "Executable", "market": "AH", "tanim": "Dep fiyat seviyesi→Home"},
    "S-OU-E1": {"tip": "Executable", "market": "OU", "tanim": "Over fiyat seviyesi→Under"},
    "S-OU-E2": {"tip": "Executable", "market": "OU", "tanim": "Under fiyat seviyesi→Over"},
    "S-OU-E3": {"tip": "Executable", "market": "OU", "tanim": "Açılış asimetri"},
    "S-AH-B1": {"tip": "Benchmark", "market": "AH", "tanim": "Handikap hareketi"},
    "S-AH-B2": {"tip": "Benchmark", "market": "AH", "tanim": "B365 ev fiyat hareketi"},
    "S-AH-B3": {"tip": "Benchmark", "market": "AH", "tanim": "B365 dep fiyat hareketi"},
    "S-AH-B4": {"tip": "Benchmark", "market": "AH", "tanim": "Ev hareket büyüklüğü"},
    "S-AH-B5": {"tip": "Benchmark", "market": "AH", "tanim": "Dep hareket büyüklüğü"},
    "S-AH-B6": {"tip": "Benchmark", "market": "AH", "tanim": "B365-Pinnacle ev farkı"},
    "S-AH-B7": {"tip": "Benchmark", "market": "AH", "tanim": "B365-Pinnacle dep farkı"},
    "S-OU-B1": {"tip": "Benchmark", "market": "OU", "tanim": "Over fiyat hareketi"},
    "S-OU-B2": {"tip": "Benchmark", "market": "OU", "tanim": "Under fiyat hareketi"},
    "S-OU-B3": {"tip": "Benchmark", "market": "OU", "tanim": "Ortak hareket — TEST EDİLMEZ"},
    "S-OU-B4": {"tip": "Benchmark", "market": "OU", "tanim": "Over hareket büyüklüğü"},
    "S-OU-B5": {"tip": "Benchmark", "market": "OU", "tanim": "Under hareket büyüklüğü"},
    "S-OU-B6": {"tip": "Benchmark", "market": "OU", "tanim": "B365-Pinnacle Over farkı"},
    "S-OU-B7": {"tip": "Benchmark", "market": "OU", "tanim": "B365-Pinnacle Under farkı"},
}

TEST_EDILMEYEN = {"S-OU-B3"}


# ==================================================================
# 1. VERİ YÜKLEME
# ==================================================================
def _guvenlik_kontrolu():
    for f in YASAKLI_DOSYALAR:
        pass  # var olsa bile açmayacağız


def _turetilmis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AH_line_fark"]   = df["AHCh"] - df["AHh"]
    df["B365_Home_fark"] = df["B365CAHH"] - df["B365AHH"]
    df["B365_Away_fark"] = df["B365CAHA"] - df["B365AHA"]
    df["B365_Home_abs"]  = df["B365_Home_fark"].abs()
    df["B365_Away_abs"]  = df["B365_Away_fark"].abs()
    df["B365AHH_seviye"] = df["B365AHH"]
    df["B365AHA_seviye"] = df["B365AHA"]
    df["AH_asimetri"]    = df["B365AHH"] - df["B365AHA"]
    df["Over_fark"]      = df["B365C>2.5"] - df["B365>2.5"]
    df["Under_fark"]     = df["B365C<2.5"] - df["B365<2.5"]
    df["Over_abs"]       = df["Over_fark"].abs()
    df["Under_abs"]      = df["Under_fark"].abs()
    df["OU_asimetri"]    = df["B365>2.5"] - df["B365<2.5"]
    df["B365P_AH_Home_Kp"]  = df["B365CAHH"] - df["PCAHH"]
    df["B365P_AH_Away_Kp"]  = df["B365CAHA"] - df["PCAHA"]
    df["B365P_OU_Over_Kp"]  = df["B365C>2.5"] - df["PC>2.5"]
    df["B365P_OU_Under_Kp"] = df["B365C<2.5"] - df["PC<2.5"]
    df["ToplamGol"] = df["FTHG"] + df["FTAG"]
    df["GolFark"]   = df["FTHG"] - df["FTAG"]
    return df


def veri_yukle() -> dict:
    """Her bölme için (df_ah, df_ou) döner. Eksik maçlar family-based dışlanır."""
    _guvenlik_kontrolu()
    eksik = pd.read_excel(ENVANTER, sheet_name=ENVANTER_SHEET)

    def _key(df):
        tarih = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
        return (df["Season"].astype(str).str.strip()
                + "|" + tarih.fillna("NA")
                + "|" + df["HomeTeam"].astype(str).str.strip()
                + "|" + df["AwayTeam"].astype(str).str.strip())

    eksik["_key"] = _key(eksik)
    eksik_ah = set(eksik.loc[eksik["AH_Eksik"] == True, "_key"].dropna().astype(str))
    eksik_ou = set(eksik.loc[eksik["OU_Eksik"] == True, "_key"].dropna().astype(str))

    bolmeler = {"Train": [], "Validation": [], "Arsiv": []}
    for etiket, (dosya, sezon, bolme) in DOSYALAR.items():
        df = pd.read_csv(dosya, encoding="utf-8-sig")
        df["Season"] = sezon
        df["_tarih"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
        df = df.sort_values("_tarih").reset_index(drop=True)
        df["_key"] = _key(df)
        df["_eksik_ah"] = df["_key"].isin(eksik_ah)
        df["_eksik_ou"] = df["_key"].isin(eksik_ou)
        df = _turetilmis(df)
        bolmeler[bolme].append(df)

    out = {}
    for k, v in bolmeler.items():
        d = pd.concat(v, ignore_index=True)
        d = d.sort_values("_tarih").reset_index(drop=True)
        # FAMILY-BASED FİLTRELEME
        df_ah = d.loc[~d["_eksik_ah"]].copy().reset_index(drop=True)
        df_ou = d.loc[~d["_eksik_ou"]].copy().reset_index(drop=True)
        out[k] = {"ah": df_ah, "ou": df_ou}
    return out


# ==================================================================
# 2. ASIAN HANDICAP SETTLEMENT
# ==================================================================
def _ah_settle_yarim(handicap: float, gol_fark: int, taraf: str) -> float:
    if taraf == "Home":
        adjusted = gol_fark + handicap
    else:
        adjusted = -gol_fark - handicap
    if adjusted > 0: return 1.0
    elif adjusted == 0: return 0.0
    else: return -1.0


def ah_settlement(handicap: float, gol_fark: int, taraf: str, oran: float) -> float:
    if handicap is None or np.isnan(handicap):
        return 0.0
    is_quarter = (abs(handicap * 4) % 2 == 1)
    if is_quarter:
        h1 = handicap - 0.25 if handicap > 0 else handicap + 0.25
        h2 = handicap + 0.25 if handicap > 0 else handicap - 0.25
        r1 = _ah_settle_yarim(h1, gol_fark, taraf)
        r2 = _ah_settle_yarim(h2, gol_fark, taraf)
        pl = 0.0
        for r in (r1, r2):
            if r == 1.0: pl += 0.5 * (oran - 1.0)
            elif r == -1.0: pl += 0.5 * (-1.0)
        return pl
    else:
        r = _ah_settle_yarim(handicap, gol_fark, taraf)
        if r == 1.0: return oran - 1.0
        elif r == -1.0: return -1.0
        else: return 0.0


def ou_settlement(toplam_gol: int, taraf: str, oran: float) -> float:
    if taraf == "Over":
        if toplam_gol > 2.5: return oran - 1.0
        elif toplam_gol < 2.5: return -1.0
        else: return 0.0
    else:
        if toplam_gol < 2.5: return oran - 1.0
        elif toplam_gol > 2.5: return -1.0
        else: return 0.0


# ==================================================================
# 3. SİNYAL TANIMLARI
# ==================================================================
def _kova_uygula(deger, kovalar):
    for etiket, fn in kovalar:
        if pd.notna(deger) and fn(deger):
            return etiket
    return None


def _bahis_ekle(rows, key, tarih, yon, oran, market, **extra):
    if pd.isna(oran):
        return
    b = {"key": key, "tarih": tarih, "yon": yon, "oran": float(oran), "market": market}
    b.update(extra)
    rows.append(b)


def sinyal_executable_ah(df):
    """df = AH filtrelenmiş DataFrame."""
    out = {k: [] for k in ["S-AH-E1", "S-AH-E2", "S-AH-E3", "S-AH-E4"]}
    for _, r in df.iterrows():
        # S-AH-E1: Açılış handikap
        kova = _kova_uygula(r["AHh"], AH_HANDICAP_KOVALARI)
        if kova and r["AHh"] != 0:
            yon = "Home" if r["AHh"] < 0 else "Away"
            oran = r["B365AHH"] if yon == "Home" else r["B365AHA"]
            _bahis_ekle(out["S-AH-E1"], r["_key"], r["_tarih"], yon, oran, "AH",
                        handikap=r["AHh"], gol_fark=r["GolFark"])
        # S-AH-E2: Açılış asimetri — nötr bölge |x| ≤ 0.10 → bahis yok
        x = r["AH_asimetri"]
        if pd.notna(x) and abs(x) > 0.10:
            kova = _kova_uygula(x, AH_ASIMETRI_KOVALARI)
            if kova:
                yon = "Home" if x < 0 else "Away"
                oran = r["B365AHH"] if yon == "Home" else r["B365AHA"]
                _bahis_ekle(out["S-AH-E2"], r["_key"], r["_tarih"], yon, oran, "AH",
                            handikap=r["AHh"], gol_fark=r["GolFark"])
        # S-AH-E3
        kova = _kova_uygula(r["B365AHH"], B365_AH_FIYAT_KOVALARI)
        if kova:
            _bahis_ekle(out["S-AH-E3"], r["_key"], r["_tarih"], "Away", r["B365AHA"], "AH",
                        handikap=r["AHh"], gol_fark=r["GolFark"])
        # S-AH-E4
        kova = _kova_uygula(r["B365AHA"], B365_AH_FIYAT_KOVALARI)
        if kova:
            _bahis_ekle(out["S-AH-E4"], r["_key"], r["_tarih"], "Home", r["B365AHH"], "AH",
                        handikap=r["AHh"], gol_fark=r["GolFark"])
    return out


def sinyal_executable_ou(df):
    """df = OU filtrelenmiş DataFrame."""
    out = {k: [] for k in ["S-OU-E1", "S-OU-E2", "S-OU-E3"]}
    for _, r in df.iterrows():
        # S-OU-E1
        kova = _kova_uygula(r["B365>2.5"], OU_FIYAT_KOVALARI)
        if kova:
            _bahis_ekle(out["S-OU-E1"], r["_key"], r["_tarih"], "Under", r["B365<2.5"], "OU",
                        toplam_gol=r["ToplamGol"])
        # S-OU-E2
        kova = _kova_uygula(r["B365<2.5"], OU_FIYAT_KOVALARI)
        if kova:
            _bahis_ekle(out["S-OU-E2"], r["_key"], r["_tarih"], "Over", r["B365>2.5"], "OU",
                        toplam_gol=r["ToplamGol"])
        # S-OU-E3: Açılış asimetri — nötr bölge |x| ≤ 0.15 → bahis yok
        x = r["OU_asimetri"]
        if pd.notna(x) and abs(x) > 0.15:
            kova = _kova_uygula(x, OU_ASIMETRI_KOVALARI)
            if kova:
                yon = "Over" if x < 0 else "Under"
                oran = r["B365>2.5"] if yon == "Over" else r["B365<2.5"]
                _bahis_ekle(out["S-OU-E3"], r["_key"], r["_tarih"], yon, oran, "OU",
                            toplam_gol=r["ToplamGol"])
    return out


def sinyal_benchmark_ah(df):
    out = {k: [] for k in ["S-AH-B1","S-AH-B2","S-AH-B3","S-AH-B4","S-AH-B5","S-AH-B6","S-AH-B7"]}
    for _, r in df.iterrows():
        # B1
        x = r["AH_line_fark"]
        if pd.notna(x) and x != 0:
            yon = "Home" if x < 0 else "Away"
            oran = r["B365CAHH"] if yon == "Home" else r["B365CAHA"]
            _bahis_ekle(out["S-AH-B1"], r["_key"], r["_tarih"], yon, oran, "AH",
                        handikap=r["AHCh"], gol_fark=r["GolFark"])
        # B2
        x = r["B365_Home_fark"]
        if pd.notna(x) and x != 0:
            yon = "Home" if x < 0 else "Away"
            oran = r["B365CAHH"] if yon == "Home" else r["B365CAHA"]
            _bahis_ekle(out["S-AH-B2"], r["_key"], r["_tarih"], yon, oran, "AH",
                        handikap=r["AHCh"], gol_fark=r["GolFark"])
        # B3
        x = r["B365_Away_fark"]
        if pd.notna(x) and x != 0:
            yon = "Away" if x < 0 else "Home"
            oran = r["B365CAHH"] if yon == "Home" else r["B365CAHA"]
            _bahis_ekle(out["S-AH-B3"], r["_key"], r["_tarih"], yon, oran, "AH",
                        handikap=r["AHCh"], gol_fark=r["GolFark"])
        # B4
        x = r["B365_Home_fark"]
        if pd.notna(x) and x != 0:
            kova = _kova_uygula(x, HAREKET_KOVALARI)
            if kova:
                yon = "Home" if x < 0 else "Away"
                oran = r["B365CAHH"] if yon == "Home" else r["B365CAHA"]
                _bahis_ekle(out["S-AH-B4"], r["_key"], r["_tarih"], yon, oran, "AH",
                            handikap=r["AHCh"], gol_fark=r["GolFark"], kova=kova)
        # B5
        x = r["B365_Away_fark"]
        if pd.notna(x) and x != 0:
            kova = _kova_uygula(x, HAREKET_KOVALARI)
            if kova:
                yon = "Away" if x < 0 else "Home"
                oran = r["B365CAHH"] if yon == "Home" else r["B365CAHA"]
                _bahis_ekle(out["S-AH-B5"], r["_key"], r["_tarih"], yon, oran, "AH",
                            handikap=r["AHCh"], gol_fark=r["GolFark"], kova=kova)
        # B6
        x = r["B365P_AH_Home_Kp"]
        if pd.notna(x) and x != 0:
            yon = "Home" if x < 0 else "Away"
            oran = r["B365CAHH"] if yon == "Home" else r["B365CAHA"]
            _bahis_ekle(out["S-AH-B6"], r["_key"], r["_tarih"], yon, oran, "AH",
                        handikap=r["AHCh"], gol_fark=r["GolFark"])
        # B7
        x = r["B365P_AH_Away_Kp"]
        if pd.notna(x) and x != 0:
            yon = "Away" if x < 0 else "Home"
            oran = r["B365CAHH"] if yon == "Home" else r["B365CAHA"]
            _bahis_ekle(out["S-AH-B7"], r["_key"], r["_tarih"], yon, oran, "AH",
                        handikap=r["AHCh"], gol_fark=r["GolFark"])
    return out


def sinyal_benchmark_ou(df):
    out = {k: [] for k in ["S-OU-B1","S-OU-B2","S-OU-B4","S-OU-B5","S-OU-B6","S-OU-B7"]}
    for _, r in df.iterrows():
        # B1
        x = r["Over_fark"]
        if pd.notna(x) and x != 0:
            yon = "Over" if x < 0 else "Under"
            oran = r["B365C>2.5"] if yon == "Over" else r["B365C<2.5"]
            _bahis_ekle(out["S-OU-B1"], r["_key"], r["_tarih"], yon, oran, "OU",
                        toplam_gol=r["ToplamGol"])
        # B2
        x = r["Under_fark"]
        if pd.notna(x) and x != 0:
            yon = "Under" if x < 0 else "Over"
            oran = r["B365C>2.5"] if yon == "Over" else r["B365C<2.5"]
            _bahis_ekle(out["S-OU-B2"], r["_key"], r["_tarih"], yon, oran, "OU",
                        toplam_gol=r["ToplamGol"])
        # B4
        x = r["Over_fark"]
        if pd.notna(x) and x != 0:
            kova = _kova_uygula(x, HAREKET_KOVALARI)
            if kova:
                yon = "Over" if x < 0 else "Under"
                oran = r["B365C>2.5"] if yon == "Over" else r["B365C<2.5"]
                _bahis_ekle(out["S-OU-B4"], r["_key"], r["_tarih"], yon, oran, "OU",
                            toplam_gol=r["ToplamGol"], kova=kova)
        # B5
        x = r["Under_fark"]
        if pd.notna(x) and x != 0:
            kova = _kova_uygula(x, HAREKET_KOVALARI)
            if kova:
                yon = "Under" if x < 0 else "Over"
                oran = r["B365C>2.5"] if yon == "Over" else r["B365C<2.5"]
                _bahis_ekle(out["S-OU-B5"], r["_key"], r["_tarih"], yon, oran, "OU",
                            toplam_gol=r["ToplamGol"], kova=kova)
        # B6
        x = r["B365P_OU_Over_Kp"]
        if pd.notna(x) and x != 0:
            yon = "Over" if x < 0 else "Under"
            oran = r["B365C>2.5"] if yon == "Over" else r["B365C<2.5"]
            _bahis_ekle(out["S-OU-B6"], r["_key"], r["_tarih"], yon, oran, "OU",
                        toplam_gol=r["ToplamGol"])
        # B7
        x = r["B365P_OU_Under_Kp"]
        if pd.notna(x) and x != 0:
            yon = "Under" if x < 0 else "Over"
            oran = r["B365C>2.5"] if yon == "Over" else r["B365C<2.5"]
            _bahis_ekle(out["S-OU-B7"], r["_key"], r["_tarih"], yon, oran, "OU",
                        toplam_gol=r["ToplamGol"])
    return out


# ==================================================================
# 4. P/L HESABI
# ==================================================================
def hesapla_pl(bets: list) -> np.ndarray:
    pl = []
    for b in bets:
        if b["market"] == "AH":
            pl.append(ah_settlement(b["handikap"], b["gol_fark"], b["yon"], b["oran"]))
        else:
            pl.append(ou_settlement(b["toplam_gol"], b["yon"], b["oran"]))
    return np.array(pl, dtype=float)


# ==================================================================
# 5. İSTATİSTİK
# ==================================================================
def bca_bootstrap_roi(pl: np.ndarray, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    if len(pl) < 5: return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(pl)
    theta_hat = pl.mean()
    theta_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        theta_boot[b] = pl[idx].mean()
    prop_less = np.mean(theta_boot < theta_hat)
    prop_less = min(max(prop_less, 1.0/(n_boot+1)), n_boot/(n_boot+1))
    z0 = norm.ppf(prop_less)
    theta_jack = np.empty(n)
    for i in range(n):
        theta_jack[i] = np.delete(pl, i).mean()
    tj_mean = theta_jack.mean()
    num = ((tj_mean - theta_jack) ** 3).sum()
    den = 6.0 * (((tj_mean - theta_jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0
    z_lo = norm.ppf(0.025); z_hi = norm.ppf(0.975)
    def _p(z):
        nn = z0 + z; dd = 1 - a * nn
        return norm.cdf(z0 + nn / dd) if dd != 0 else np.nan
    p_lo = min(max(_p(z_lo), 0.0), 1.0)
    p_hi = min(max(_p(z_hi), 0.0), 1.0)
    return float(np.quantile(theta_boot, p_lo)), float(np.quantile(theta_boot, p_hi))


def permutation_test(bets: list, n_perm=N_PERMUTATION, seed=RANDOM_SEED):
    """
    DOĞRU PERMUTASYON: yalnızca sonuç etiketleri permüte edilir.
    Oran, yön, handikap SABİT kalır.
    İki taraflı test.
    """
    if len(bets) < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    n = len(bets)

    # Bahisleri ayrıştır: her bahis için sabit alanlar ayrı, sonuç ayrı
    yonler  = [b["yon"] for b in bets]
    oranlar = [b["oran"] for b in bets]
    marketler = [b["market"] for b in bets]

    # Sonuç etiketleri
    gol_farklari = np.array([b["gol_fark"] if b["market"] == "AH" else None for b in bets], dtype=object)
    toplam_goller = np.array([b["toplam_gol"] if b["market"] == "OU" else None for b in bets], dtype=object)
    handikaplar = np.array([b["handikap"] if b["market"] == "AH" else None for b in bets], dtype=object)

    # Gözlenen ROI
    def _hesapla(perm_idx):
        pl = []
        for i in range(n):
            if marketler[i] == "AH":
                gf = gol_farklari[perm_idx[i]]
                if gf is None:
                    pl.append(0.0); continue
                pl.append(ah_settlement(handikaplar[i], gf, yonler[i], oranlar[i]))
            else:
                tg = toplam_goller[perm_idx[i]]
                if tg is None:
                    pl.append(0.0); continue
                pl.append(ou_settlement(tg, yonler[i], oranlar[i]))
        return np.mean(pl)

    # Gözlenen (perm_idx = identity)
    identity = np.arange(n)
    obs = _hesapla(identity)

    ekstrem = 0
    for _ in range(n_perm):
        perm_idx = rng.permutation(n)
        r = _hesapla(perm_idx)
        if abs(r) >= abs(obs):
            ekstrem += 1
    return (1 + ekstrem) / (n_perm + 1)


def max_drawdown(pl: np.ndarray) -> float:
    if len(pl) == 0: return 0.0
    cum = np.cumsum(pl)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max() / BANKROLL)


# ==================================================================
# 6. SİNYAL ANALİZİ
# ==================================================================
def analiz_sinyal(bets: list, kod: str) -> dict:
    if len(bets) == 0:
        return {"kod": kod, "n": 0, "not": "bahis yok"}
    bets_sorted = sorted(bets, key=lambda x: x["tarih"])
    pl = hesapla_pl(bets_sorted)
    n = len(pl)
    win = (pl > 0).sum()
    push = (pl == 0).sum()
    oranlar = [b["oran"] for b in bets_sorted]
    ort_oran = float(np.mean(oranlar))
    net = float(pl.sum())
    roi = net / n
    ci_lo, ci_hi = bca_bootstrap_roi(pl)
    p_perm = permutation_test(bets_sorted)
    mdd = max_drawdown(pl)
    return {
        "kod": kod, "n": n, "win": int(win), "push": int(push),
        "win_pct": round(win/n*100, 2),
        "ort_oran": round(ort_oran, 4),
        "stake": n, "net_pl": round(net, 4), "roi": round(roi, 4),
        "roi_ci_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else None,
        "roi_ci_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else None,
        "perm_p": round(p_perm, 4) if not np.isnan(p_perm) else None,
        "mdd": round(mdd, 4),
    }


def karar_uygula(metrik: dict) -> str:
    if metrik["n"] < N_MIN: return "Kanıt yetersiz (N<100)"
    if metrik["roi"] <= 0: return "Reddedildi (ROI≤0)"
    if metrik["roi_ci_lo"] is None or metrik["roi_ci_lo"] <= 0: return "Kanıt yetersiz (GA≤0)"
    if metrik["perm_p"] is None or metrik["perm_p"] >= ALPHA: return "Kanıt yetersiz (p≥0.05)"
    if metrik["mdd"] > MDD_ESIK: return "Kanıt yetersiz (MDD>%20)"
    return "GÜÇLÜ KANIT"


# ==================================================================
# 7. ANA AKIŞ
# ==================================================================
def main():
    print("=" * 70)
    print("ROI/EDGE v1.1 — FROZEN (impl v2)")
    print("E0 (10).csv KESİNLİKLE OKUNMAYACAK.")
    print("=" * 70)

    bolmeler = veri_yukle()
    for k, v in bolmeler.items():
        print(f"  {k}: AH={len(v['ah'])}  OU={len(v['ou'])}")

    # Test edilmeyen sinyal kaydı
    test_edilmeyen_kayitlar = {
        "Train": [], "Validation": [], "Arsiv": [],
    }
    for kod in TEST_EDILMEYEN:
        for bolme in test_edilmeyen_kayitlar:
            test_edilmeyen_kayitlar[bolme].append({
                "kod": kod, "n": 0, "not": "TEST EDİLMEDİ (protokol gereği)",
                "karar": "—",
            })

    sonuclar = {"Train": [], "Validation": [], "Arsiv": []}
    for bolme, data in bolmeler.items():
        df_ah = data["ah"]; df_ou = data["ou"]
        sinyaller = {}
        # AH sinyalleri → df_ah
        sinyaller.update(sinyal_executable_ah(df_ah))
        sinyaller.update(sinyal_benchmark_ah(df_ah))
        # OU sinyalleri → df_ou
        sinyaller.update(sinyal_executable_ou(df_ou))
        sinyaller.update(sinyal_benchmark_ou(df_ou))

        for kod, bets in sinyaller.items():
            met = analiz_sinyal(bets, kod)
            if met["n"] > 0:
                met["tip"] = SINYAL_META[kod]["tip"]
                met["market"] = SINYAL_META[kod]["market"]
                met["karar"] = karar_uygula(met) if bolme == "Validation" else "—"
                sonuclar[bolme].append(met)
        sonuclar[bolme].extend(test_edilmeyen_kayitlar[bolme])

    print(f"\n{CIKTI} yazılıyor...")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        for bolme in ("Train", "Validation", "Arsiv"):
            df_out = pd.DataFrame(sonuclar[bolme])
            if df_out.empty:
                df_out = pd.DataFrame(columns=["kod", "n", "not", "karar"])
            df_out = df_out.sort_values("kod") if "kod" in df_out.columns else df_out
            sheet = "Arsiv_2024_25" if bolme == "Arsiv" else bolme
            df_out.to_excel(w, sheet_name=sheet, index=False)

        # Karar_Ozeti — HER ZAMAN oluştur
        val_df = pd.DataFrame(sonuclar["Validation"])
        if not val_df.empty and "karar" in val_df.columns:
            guclu = val_df[val_df["karar"] == "GÜÇLÜ KANIT"]
        else:
            guclu = pd.DataFrame(columns=["kod", "n", "karar"])
        guclu.to_excel(w, sheet_name="Karar_Ozeti", index=False)

        pd.DataFrame([{
            "Protokol": "ROI/Edge v1.1 (FROZEN)",
            "Implementasyon": "v2",
            "Tarih": "2026-10-08",
            "Train": "2019/20 + 2020/21",
            "Validation": "2021/22 + 2022/23 + 2023/24",
            "Arsiv": "2024/25 (yalnızca raporlama)",
            "OOS": "2026/27 (KİLİTLİ — okunmadı)",
            "Sinyal_tanimli": 21,
            "Sinyal_test_edilen": 20,
            "Sinyal_test_edilmeyen": "S-OU-B3",
            "Stake": 1,
            "Bootstrap": "BCa 10.000, seed=20261008",
            "Permutation": "10.000, sadece sonuç etiketleri, iki taraflı",
            "Bankroll": 100,
            "MDD_eşik": 0.20,
            "N_min": 100,
            "Alpha": 0.05,
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} başarıyla yazıldı.")

    print("\n=== VALIDATION — GÜÇLÜ KANIT ===")
    val_df = pd.DataFrame(sonuclar["Validation"])
    if not val_df.empty and "karar" in val_df.columns:
        guclu = val_df[val_df["karar"] == "GÜÇLÜ KANIT"]
        if len(guclu) == 0:
            print("  Hiçbir sinyal GÜÇLÜ KANIT kriterlerini karşılamadı.")
        else:
            for _, r in guclu.iterrows():
                print(f"  {r['kod']:12s} N={r['n']:5d} ROI={r['roi']:+.4f} "
                      f"CI=[{r['roi_ci_lo']:+.4f},{r['roi_ci_hi']:+.4f}] "
                      f"p={r['perm_p']:.4f} MDD={r['mdd']:.4f}")


if __name__ == "__main__":
    main()