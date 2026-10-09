# -*- coding: utf-8 -*-
"""
validation_v1.py
Validation Protocol v1.0 (FROZEN) — birebir implementasyon
Tarih: 2026-10-08
Kapsam: 2021/22 + 2022/23
Test sayısı: 12 primary (8 korelasyon + 4 Kruskal–Wallis)
FDR: 12 primary p üzerinde tek BH ailesi
Bootstrap: 10.000 BCa (bağımsız implementasyon, SciPy API'sine bağımlı değil)
Maç anahtarı: Season | Date | HomeTeam | AwayTeam
Eksik maç dışlama: aile bazlı (AH ve O/U ayrı)
"""

import numpy as np
import pandas as pd
from scipy.stats import kruskal, pearsonr, spearmanr, norm
import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

# ==================================================================
# 0. SABİTLER (KİLİTLİ — protokolden)
# ==================================================================
SEZON_21_22 = "E0 (6).csv"
SEZON_22_23 = "E0 (7).csv"
ENVANTER = "envanter_v2.xlsx"
ENVANTER_SHEET = "Eksik_Maclar"
CIKTI = "validation_v1_results.xlsx"

N_BOOTSTRAP = 10_000
RANDOM_SEED = 20261008
ALPHA_FDR = 0.05
ALPHA_BONF = 0.05 / 12  # referans; karar FDR'a göre

# Mutlak etki eşikleri (protokol Bölüm 2 — değiştirilemez)
ESIK = {
    "V-A1": {"r": 0.20, "rho": 0.15, "isaret": +1},
    "V-A2": {"r": 0.20, "rho": 0.15, "isaret": +1},
    "V-B1": {"r": 0.50, "rho": 0.60, "isaret": -1},
    "V-B2": {"r": 0.10, "rho": 0.10, "isaret": +1},
    "V-B3": {"r": 0.08, "rho": 0.15, "isaret": -1},
    "V-C1": {"r": 0.30, "rho": 0.40, "isaret": +1},
    "V-C2": {"r": 0.10, "rho": 0.08, "isaret": +1},
    "V-C3": {"r": 0.10, "rho": 0.10, "isaret": +1},
    "V-D1": {"isaret": -1, "sira": ["Dustu", "Ayni", "Yukseldi"]},
    "V-D2": {"isaret": +1, "sira": ["Dustu", "Ayni", "Yukseldi"]},
    "V-D3": {"isaret": -1, "sira": ["Dustu", "Ayni", "Yukseldi"]},
    "V-D4": {"isaret": +1, "sira": ["Dustu", "Ayni", "Yukseldi"]},
}

# Test tanımları: (kod, x, y, aile)
KORELASYON_TESTLERI = [
    ("V-A1", "AH_line_abs", "B365_Home_abs",         "AH"),
    ("V-A2", "AH_line_abs", "B365_Away_abs",         "AH"),
    ("V-B1", "Over_fark",   "Under_fark",            "OU"),
    ("V-B2", "Over_fark",   "MaxAvg_Over_Kayma",     "OU"),
    ("V-B3", "Over_fark",   "MaxAvg_Under_Kayma",    "OU"),
    ("V-C1", "Over_abs",    "Under_abs",             "OU"),
    ("V-C2", "Over_abs",    "B365P_Over_Kayma_abs",  "OU"),
    ("V-C3", "Over_abs",    "MaxAvg_Over_Kayma_abs", "OU"),
]

D_TESTLERI = [
    ("V-D1", "Over_yon",  "B365P_Over_Kayma",   "OU"),
    ("V-D2", "Over_yon",  "MaxAvg_Over_Kayma",  "OU"),
    ("V-D3", "Under_yon", "B365P_Under_Kayma",  "OU"),
    ("V-D4", "Under_yon", "MaxAvg_Under_Kayma", "OU"),
]


# ==================================================================
# 1. MAÇ ANAHTARI
# ==================================================================
def mac_anahtari(df: pd.DataFrame) -> pd.Series:
    """
    Gerçek maç kimliği: Season | Date | HomeTeam | AwayTeam
    envanter_v2.py'nin Eksik_Maclar sayfasındaki 4 alanla aynı.
    """
    tarih = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
    return (
        df["Season"].astype(str).str.strip()
        + "|"
        + tarih.fillna("NA")
        + "|"
        + df["HomeTeam"].astype(str).str.strip()
        + "|"
        + df["AwayTeam"].astype(str).str.strip()
    )


# ==================================================================
# 2. VERİ YÜKLEME (aile bazlı eksik maç işaretleme)
# ==================================================================
def _turetilmis(df: pd.DataFrame) -> pd.DataFrame:
    """Protokol Bölüm 1.3 — türetilmiş değişkenler (birebir)."""
    df = df.copy()

    # --- AH ---
    df["AH_line_fark"]   = df["AHCh"] - df["AHh"]
    df["AH_line_abs"]    = df["AH_line_fark"].abs()
    df["B365_Home_fark"] = df["B365CAHH"] - df["B365AHH"]
    df["B365_Away_fark"] = df["B365CAHA"] - df["B365AHA"]
    df["B365_Home_abs"]  = df["B365_Home_fark"].abs()
    df["B365_Away_abs"]  = df["B365_Away_fark"].abs()

    # --- O/U ---
    df["Over_fark"]  = df["B365C>2.5"] - df["B365>2.5"]
    df["Under_fark"] = df["B365C<2.5"] - df["B365<2.5"]
    df["Over_abs"]   = df["Over_fark"].abs()
    df["Under_abs"]  = df["Under_fark"].abs()

    df["B365P_Over_Kayma"]   = (df["B365C>2.5"] - df["PC>2.5"]) - (df["B365>2.5"] - df["P>2.5"])
    df["B365P_Under_Kayma"]  = (df["B365C<2.5"] - df["PC<2.5"]) - (df["B365<2.5"] - df["P<2.5"])
    df["MaxAvg_Over_Kayma"]  = (df["MaxC>2.5"] - df["AvgC>2.5"]) - (df["Max>2.5"] - df["Avg>2.5"])
    df["MaxAvg_Under_Kayma"] = (df["MaxC<2.5"] - df["AvgC<2.5"]) - (df["Max<2.5"] - df["Avg<2.5"])

    # V-C2 / V-C3 için abs türevleri
    df["B365P_Over_Kayma_abs"]  = df["B365P_Over_Kayma"].abs()
    df["MaxAvg_Over_Kayma_abs"] = df["MaxAvg_Over_Kayma"].abs()

    # --- Yön (D ailesi) ---
    df["Over_yon"]  = np.where(df["Over_fark"]  < 0, "Dustu",
                       np.where(df["Over_fark"]  > 0, "Yukseldi", "Ayni"))
    df["Under_yon"] = np.where(df["Under_fark"] < 0, "Dustu",
                       np.where(df["Under_fark"] > 0, "Yukseldi", "Ayni"))
    return df


def veri_yukle() -> tuple[pd.DataFrame, pd.DataFrame]:
    df1 = pd.read_csv(SEZON_21_22, encoding="utf-8-sig")
    df2 = pd.read_csv(SEZON_22_23, encoding="utf-8-sig")
    df1["Season"] = "2021/22"
    df2["Season"] = "2022/23"

    # Envanter'deki eksik maçlar (aile bazlı)
    eksik = pd.read_excel(ENVANTER, sheet_name=ENVANTER_SHEET)
    eksik["_key"] = mac_anahtari(eksik)

    eksik_ou = set(eksik.loc[eksik["OU_Eksik"] == True, "_key"].dropna().astype(str))
    eksik_ah = set(eksik.loc[eksik["AH_Eksik"] == True, "_key"].dropna().astype(str))

    for df in (df1, df2):
        df["_key"] = mac_anahtari(df)
        df["_eksik_ah"] = df["_key"].isin(eksik_ah)
        df["_eksik_ou"] = df["_key"].isin(eksik_ou)

    df1 = _turetilmis(df1)
    df2 = _turetilmis(df2)
    return df1, df2


def aile_filtrele(df: pd.DataFrame, aile: str) -> pd.DataFrame:
    """
    Aileye göre eksik maçları dışla:
    - AH testleri → AH eksik satırlar dışlanır
    - OU testleri → OU eksik satırlar dışlanır
    """
    if aile == "AH":
        return df[~df["_eksik_ah"]].copy()
    elif aile == "OU":
        return df[~df["_eksik_ou"]].copy()
    else:
        raise ValueError(f"Bilinmeyen aile: {aile}")


# ==================================================================
# 3. BCa BOOTSTRAP (bağımsız implementasyon)
# ==================================================================
def _bca_bootstrap_pearson(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Efron & Tibshirani BCa %95 güven aralığı — Pearson r için.
    Protokol: 10.000 resample, BCa, seed=20261008.
    SciPy bootstrap API'sine bağımlı değil (SciPy >= 1.16 / Python 3.14 uyumu).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    rng = np.random.default_rng(RANDOM_SEED)

    # 1) Orijinal istatistik
    theta_hat = float(pearsonr(x, y)[0])

    # 2) Bootstrap dağılımı
    theta_boot = np.empty(N_BOOTSTRAP, dtype=float)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            theta_boot[b] = np.nan
        else:
            theta_boot[b] = pearsonr(xb, yb)[0]

    theta_boot = theta_boot[~np.isnan(theta_boot)]
    B = len(theta_boot)
    if B < 100:
        # Aşırı dejenere durum — protokol dışı, güvenli geri dönüş
        return np.nan, np.nan

    # 3) Bias correction z0
    prop_less = np.mean(theta_boot < theta_hat)
    prop_less = min(max(prop_less, 1.0 / (B + 1)), B / (B + 1))
    z0 = norm.ppf(prop_less)

    # 4) Acceleration a (jackknife)
    theta_jack = np.empty(n, dtype=float)
    for i in range(n):
        xj = np.delete(x, i)
        yj = np.delete(y, i)
        theta_jack[i] = pearsonr(xj, yj)[0]
    theta_jack_mean = theta_jack.mean()
    num = ((theta_jack_mean - theta_jack) ** 3).sum()
    den = 6.0 * (((theta_jack_mean - theta_jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0

    # 5) BCa uçları
    alpha = 0.05  # %95 GA
    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)

    def _bca_p(z_alpha):
        num_ = z0 + z_alpha
        den_ = 1 - a * num_
        if den_ == 0:
            return np.nan
        return float(norm.cdf(z0 + num_ / den_))

    p_lo = _bca_p(z_lo)
    p_hi = _bca_p(z_hi)
    p_lo = min(max(p_lo, 0.0), 1.0)
    p_hi = min(max(p_hi, 0.0), 1.0)

    ci_lo = float(np.quantile(theta_boot, p_lo))
    ci_hi = float(np.quantile(theta_boot, p_hi))
    return ci_lo, ci_hi


# ==================================================================
# 4. KORELASYON TESTİ (A / B / C aileleri)
# ==================================================================
def korelasyon_testi(df: pd.DataFrame, x: str, y: str, kod: str) -> dict:
    d = df[[x, y]].dropna()
    n = len(d)
    if n < 30:
        return {"kod": kod, "n": n, "hata": "yetersiz veri"}

    xv, yv = d[x].values, d[y].values
    r, p_r = pearsonr(xv, yv)
    rho, p_rho = spearmanr(xv, yv)
    ci_lo, ci_hi = _bca_bootstrap_pearson(xv, yv)

    return {
        "kod": kod, "n": n,
        "pearson_r": r, "pearson_p": p_r,
        "spearman_rho": rho, "spearman_p": p_rho,
        "boot_lo": ci_lo, "boot_hi": ci_hi,
        "boot_sifir_disi": (ci_lo > 0) or (ci_hi < 0),
    }


# ==================================================================
# 5. D TESTİ (Kruskal–Wallis primary + Dunn post-hoc)
# ==================================================================
def d_testi(df: pd.DataFrame, grup: str, deger: str, kod: str) -> dict:
    d = df[[grup, deger]].dropna()
    gruplar = {g: d[d[grup] == g][deger].values for g in ["Dustu", "Yukseldi", "Ayni"]}
    if any(len(v) < 5 for v in gruplar.values()):
        return {"kod": kod, "n": len(d), "hata": "yetersiz grup"}

    H, p_kw = kruskal(*gruplar.values())

    # LOCKED protokol: Dunn YALNIZCA KW anlamlıysa hesaplanır.
    dunn_raw = None
    if p_kw < ALPHA_FDR:
        dunn_raw = sp.posthoc_dunn(
            d,
            val_col=deger,
            group_col=grup,
            p_adjust=None,
        )

    return {
        "kod": kod, "n": len(d),
        "kw_H": H, "kw_p": p_kw,
        "dunn_raw": dunn_raw,
        "grup_medyan": {g: float(np.median(v)) for g, v in gruplar.items()},
        "grup_n":      {g: int(len(v))         for g, v in gruplar.items()},
    }


# ==================================================================
# 6. KARAR MANTIĞI
# ==================================================================
def _dunn_bh_uygula(dunn_raw) -> dict:
    """
    3 çift (Dustu-Yukseldi, Dustu-Ayni, Yukseldi-Ayni) için
    BH düzeltmesi TEK SEFERDE uygulanır.
    dunn_raw None ise (KW anlamsız) → post-hoc yok.
    """
    if dunn_raw is None:
        return {"p_bh": {}, "posthoc_anlamli": False}

    ciftler = [("Dustu", "Yukseldi"), ("Dustu", "Ayni"), ("Yukseldi", "Ayni")]
    p_ham = []
    gecerli = []
    for a, b in ciftler:
        if a in dunn_raw.index and b in dunn_raw.columns:
            p = dunn_raw.loc[a, b]
            if pd.notna(p):
                p_ham.append(float(p))
                gecerli.append((a, b))

    if not p_ham:
        return {"p_bh": {}, "posthoc_anlamli": False}

    _, p_bh, _, _ = multipletests(p_ham, alpha=ALPHA_FDR, method="fdr_bh")
    return {
        "p_bh": {f"{a}-{b}": float(p) for (a, b), p in zip(gecerli, p_bh)},
        "posthoc_anlamli": any(p < ALPHA_FDR for p in p_bh),
    }


def karar_korelasyon(kod, sonuc, q, s21, s22):
    esik = ESIK[kod]

    isaret_ok = (np.sign(sonuc["pearson_r"])  == esik["isaret"]
                 and np.sign(sonuc["spearman_rho"]) == esik["isaret"])
    q_ok      = q < ALPHA_FDR
    boot_ok   = sonuc["boot_sifir_disi"]
    r_ok      = abs(sonuc["pearson_r"])   >= esik["r"]
    rho_ok    = abs(sonuc["spearman_rho"]) >= esik["rho"]

    s21_ok = np.sign(s21.get("pearson_r", 0)) == esik["isaret"]
    s22_ok = np.sign(s22.get("pearson_r", 0)) == esik["isaret"]
    sezon_tutarli = s21_ok and s22_ok

    basarili = all([isaret_ok, q_ok, boot_ok, r_ok, rho_ok, sezon_tutarli])
    kirmizi  = np.sign(sonuc["pearson_r"]) == -esik["isaret"]
    kismi    = (not basarili) and q_ok and (s21_ok != s22_ok)

    return {
        "kod": kod,
        "isaret_ok": isaret_ok, "q_ok": q_ok, "boot_ok": boot_ok,
        "r_ok": r_ok, "rho_ok": rho_ok, "sezon_tutarli": sezon_tutarli,
        "karar": "BAŞARILI" if basarili else ("KISMİ" if kismi else "BAŞARISIZ"),
        "kirmizi_bayrak": kirmizi,
    }


def karar_d(kod, sonuc, q, s21, s22):
    q_ok = q < ALPHA_FDR
    med  = sonuc["grup_medyan"]

    if kod in ("V-D1", "V-D3"):
        sira_ok     = med["Dustu"] < med["Ayni"] < med["Yukseldi"]
        s21_sira_ok = s21["grup_medyan"]["Dustu"] < s21["grup_medyan"]["Ayni"] < s21["grup_medyan"]["Yukseldi"]
        s22_sira_ok = s22["grup_medyan"]["Dustu"] < s22["grup_medyan"]["Ayni"] < s22["grup_medyan"]["Yukseldi"]
    else:  # V-D2, V-D4
        sira_ok     = med["Yukseldi"] > med["Dustu"] and med["Yukseldi"] > med["Ayni"]
        s21_sira_ok = (s21["grup_medyan"]["Yukseldi"] > s21["grup_medyan"]["Dustu"]
                       and s21["grup_medyan"]["Yukseldi"] > s21["grup_medyan"]["Ayni"])
        s22_sira_ok = (s22["grup_medyan"]["Yukseldi"] > s22["grup_medyan"]["Dustu"]
                       and s22["grup_medyan"]["Yukseldi"] > s22["grup_medyan"]["Ayni"])

    sezon_tutarli = s21_sira_ok and s22_sira_ok

    ph = _dunn_bh_uygula(sonuc["dunn_raw"])
    posthoc_ok = ph["posthoc_anlamli"]

    basarili = q_ok and sira_ok and posthoc_ok and sezon_tutarli
    kismi    = (not basarili) and q_ok and (s21_sira_ok != s22_sira_ok)

    return {
        "kod": kod,
        "q_ok": q_ok, "sira_ok": sira_ok, "posthoc_ok": posthoc_ok,
        "sezon_tutarli": sezon_tutarli,
        "karar": "BAŞARILI" if basarili else ("KISMİ" if kismi else "BAŞARISIZ"),
        "kirmizi_bayrak": False,
        "dunn_p_bh": ph["p_bh"],
    }


# ==================================================================
# 7. ANA AKIŞ
# ==================================================================
def main():
    print("Veri yükleniyor...")
    df1, df2 = veri_yukle()

    # Aile bazlı geçerli satır sayıları
    df1_ah = aile_filtrele(df1, "AH"); df2_ah = aile_filtrele(df2, "AH")
    df1_ou = aile_filtrele(df1, "OU"); df2_ou = aile_filtrele(df2, "OU")
    df_all_ah = pd.concat([df1_ah, df2_ah], ignore_index=True)
    df_all_ou = pd.concat([df1_ou, df2_ou], ignore_index=True)

    print(f"  AH geçerli: 2021/22={len(df1_ah)}, 2022/23={len(df2_ah)}, toplam={len(df_all_ah)}")
    print(f"  OU geçerli: 2021/22={len(df1_ou)}, 2022/23={len(df2_ou)}, toplam={len(df_all_ou)}")

    # --- Korelasyon testleri ---
    print("\nKorelasyon testleri çalışıyor...")
    kor_sonuc = {}
    for kod, x, y, aile in KORELASYON_TESTLERI:
        if aile == "AH":
            g_all, g_21, g_22 = df_all_ah, df1_ah, df2_ah
        else:
            g_all, g_21, g_22 = df_all_ou, df1_ou, df2_ou
        kor_sonuc[kod] = {
            "genel": korelasyon_testi(g_all, x, y, kod),
            "s21":   korelasyon_testi(g_21,  x, y, kod),
            "s22":   korelasyon_testi(g_22,  x, y, kod),
        }
        print(f"  {kod} tamam.")

    # --- D testleri ---
    print("\nD testleri çalışıyor...")
    d_sonuc = {}
    for kod, grup, deger, aile in D_TESTLERI:
        if aile == "AH":
            g_all, g_21, g_22 = df_all_ah, df1_ah, df2_ah
        else:
            g_all, g_21, g_22 = df_all_ou, df1_ou, df2_ou
        d_sonuc[kod] = {
            "genel": d_testi(g_all, grup, deger, kod),
            "s21":   d_testi(g_21,  grup, deger, kod),
            "s22":   d_testi(g_22,  grup, deger, kod),
        }
        print(f"  {kod} tamam.")

    # --- FDR: 12 primary p üzerinde TEK aile ---
    print("\nFDR (BH) uygulanıyor — 12 primary p...")
    p_primary = []
    for kod, *_ in KORELASYON_TESTLERI:
        p_primary.append(kor_sonuc[kod]["genel"]["pearson_p"])
    for kod, *_ in D_TESTLERI:
        p_primary.append(d_sonuc[kod]["genel"]["kw_p"])

    assert len(p_primary) == 12, "Primary p sayısı 12 olmalı!"

    _, q_vals, _, _ = multipletests(p_primary, alpha=ALPHA_FDR, method="fdr_bh")
    q_map = {}
    idx = 0
    for kod, *_ in KORELASYON_TESTLERI:
        q_map[kod] = q_vals[idx]; idx += 1
    for kod, *_ in D_TESTLERI:
        q_map[kod] = q_vals[idx]; idx += 1

    # Bonferroni (duyarlılık raporu)
    bonf_map = {kod: min(1.0, p * 12) for kod, p in zip(q_map.keys(), p_primary)}

    # --- Kararlar ---
    print("\nKararlar hesaplanıyor...")
    kararlar = []
    for kod, x, y, _ in KORELASYON_TESTLERI:
        k = karar_korelasyon(
            kod,
            kor_sonuc[kod]["genel"],
            q_map[kod],
            kor_sonuc[kod]["s21"],
            kor_sonuc[kod]["s22"],
        )
        k["q"]    = q_map[kod]
        k["bonf"] = bonf_map[kod]
        kararlar.append(k)

    for kod, grup, deger, _ in D_TESTLERI:
        k = karar_d(
            kod,
            d_sonuc[kod]["genel"],
            q_map[kod],
            d_sonuc[kod]["s21"],
            d_sonuc[kod]["s22"],
        )
        k["q"]    = q_map[kod]
        k["bonf"] = bonf_map[kod]
        kararlar.append(k)

    # --- Excel çıktısı ---
    print(f"\n{CIKTI} yazılıyor...")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:

        # Özet
        ozet = pd.DataFrame([{
            "Test":           k["kod"],
            "Karar":          k["karar"],
            "q":              k["q"],
            "Bonferroni":     k["bonf"],
            "Kirmizi_Bayrak": k.get("kirmizi_bayrak", False),
        } for k in kararlar])
        ozet.to_excel(w, sheet_name="Ozet", index=False)

        # Korelasyon test sheet'leri
        for kod, x, y, _ in KORELASYON_TESTLERI:
            g   = kor_sonuc[kod]["genel"]
            s21 = kor_sonuc[kod]["s21"]
            s22 = kor_sonuc[kod]["s22"]
            rows = []
            for etiket, s in [("Genel", g), ("2021/22", s21), ("2022/23", s22)]:
                rows.append({
                    "Sezon":        etiket,
                    "n":            s.get("n"),
                    "Pearson_r":    s.get("pearson_r"),
                    "Pearson_p":    s.get("pearson_p"),
                    "Spearman_rho": s.get("spearman_rho"),
                    "Spearman_p":   s.get("spearman_p"),
                    "Boot_lo":      s.get("boot_lo"),
                    "Boot_hi":      s.get("boot_hi"),
                })
            pd.DataFrame(rows).to_excel(w, sheet_name=kod, index=False)

        # D test sheet'leri
        for kod, grup, deger, _ in D_TESTLERI:
            rows = []
            for etiket, s in [("Genel",   d_sonuc[kod]["genel"]),
                              ("2021/22", d_sonuc[kod]["s21"]),
                              ("2022/23", d_sonuc[kod]["s22"])]:
                rows.append({
                    "Sezon":           etiket,
                    "n":               s.get("n"),
                    "KW_H":            s.get("kw_H"),
                    "KW_p":            s.get("kw_p"),
                    "Medyan_Dustu":    s["grup_medyan"]["Dustu"],
                    "Medyan_Yukseldi": s["grup_medyan"]["Yukseldi"],
                    "Medyan_Ayni":     s["grup_medyan"]["Ayni"],
                    "n_Dustu":         s["grup_n"]["Dustu"],
                    "n_Yukseldi":      s["grup_n"]["Yukseldi"],
                    "n_Ayni":          s["grup_n"]["Ayni"],
                })
            pd.DataFrame(rows).to_excel(w, sheet_name=kod, index=False)

        # Notlar
        pd.DataFrame([{
            "Protokol":       "Validation Protocol v1.0 (FROZEN)",
            "Tarih":          "2026-10-08",
            "Test_sayisi":    12,
            "FDR":            "BH, q<0.05, 12 primary p",
            "Bonferroni":     "alpha/12 = 0.00417 (duyarlılık)",
            "Bootstrap":      "10.000 BCa (bağımsız implementasyon)",
            "Kirmizi_bayrak": "isaret ters donmesi",
            "Mac_anahtari":   "Season|Date|HomeTeam|AwayTeam",
            "Dunn_kurali":    "Yalnizca KW anlamliysa; kendi 3'lu ailesinde BH",
            "Eksik_dislama":  "Aile bazli (AH ve OU ayri)",
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} başarıyla yazıldı.")
    print("\nÖzet kararlar:")
    for k in kararlar:
        print(f"  {k['kod']:6s} → {k['karar']:10s} (q={k['q']:.4f})")


if __name__ == "__main__":
    main()