# -*- coding: utf-8 -*-
"""
confirmation_v1.py
Confirmation Protocol v1.0 (FROZEN) — birebir implementasyon
Tarih: 2026-10-08
Kapsam: 2023/24 (tek sezon)
Test sayısı: 12 primary (8 korelasyon + 4 Kruskal–Wallis)
FDR: 12 primary p üzerinde tek BH ailesi
Bootstrap: 10.000 BCa (bağımsız implementasyon)
Maç anahtarı: Season | Date | HomeTeam | AwayTeam
Eksik maç dışlama: aile bazlı (AH ve O/U ayrı)
Sezon tutarlılığı: UYGULANMAZ (tek sezon)
KISMİ testler için özel kural: YOK
"""

import numpy as np
import pandas as pd
from scipy.stats import kruskal, pearsonr, spearmanr, norm
import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

# ==================================================================
# 0. SABİTLER (KİLİTLİ — Confirmation Protocol v1.0)
# ==================================================================
SEZON_23_24 = "E0 (8).csv"
ENVANTER = "envanter_v2.xlsx"
ENVANTER_SHEET = "Eksik_Maclar"
CIKTI = "confirmation_v1_results.xlsx"

N_BOOTSTRAP = 10_000
RANDOM_SEED = 20261008
ALPHA_FDR = 0.05
ALPHA_BONF = 0.05 / 12  # referans; karar FDR'a göre

# Mutlak etki eşikleri (Validation v1.0 Bölüm 2 ile birebir aynı)
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

# Validation'da KISMİ olan testler (yalnızca raporlama amaçlı)
VALIDATION_KISMI = {"V-B3", "V-D1", "V-D4"}

# Test tanımları: (kod, x, y, aile) — Validation v1.0 ile birebir aynı
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
    """Season | Date | HomeTeam | AwayTeam (Validation ile birebir aynı)."""
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
    """Validation v1.0 Bölüm 1.3 ile birebir aynı türevler."""
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

    df["B365P_Over_Kayma_abs"]  = df["B365P_Over_Kayma"].abs()
    df["MaxAvg_Over_Kayma_abs"] = df["MaxAvg_Over_Kayma"].abs()

    # --- Yön (D ailesi) ---
    df["Over_yon"]  = np.where(df["Over_fark"]  < 0, "Dustu",
                       np.where(df["Over_fark"]  > 0, "Yukseldi", "Ayni"))
    df["Under_yon"] = np.where(df["Under_fark"] < 0, "Dustu",
                       np.where(df["Under_fark"] > 0, "Yukseldi", "Ayni"))
    return df


def veri_yukle() -> pd.DataFrame:
    df = pd.read_csv(SEZON_23_24, encoding="utf-8-sig")
    df["Season"] = "2023/24"

    # Envanter'deki eksik maçlar (aile bazlı)
    eksik = pd.read_excel(ENVANTER, sheet_name=ENVANTER_SHEET)
    eksik["_key"] = mac_anahtari(eksik)

    eksik_ou = set(eksik.loc[eksik["OU_Eksik"] == True, "_key"].dropna().astype(str))
    eksik_ah = set(eksik.loc[eksik["AH_Eksik"] == True, "_key"].dropna().astype(str))

    df["_key"] = mac_anahtari(df)
    df["_eksik_ah"] = df["_key"].isin(eksik_ah)
    df["_eksik_ou"] = df["_key"].isin(eksik_ou)

    df = _turetilmis(df)
    return df


def aile_filtrele(df: pd.DataFrame, aile: str) -> pd.DataFrame:
    if aile == "AH":
        return df[~df["_eksik_ah"]].copy()
    elif aile == "OU":
        return df[~df["_eksik_ou"]].copy()
    else:
        raise ValueError(f"Bilinmeyen aile: {aile}")


# ==================================================================
# 3. BCa BOOTSTRAP (Validation v1.0 ile birebir aynı)
# ==================================================================
def _bca_bootstrap_pearson(x: np.ndarray, y: np.ndarray) -> tuple:
    """Efron & Tibshirani BCa %95 — Pearson r. 10.000 resample, seed=20261008."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    rng = np.random.default_rng(RANDOM_SEED)

    theta_hat = float(pearsonr(x, y)[0])

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
        return np.nan, np.nan

    prop_less = np.mean(theta_boot < theta_hat)
    prop_less = min(max(prop_less, 1.0 / (B + 1)), B / (B + 1))
    z0 = norm.ppf(prop_less)

    theta_jack = np.empty(n, dtype=float)
    for i in range(n):
        xj = np.delete(x, i)
        yj = np.delete(y, i)
        theta_jack[i] = pearsonr(xj, yj)[0]
    theta_jack_mean = theta_jack.mean()
    num = ((theta_jack_mean - theta_jack) ** 3).sum()
    den = 6.0 * (((theta_jack_mean - theta_jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0

    alpha = 0.05
    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)

    def _bca_p(z_alpha):
        num_ = z0 + z_alpha
        den_ = 1 - a * num_
        if den_ == 0:
            return np.nan
        return float(norm.cdf(z0 + num_ / den_))

    p_lo = min(max(_bca_p(z_lo), 0.0), 1.0)
    p_hi = min(max(_bca_p(z_hi), 0.0), 1.0)

    ci_lo = float(np.quantile(theta_boot, p_lo))
    ci_hi = float(np.quantile(theta_boot, p_hi))
    return ci_lo, ci_hi


# ==================================================================
# 4. KORELASYON TESTİ
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
# 5. D TESTİ
# ==================================================================
def d_testi(df: pd.DataFrame, grup: str, deger: str, kod: str) -> dict:
    d = df[[grup, deger]].dropna()
    gruplar = {g: d[d[grup] == g][deger].values for g in ["Dustu", "Yukseldi", "Ayni"]}
    if any(len(v) < 5 for v in gruplar.values()):
        return {"kod": kod, "n": len(d), "hata": "yetersiz grup"}

    H, p_kw = kruskal(*gruplar.values())

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
# 6. KARAR MANTIĞI (tek sezon — Validation ile birebir aynı kriterler,
#    yalnızca sezon tutarlılığı kriteri uygulanmaz)
# ==================================================================
def _dunn_bh_uygula(dunn_raw) -> dict:
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


def karar_korelasyon(kod, sonuc, q):
    esik = ESIK[kod]

    isaret_ok = (np.sign(sonuc["pearson_r"])  == esik["isaret"]
                 and np.sign(sonuc["spearman_rho"]) == esik["isaret"])
    q_ok    = q < ALPHA_FDR
    boot_ok = sonuc["boot_sifir_disi"]
    r_ok    = abs(sonuc["pearson_r"])   >= esik["r"]
    rho_ok  = abs(sonuc["spearman_rho"]) >= esik["rho"]

    basarili = all([isaret_ok, q_ok, boot_ok, r_ok, rho_ok])
    kirmizi  = np.sign(sonuc["pearson_r"]) == -esik["isaret"]

    # KISMİ: Validation koduyla birebir aynı mantık — tek sezon uyarlaması
    # (Validation'da "anlamlı ama sezonlardan yalnızca biri" kriteriydi;
    #  Confirmation tek sezon olduğu için bu kriter uygulanamaz.
    #  Bu nedenle Confirmation'da KISMİ yalnızca "sınırda ama q anlamlı değil"
    #  yerine, protokolün izin verdiği tek biçim: "anlamlı ama etki eşiğinin
    #  altında ve işaret doğru" durumu BAŞARISIZ sayılır; KISMİ yoktur.)
    kismi = False

    return {
        "kod": kod,
        "isaret_ok": isaret_ok, "q_ok": q_ok, "boot_ok": boot_ok,
        "r_ok": r_ok, "rho_ok": rho_ok,
        "karar": "BAŞARILI" if basarili else "BAŞARISIZ",
        "kirmizi_bayrak": kirmizi,
    }


def karar_d(kod, sonuc, q):
    q_ok = q < ALPHA_FDR
    med  = sonuc["grup_medyan"]

    if kod in ("V-D1", "V-D3"):
        sira_ok = med["Dustu"] < med["Ayni"] < med["Yukseldi"]
    else:  # V-D2, V-D4
        sira_ok = med["Yukseldi"] > med["Dustu"] and med["Yukseldi"] > med["Ayni"]

    ph = _dunn_bh_uygula(sonuc["dunn_raw"])
    posthoc_ok = ph["posthoc_anlamli"]

    basarili = q_ok and sira_ok and posthoc_ok

    return {
        "kod": kod,
        "q_ok": q_ok, "sira_ok": sira_ok, "posthoc_ok": posthoc_ok,
        "karar": "BAŞARILI" if basarili else "BAŞARISIZ",
        "kirmizi_bayrak": False,
        "dunn_p_bh": ph["p_bh"],
    }


# ==================================================================
# 7. ANA AKIŞ
# ==================================================================
def main():
    print("Veri yükleniyor (2023/24)...")
    df = veri_yukle()

    df_ah = aile_filtrele(df, "AH")
    df_ou = aile_filtrele(df, "OU")
    print(f"  AH geçerli: {len(df_ah)}")
    print(f"  OU geçerli: {len(df_ou)}")

    # --- Korelasyon testleri ---
    print("\nKorelasyon testleri çalışıyor...")
    kor_sonuc = {}
    for kod, x, y, aile in KORELASYON_TESTLERI:
        g = df_ah if aile == "AH" else df_ou
        kor_sonuc[kod] = {"genel": korelasyon_testi(g, x, y, kod)}
        print(f"  {kod} tamam.")

    # --- D testleri ---
    print("\nD testleri çalışıyor...")
    d_sonuc = {}
    for kod, grup, deger, aile in D_TESTLERI:
        g = df_ah if aile == "AH" else df_ou
        d_sonuc[kod] = {"genel": d_testi(g, grup, deger, kod)}
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

    bonf_map = {kod: min(1.0, p * 12) for kod, p in zip(q_map.keys(), p_primary)}

    # --- Kararlar ---
    print("\nKararlar hesaplanıyor...")
    kararlar = []
    for kod, x, y, _ in KORELASYON_TESTLERI:
        k = karar_korelasyon(kod, kor_sonuc[kod]["genel"], q_map[kod])
        k["q"]    = q_map[kod]
        k["bonf"] = bonf_map[kod]
        kararlar.append(k)

    for kod, grup, deger, _ in D_TESTLERI:
        k = karar_d(kod, d_sonuc[kod]["genel"], q_map[kod])
        k["q"]    = q_map[kod]
        k["bonf"] = bonf_map[kod]
        kararlar.append(k)

    # --- Özet sayım ---
    n_basarili = sum(1 for k in kararlar if k["karar"] == "BAŞARILI")
    n_kismi    = sum(1 for k in kararlar if k["karar"] == "KISMİ")
    n_basarisiz= sum(1 for k in kararlar if k["karar"] == "BAŞARISIZ")
    n_kirmizi  = sum(1 for k in kararlar if k.get("kirmizi_bayrak", False))

    # Karar ağacı (Confirmation Protocol v1.0 Bölüm 5)
    if n_kirmizi > 0:
        sinif = "ZORUNLU REVIZYON"
    elif n_basarili >= 10:
        sinif = "STRONG CONFIRMATION"
    elif n_basarili >= 7:
        sinif = "PARTIAL CONFIRMATION"
    elif n_basarili >= 4:
        sinif = "CONFIRMATION BAŞARISIZ (metodoloji revizyonu / yeni Discovery)"
    else:
        sinif = "CONFIRMATION BAŞARISIZ (metodoloji revizyonu)"

    # --- Excel çıktısı ---
    print(f"\n{CIKTI} yazılıyor...")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:

        # Özet
        ozet = pd.DataFrame([{
            "Test":              k["kod"],
            "Karar":             k["karar"],
            "q":                 k["q"],
            "Bonferroni":        k["bonf"],
            "Validation_Kismi":  k["kod"] in VALIDATION_KISMI,
            "Kirmizi_Bayrak":    k.get("kirmizi_bayrak", False),
        } for k in kararlar])
        ozet.to_excel(w, sheet_name="Ozet", index=False)

        # Korelasyon test sheet'leri
        for kod, x, y, _ in KORELASYON_TESTLERI:
            s = kor_sonuc[kod]["genel"]
            pd.DataFrame([{
                "Sezon":        "2023/24",
                "n":            s.get("n"),
                "Pearson_r":    s.get("pearson_r"),
                "Pearson_p":    s.get("pearson_p"),
                "Spearman_rho": s.get("spearman_rho"),
                "Spearman_p":   s.get("spearman_p"),
                "Boot_lo":      s.get("boot_lo"),
                "Boot_hi":      s.get("boot_hi"),
            }]).to_excel(w, sheet_name=kod, index=False)

        # D test sheet'leri
        for kod, grup, deger, _ in D_TESTLERI:
            s = d_sonuc[kod]["genel"]
            pd.DataFrame([{
                "Sezon":           "2023/24",
                "n":               s.get("n"),
                "KW_H":            s.get("kw_H"),
                "KW_p":            s.get("kw_p"),
                "Medyan_Dustu":    s["grup_medyan"]["Dustu"],
                "Medyan_Yukseldi": s["grup_medyan"]["Yukseldi"],
                "Medyan_Ayni":     s["grup_medyan"]["Ayni"],
                "n_Dustu":         s["grup_n"]["Dustu"],
                "n_Yukseldi":      s["grup_n"]["Yukseldi"],
                "n_Ayni":          s["grup_n"]["Ayni"],
            }]).to_excel(w, sheet_name=kod, index=False)

        # Notlar
        pd.DataFrame([{
            "Protokol":         "Confirmation Protocol v1.0 (FROZEN)",
            "Tarih":            "2026-10-08",
            "Veri":             "2023/24 (tek sezon)",
            "Test_sayisi":      12,
            "FDR":              "BH, q<0.05, 12 primary p",
            "Bonferroni":       "alpha/12 = 0.00417 (duyarlılık)",
            "Bootstrap":        "10.000 BCa (bağımsız implementasyon)",
            "Sezon_tutarliligi":"UYGULANMAZ (tek sezon)",
            "Kismi_ozel_kural": "YOK (Validation v1.0 ile birebir aynı)",
            "Kirmizi_bayrak":   "isaret ters donmesi",
            "Karar_agaci":      "10-12 STRONG / 7-9 PARTIAL / 4-6 FAIL / 0-3 FAIL",
            "Ozet_Basarili":    n_basarili,
            "Ozet_Kismi":       n_kismi,
            "Ozet_Basarisiz":   n_basarisiz,
            "Ozet_Kirmizi":     n_kirmizi,
            "Siniflandirma":    sinif,
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} başarıyla yazıldı.")
    print("\nÖzet kararlar:")
    for k in kararlar:
        vk = " [Validation: KISMİ]" if k["kod"] in VALIDATION_KISMI else ""
        print(f"  {k['kod']:6s} → {k['karar']:10s} (q={k['q']:.4f}){vk}")

    print(f"\nToplam: {n_basarili} BAŞARILI / {n_kismi} KISMİ / {n_basarisiz} BAŞARISIZ / {n_kirmizi} KIRMIZI BAYRAK")
    print(f"Sınıflandırma: {sinif}")


if __name__ == "__main__":
    main()