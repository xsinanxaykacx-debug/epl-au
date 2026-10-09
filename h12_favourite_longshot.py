# -*- coding: utf-8 -*-
"""
h12_favourite_longshot.py
H12 FROZEN v1.0 — Favourite-Longshot Bias
Tarih: 2026-10-08

Hipotez:
EPL'de B365 1X2 açılış oranı düşük seviyelerde (favoriler),
implied probability ile gerçekleşen sonuç arasında sistematik
pozitif sapma vardır. Belirli oran bantlarında favori tarafa
açılış oranıyla bahis oynandığında pozitif ROI elde edilir.

Test yapısı:
- Favori: her maçta en düşük B365 1X2 oranlı taraf (H/D/A)
- Favori eşitliği: H > D > A (deterministik, np.argmin ile)
- 6 oran bandı: [1.00-1.20], (1.20-1.40], (1.40-1.60],
                (1.60-1.80], (1.80-2.00], (2.00-+inf]
- Bahis: favori tarafa, açılış oranıyla
- Stake: 1 birim sabit
- Settlement: doğru → +(oran-1); yanlış → -1
- Her maçta 1 bahis
- 6 test (6 bant)
- BH-FDR tek aile, q < 0.05

KİLİTLER:
- E0 (10).csv ASLA OKUNMAZ.
- Bant sınırları FROZEN.
- Favori eşitliği: H > D > A (protokol addendum 2026-10-08).
- Bootstrap: BCa 10.000, seed=20261008
- Permutation: FTR etiketleri, 10.000, iki taraflı, seed=20261008
- MDD: bankroll 100, stable kronolojik
- N_MIN = 100, ALPHA = 0.05, MDD_ESIK = 0.20
- KIRMIZI BAYRAK raporlanır (Train ROI>0 ∧ OOS ROI<0).
- KARAR HAM DEĞERLERLE VERİLİR (yuvarlama yalnız gösterimde).
- Post-hoc değişiklik YASAK.

Not:
H12 kapanış (B365CH/CD/CA) kullanmaz. Bu yüzden 2015/16-2024/25
(10 sezon) kullanılır. H10/H11'den bağımsızdır.

ÇIKTI: h12_favourite_longshot_results.xlsx
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ==================================================================
# 0. SABİTLER (KİLİTLİ)
# ==================================================================
CIKTI = "h12_favourite_longshot_results.xlsx"

YASAKLI_DOSYALAR = {
    "E0 (10).csv",   # 2026/27 — kör OOS
}

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

TRAIN_SEZONLAR = ["2015/16", "2016/17", "2017/18", "2018/19",
                  "2019/20", "2020/21", "2021/22", "2022/23"]
VALIDATION_SEZON = "2023/24"
OOS_SEZON = "2024/25"

# Oran bantları — FROZEN
BANT_SINIRLARI = [1.00, 1.20, 1.40, 1.60, 1.80, 2.00, np.inf]
BANT_ETIKETLERI = [
    "[1.00, 1.20]",
    "(1.20, 1.40]",
    "(1.40, 1.60]",
    "(1.60, 1.80]",
    "(1.80, 2.00]",
    "(2.00, +inf]",
]

N_BOOTSTRAP = 10_000
N_PERMUTATION = 10_000
RANDOM_SEED = 20261008
BANKROLL = 100.0
MDD_ESIK = 0.20
ALPHA = 0.05
N_MIN = 100

BEKLENEN_TOPLAM = 3800
BEKLENEN_TRAIN = 3039
BEKLENEN_VAL = 380
BEKLENEN_OOS = 380

_okuma_sayaci = defaultdict(int)


# ==================================================================
# 1. VERİ YÜKLEME
# ==================================================================
def _guvenli_oku(dosya):
    ad = os.path.basename(dosya)
    if ad in YASAKLI_DOSYALAR:
        raise RuntimeError(f"YASAKLI DOSYA OKUNAMAZ: {ad}")
    _okuma_sayaci[ad] += 1
    return pd.read_csv(dosya, encoding="utf-8-sig")


def veri_yukle():
    print("\n--- Veri yükleniyor ---")

    GEREKLI = ["Date", "FTR", "B365H", "B365D", "B365A"]

    frames = []
    for dosya, sezon in TUM_DOSYALAR:
        if not os.path.exists(dosya):
            raise RuntimeError(f"EKSİK DOSYA: {dosya}")
        df = _guvenli_oku(dosya)
        eksik = [c for c in GEREKLI if c not in df.columns]
        if eksik:
            raise RuntimeError(f"{dosya}: eksik sütunlar {eksik}")
        df["Season"] = sezon
        frames.append(df[["Season"] + GEREKLI].copy())

    df = pd.concat(frames, ignore_index=True)
    print(f"  Ham toplam satır: {len(df)}")
    if len(df) != BEKLENEN_TOPLAM:
        raise RuntimeError(f"Ham toplam {len(df)} != {BEKLENEN_TOPLAM}")

    print("\n  Temizlik raporu:")
    for c in ["B365H", "B365D", "B365A", "FTR", "Date"]:
        n_nan = int(df[c].isna().sum())
        print(f"    {c:8s} NaN={n_nan}")

    gecerli = (
        df["B365H"].notna() & (df["B365H"] > 0) &
        df["B365D"].notna() & (df["B365D"] > 0) &
        df["B365A"].notna() & (df["B365A"] > 0) &
        df["FTR"].notna() &
        df["Date"].notna()
    )
    n_dusen = int((~gecerli).sum())
    print(f"  Toplam düşen satır: {n_dusen}")
    df = df[gecerli].copy().reset_index(drop=True)
    print(f"  Temizlik sonrası: {len(df)}")

    return df


# ==================================================================
# 2. FAVORİ VE BANT
# ==================================================================
def favori_belirle(df):
    """
    Her maç için favori taraf (en düşük B365 oranlı).
    Favori eşitliği: H > D > A (np.argmin doğal önceliği).
    0 = H, 1 = D, 2 = A
    """
    df = df.copy()
    oranlar = df[["B365H", "B365D", "B365A"]].values
    fav_idx = np.argmin(oranlar, axis=1)
    df["fav_idx"] = fav_idx
    df["fav_oran"] = oranlar[np.arange(len(df)), fav_idx]
    df["fav_taraf"] = np.where(fav_idx == 0, "H",
                       np.where(fav_idx == 1, "D", "A"))
    return df


def bant_ata(oran):
    """Oranı bant etiketine çevir."""
    for i in range(len(BANT_SINIRLARI) - 1):
        alt = BANT_SINIRLARI[i]
        ust = BANT_SINIRLARI[i + 1]
        if i == 0:
            if oran >= alt and oran <= ust:
                return BANT_ETIKETLERI[i]
        else:
            if oran > alt and oran <= ust:
                return BANT_ETIKETLERI[i]
    return None


# ==================================================================
# 3. BÖLME
# ==================================================================
def bolmeleri_ayir(df):
    train = df[df["Season"].isin(TRAIN_SEZONLAR)].copy().reset_index(drop=True)
    val = df[df["Season"] == VALIDATION_SEZON].copy().reset_index(drop=True)
    oos = df[df["Season"] == OOS_SEZON].copy().reset_index(drop=True)

    print(f"\n  Bölme boyutları:")
    print(f"    Train: {len(train)} (beklenen {BEKLENEN_TRAIN})")
    print(f"    Val:   {len(val)} (beklenen {BEKLENEN_VAL})")
    print(f"    OOS:   {len(oos)} (beklenen {BEKLENEN_OOS})")

    return train, val, oos


# ==================================================================
# 4. BANT ANALİZİ (gösterim)
# ==================================================================
def bant_analiz(df_bolme, etiket):
    sonuclar = []
    for bant in BANT_ETIKETLERI:
        mask = df_bolme["fav_oran"].apply(lambda x: bant_ata(x) == bant)
        alt = df_bolme[mask]
        n = len(alt)

        if n == 0:
            sonuclar.append({
                "Bolme": etiket, "Bant": bant,
                "N": 0, "Win": 0, "Win_pct": None,
                "Implied_pct": None, "Sapma_pp": None,
                "Ort_oran": None, "Net_PL": None, "ROI": None,
            })
            continue

        kazandi = (alt["FTR"].values == alt["fav_taraf"].values)
        odds = alt["fav_oran"].values
        pl = np.where(kazandi, odds - 1.0, -1.0)

        win_pct = kazandi.sum() / n * 100
        implied = (1.0 / odds) * 100
        implied_ort = implied.mean()
        sapma = win_pct - implied_ort

        sonuclar.append({
            "Bolme": etiket, "Bant": bant,
            "N": n, "Win": int(kazandi.sum()),
            "Win_pct": round(win_pct, 2),
            "Implied_pct": round(implied_ort, 2),
            "Sapma_pp": round(sapma, 2),
            "Ort_oran": round(float(odds.mean()), 4),
            "Net_PL": round(float(pl.sum()), 4),
            "ROI": round(float(pl.mean()), 6),
        })

    return pd.DataFrame(sonuclar)


# ==================================================================
# 5. BCa BOOTSTRAP
# ==================================================================
def bca_bootstrap_roi(pl, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    if len(pl) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(pl)
    theta_hat = pl.mean()

    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = pl[idx].mean()

    prop_less = np.mean(boot < theta_hat)
    prop_less = min(max(prop_less, 1.0 / (n_boot + 1)), n_boot / (n_boot + 1))
    z0 = norm.ppf(prop_less)

    jack = np.array([np.delete(pl, i).mean() for i in range(n)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0

    z_lo = norm.ppf(0.025)
    z_hi = norm.ppf(0.975)

    def _p(z):
        nn = z0 + z
        dd = 1 - a * nn
        return norm.cdf(z0 + nn / dd) if dd != 0 else np.nan

    p_lo = min(max(_p(z_lo), 0.0), 1.0)
    p_hi = min(max(_p(z_hi), 0.0), 1.0)

    ci_lo = float(np.quantile(boot, p_lo))
    ci_hi = float(np.quantile(boot, p_hi))
    return ci_lo, ci_hi


# ==================================================================
# 6. PERMUTATION (FTR etiketleri)
# ==================================================================
def permutation_test_ftr(ftr_arr, fav_taraf_arr, odds_arr,
                        n_perm=N_PERMUTATION, seed=RANDOM_SEED):
    n = len(ftr_arr)
    if n < 5:
        return np.nan

    rng = np.random.default_rng(seed)

    def _pl(ftr):
        kazandi = (ftr == fav_taraf_arr)
        return np.where(kazandi, odds_arr - 1.0, -1.0)

    obs = _pl(ftr_arr).mean()
    ekstrem = 0
    for _ in range(n_perm):
        perm_ftr = rng.permutation(ftr_arr)
        if abs(_pl(perm_ftr).mean()) >= abs(obs):
            ekstrem += 1

    return (1 + ekstrem) / (n_perm + 1)


# ==================================================================
# 7. MDD (stable)
# ==================================================================
def max_drawdown(pl, tarih):
    if len(pl) == 0:
        return 0.0
    idx = np.argsort(tarih, kind="stable")
    pl_sorted = pl[idx]
    cum = np.cumsum(pl_sorted)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max() / BANKROLL)


# ==================================================================
# 8. HÜCRE DETAYI (HAM DEĞERLER)
# ==================================================================
def hucre_detay(df_bolme, bant):
    mask = df_bolme["fav_oran"].apply(lambda x: bant_ata(x) == bant)
    alt = df_bolme[mask].copy()

    n = len(alt)
    if n < 5:
        return None

    ftr = alt["FTR"].values
    fav_taraf = alt["fav_taraf"].values
    odds = alt["fav_oran"].values
    kazandi = (ftr == fav_taraf)
    pl = np.where(kazandi, odds - 1.0, -1.0)

    tarih = pd.to_datetime(
        alt["Date"], dayfirst=True, format="mixed", errors="coerce"
    ).values

    ci_lo, ci_hi = bca_bootstrap_roi(pl)
    p_perm = permutation_test_ftr(ftr, fav_taraf, odds)
    mdd = max_drawdown(pl, tarih)

    win_pct = float(kazandi.mean() * 100)
    implied_pct = float((1.0 / odds).mean() * 100)
    sapma_pp = win_pct - implied_pct

    return {
        "N": n,
        "Win": int(kazandi.sum()),
        "Win_pct": round(win_pct, 2),
        "Implied_pct": round(implied_pct, 2),
        "Sapma_pp": round(sapma_pp, 2),
        "Ort_oran": round(float(odds.mean()), 4),
        "Net_PL": round(float(pl.sum()), 4),
        "ROI": float(pl.mean()),
        "ROI_GA_lo": float(ci_lo) if not np.isnan(ci_lo) else None,
        "ROI_GA_hi": float(ci_hi) if not np.isnan(ci_hi) else None,
        "Perm_p": float(p_perm) if not np.isnan(p_perm) else None,
        "MDD": float(mdd),
    }


# ==================================================================
# 9. BH-FDR
# ==================================================================
def bh_fdr(p_values, q=0.05):
    n = len(p_values)
    idx_valid = [i for i, p in enumerate(p_values)
                 if p is not None and not np.isnan(p)]
    m = len(idx_valid)

    q_out = [None] * n
    rej_out = [False] * n

    if m == 0:
        return q_out, rej_out

    sorted_idx = sorted(idx_valid, key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in sorted_idx]

    q_vals_sorted = [None] * m
    prev_q = 1.0
    for k in range(m - 1, -1, -1):
        qk = sorted_p[k] * m / (k + 1)
        qk = min(qk, prev_q)
        q_vals_sorted[k] = qk
        prev_q = qk

    for k, orig_i in enumerate(sorted_idx):
        q_out[orig_i] = q_vals_sorted[k]
        rej_out[orig_i] = q_vals_sorted[k] < q

    return q_out, rej_out


# ==================================================================
# 10. KARAR (HAM DEĞERLER)
# ==================================================================
def karar_uygula(detay_oos, detay_val, q_value):
    if detay_oos is None or detay_val is None:
        return "YETERSIZ_VERI", {}

    roi = detay_oos["ROI"]
    ga_lo = detay_oos["ROI_GA_lo"]
    n = detay_oos["N"]
    p = detay_oos["Perm_p"]
    mdd = detay_oos["MDD"]
    roi_v = detay_val["ROI"]

    k_val = roi_v is not None and roi_v > 0
    k_n = n is not None and n >= N_MIN
    k_roi = roi is not None and roi > 0
    k_ga = ga_lo is not None and ga_lo > 0
    k_p = p is not None and p < ALPHA
    k_mdd = mdd is not None and mdd <= MDD_ESIK
    k_q = q_value is not None and q_value < ALPHA

    k_detay = {
        "K_Val_ROI": k_val, "K_N": k_n, "K_ROI": k_roi,
        "K_GA": k_ga, "K_P": k_p, "K_MDD": k_mdd, "K_Q": k_q,
    }

    if k_val and k_n and k_roi and k_ga and k_p and k_mdd and k_q:
        return "SUPPORTED", k_detay
    elif roi is not None and roi > 0:
        return "WEAK / UNCERTAIN", k_detay
    else:
        return "NOT SUPPORTED", k_detay


# ==================================================================
# 11. ANA AKIŞ
# ==================================================================
def main():
    print("=" * 80)
    print("H12 FROZEN v1.0 — Favourite-Longshot Bias")
    print("Favori: en düşük B365 1X2 oranı | Eşitlik: H > D > A")
    print("E0 (10).csv KESİNLİKLE OKUNMAYACAK.")
    print("Oran bantları FROZEN: [1.00-1.20], (1.20-1.40], ..., (2.00-+inf]")
    print("Train: 2015/16 – 2022/23 | Val: 2023/24 | OOS: 2024/25")
    print("=" * 80)

    df = veri_yukle()
    df = favori_belirle(df)

    print(f"\n  Favori taraf dağılımı (tüm veri):")
    print(f"    H: {(df['fav_taraf'] == 'H').sum()}")
    print(f"    D: {(df['fav_taraf'] == 'D').sum()}")
    print(f"    A: {(df['fav_taraf'] == 'A').sum()}")

    train, val, oos = bolmeleri_ayir(df)

    print("\n--- Bant Analizi (Train) ---")
    tab_train = bant_analiz(train, "Train")
    print(tab_train.to_string(index=False))

    print("\n--- Bant Analizi (Validation) ---")
    tab_val = bant_analiz(val, "Validation")
    print(tab_val.to_string(index=False))

    print("\n--- Bant Analizi (OOS) ---")
    tab_oos = bant_analiz(oos, "OOS")
    print(tab_oos.to_string(index=False))

    print("\n--- Detaylı Hücre Analizi (her bant, ham değerler) ---")
    detay_satirlari = []
    p_list = []

    for bant in BANT_ETIKETLERI:
        d_train = hucre_detay(train, bant)
        d_val = hucre_detay(val, bant)
        d_oos = hucre_detay(oos, bant)

        p_oos = d_oos["Perm_p"] if d_oos else None
        p_list.append(p_oos)

        detay_satirlari.append({
            "Bant": bant,
            "Train_N": d_train["N"] if d_train else 0,
            "Train_ROI": d_train["ROI"] if d_train else None,
            "Train_Sapma": d_train["Sapma_pp"] if d_train else None,
            "Val_N": d_val["N"] if d_val else 0,
            "Val_ROI": d_val["ROI"] if d_val else None,
            "OOS_N": d_oos["N"] if d_oos else 0,
            "OOS_Win_pct": d_oos["Win_pct"] if d_oos else None,
            "OOS_Implied_pct": d_oos["Implied_pct"] if d_oos else None,
            "OOS_Sapma_pp": d_oos["Sapma_pp"] if d_oos else None,
            "OOS_Ort_oran": d_oos["Ort_oran"] if d_oos else None,
            "OOS_Net_PL": d_oos["Net_PL"] if d_oos else None,
            "OOS_ROI": d_oos["ROI"] if d_oos else None,
            "OOS_ROI_GA_lo": d_oos["ROI_GA_lo"] if d_oos else None,
            "OOS_ROI_GA_hi": d_oos["ROI_GA_hi"] if d_oos else None,
            "OOS_Perm_p": p_oos,
            "OOS_MDD": d_oos["MDD"] if d_oos else None,
            "_detay_oos": d_oos,
            "_detay_val": d_val,
        })

    q_values, rejected = bh_fdr(p_list, q=ALPHA)
    print("\n--- BH-FDR (tek aile, 6 test, q < 0.05) — HAM p ---")
    for i, bant in enumerate(BANT_ETIKETLERI):
        print(f"  {bant:16s} p={p_list[i]}  q={q_values[i]}  rejected={rejected[i]}")

    print("\n--- KARAR ---")
    karar_satirlari = []
    r6 = lambda x: round(x, 6) if isinstance(x, (int, float)) and x is not None else x
    r2 = lambda x: round(x, 2) if isinstance(x, (int, float)) and x is not None else x

    for i, bant in enumerate(BANT_ETIKETLERI):
        row = detay_satirlari[i]
        d_oos = row["_detay_oos"]
        d_val = row["_detay_val"]
        q = q_values[i]

        karar, k_detay = karar_uygula(d_oos, d_val, q)

        # KIRMIZI BAYRAK tespiti (raporlama; karar mekanizmasına etkisi yok)
        train_roi_raw = row["Train_ROI"]
        oos_roi_raw = row["OOS_ROI"]
        kirmizi_bayrak = (
            train_roi_raw is not None and oos_roi_raw is not None
            and train_roi_raw > 0 and oos_roi_raw < 0
        )

        satir = {
            "Bant": bant,
            "Train_N": row["Train_N"],
            "Train_ROI": r6(row["Train_ROI"]),
            "Train_Sapma": r2(row["Train_Sapma"]),
            "Val_N": row["Val_N"],
            "Val_ROI": r6(row["Val_ROI"]),
            "OOS_N": row["OOS_N"],
            "OOS_Win_pct": r2(row["OOS_Win_pct"]),
            "OOS_Implied_pct": r2(row["OOS_Implied_pct"]),
            "OOS_Sapma_pp": r2(row["OOS_Sapma_pp"]),
            "OOS_Ort_oran": row["OOS_Ort_oran"],
            "OOS_Net_PL": row["OOS_Net_PL"],
            "OOS_ROI": r6(row["OOS_ROI"]),
            "OOS_ROI_GA_lo": r6(row["OOS_ROI_GA_lo"]),
            "OOS_ROI_GA_hi": r6(row["OOS_ROI_GA_hi"]),
            "OOS_Perm_p": r6(row["OOS_Perm_p"]),
            "OOS_q_FDR": r6(q),
            "OOS_MDD": r6(row["OOS_MDD"]),
            "K_Val_ROI": k_detay.get("K_Val_ROI"),
            "K_N": k_detay.get("K_N"),
            "K_ROI": k_detay.get("K_ROI"),
            "K_GA": k_detay.get("K_GA"),
            "K_P": k_detay.get("K_P"),
            "K_MDD": k_detay.get("K_MDD"),
            "K_Q": k_detay.get("K_Q"),
            "KIRMIZI_BAYRAK": kirmizi_bayrak,
            "KARAR": karar,
        }
        karar_satirlari.append(satir)
        kirmizi_str = " ⚠ KIRMIZI BAYRAK" if kirmizi_bayrak else ""
        print(f"  {bant:16s} → {karar}{kirmizi_str}")

    karar_df = pd.DataFrame(karar_satirlari)

    print("\n--- KARAR ÖZETİ ---")
    print(karar_df["KARAR"].value_counts().to_string())

    kb_sayisi = int(karar_df["KIRMIZI_BAYRAK"].sum())
    print(f"\n  KIRMIZI BAYRAK sayısı: {kb_sayisi}")
    if kb_sayisi > 0:
        print("  Kırmızı bayraklı bantlar:")
        for _, r in karar_df[karar_df["KIRMIZI_BAYRAK"]].iterrows():
            print(f"    {r['Bant']}: Train ROI={r['Train_ROI']} → OOS ROI={r['OOS_ROI']}")

    print("\n--- TAM KARAR TABLOSU ---")
    cols_goster = ["Bant", "Train_N", "Train_ROI", "Train_Sapma",
                   "Val_N", "Val_ROI",
                   "OOS_N", "OOS_Win_pct", "OOS_Implied_pct", "OOS_Sapma_pp",
                   "OOS_ROI", "OOS_ROI_GA_lo", "OOS_Perm_p",
                   "OOS_q_FDR", "OOS_MDD", "KIRMIZI_BAYRAK", "KARAR"]
    print(karar_df[cols_goster].to_string(index=False))

    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        tab_train.to_excel(w, sheet_name="Band_Train", index=False)
        tab_val.to_excel(w, sheet_name="Band_Validation", index=False)
        tab_oos.to_excel(w, sheet_name="Band_OOS", index=False)
        karar_df.drop(columns=["_detay_oos", "_detay_val"],
                      errors="ignore").to_excel(w, sheet_name="Karar", index=False)

        pd.DataFrame([{
            "Protokol": "H12 FROZEN v1.0 — Favourite-Longshot Bias",
            "Tarih": "2026-10-08",
            "Addendum": "Favori esitligi: H > D > A (deterministik)",
            "Hipotez": "Favoriler implied prob'dan daha sik kazanir",
            "Favori_tanim": "En dusuk B365 1X2 orani",
            "Favori_esitlik": "H > D > A",
            "Test_sayisi": 6,
            "Oran_bantlari": "[1.00-1.20], (1.20-1.40], (1.40-1.60], (1.60-1.80], (1.80-2.00], (2.00-+inf]",
            "Bahis_orani": "ACILIS B365 (favori taraf)",
            "Stake": 1,
            "Settlement": "Dogru: +(oran-1); Yanlis: -1",
            "Bootstrap": "BCa 10.000, seed=20261008",
            "Permutation": "FTR etiketleri, 10.000, iki tarafli, seed=20261008",
            "MDD_bankroll": 100,
            "MDD_sort": "stable kronolojik",
            "N_MIN": N_MIN,
            "ALPHA": ALPHA,
            "MDD_ESIK": MDD_ESIK,
            "FDR": "BH, tek aile, 6 test, q<0.05",
            "Karar_ham": "Karar ham degerlerle; yuvarlama yalniz gosterimde",
            "Kirmizi_bayrak_kurali": "Train ROI>0 ∧ OOS ROI<0",
            "Train": "2015/16 - 2022/23 (3039)",
            "Validation": "2023/24 (380)",
            "OOS": "2024/25 (380)",
            "2026_27": "KILITLI - okunmadi",
            "KARAR_OZET": str(karar_df["KARAR"].value_counts().to_dict()),
            "KIRMIZI_BAYRAK_SAYISI": kb_sayisi,
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} başarıyla yazıldı.")

    print("\n" + "=" * 80)
    print("VERİ BÜTÜNLÜĞÜ VE KANIT")
    print("=" * 80)
    print(f"  Toplam: {len(df)}")
    print(f"  Train N: {len(train)}")
    print(f"  Val N:   {len(val)}")
    print(f"  OOS N:   {len(oos)}")
    print(f"  Okuma sayacı: {dict(_okuma_sayaci)}")
    print(f"  Yasaklı dosyalar: {sorted(YASAKLI_DOSYALAR)}")
    print(f"  Yasaklılardan hiçbiri okunmadı: "
          f"{all(d not in _okuma_sayaci for d in YASAKLI_DOSYALAR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()