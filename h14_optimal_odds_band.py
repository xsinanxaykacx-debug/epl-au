# -*- coding: utf-8 -*-
"""
h14_optimal_odds_band.py
H14 FROZEN v1.0 — Optimal Odds Band Discovery + OOS Confirmation
Tarih: 2026-10-08

Hipotez:
EPL 1X2 piyasasında, belirli bir oran bandında favori tarafa bahis
oynandığında sistematik pozitif ROI elde edilir. Bu band, veri
üzerinde KEŞİF (Train+Val) ile bulunur, OOS'ta DOĞRULANIR.

Yöntem:
1. Train+Val'de 0.05 aralıklı oran bantları tara (1.00-3.00 arası)
2. Her bant için: N, Win%, ROI hesapla
3. Discovery filtresi: N≥100 VE Val_ROI>0 VE Train_ROI>0
4. En yüksek Val_ROI'li bandı seç (OOS'a BAKMADAN)
5. OOS'ta yalnızca o bandı test et
6. BH-FDR ile çoklu test düzeltmesi

KİLİTLER:
- E0 (10).csv ASLA OKUNMAZ.
- Oran bandı aralığı: 0.05 (FROZEN)
- Discovery filtresi: N≥100, Train_ROI>0, Val_ROI>0 (FROZEN)
- Seçim kriteri: en yüksek Val_ROI (FROZEN)
- OOS doğrulama: ROI>0, GA_lo>0, p<0.05, MDD≤%20 (FROZEN)
- BH-FDR: tüm keşfedilen bantlar üzerinde, q<0.05
- Bootstrap: BCa 10.000, seed=20261008
- Permutation: sign-permutation, 10.000, iki taraflı, seed=20261008
- MDD: bankroll 100, stable
- N_MIN=100, ALPHA=0.05, MDD_ESIK=0.20

ÇIKTI: h14_optimal_odds_band_results.xlsx

UYARI:
Bu bir keşif+doğrulama çalışmasıdır. OOS sonucu, keşifte bulunan
bandın gerçek olup olmadığını gösterir.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ==================================================================
# 0. SABİTLER
# ==================================================================
CIKTI = "h14_optimal_odds_band_results.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

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

# Oran bandı taraması: 1.00 - 3.00, adım 0.05
BANT_ADIM = 0.05
BANT_MIN = 1.00
BANT_MAX = 3.00

N_BOOTSTRAP = 10_000
N_PERMUTATION = 10_000
RANDOM_SEED = 20261008
BANKROLL = 100.0
MDD_ESIK = 0.20
ALPHA = 0.05
N_MIN = 100

BEKLENEN_TOPLAM = 3800

_okuma_sayaci = defaultdict(int)


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

    gecerli = (
        df["B365H"].notna() & (df["B365H"] > 0) &
        df["B365D"].notna() & (df["B365D"] > 0) &
        df["B365A"].notna() & (df["B365A"] > 0) &
        df["FTR"].notna() & df["Date"].notna()
    )
    n_dusen = int((~gecerli).sum())
    print(f"  Düşen satır: {n_dusen}")
    df = df[gecerli].copy().reset_index(drop=True)

    # Favori taraf
    oranlar = df[["B365H", "B365D", "B365A"]].values
    fav_idx = np.argmin(oranlar, axis=1)
    df["fav_oran"] = oranlar[np.arange(len(df)), fav_idx]
    df["fav_taraf"] = np.where(fav_idx == 0, "H",
                       np.where(fav_idx == 1, "D", "A"))

    df["Date_p"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    return df


def bolmeleri_ayir(df):
    train = df[df["Season"].isin(TRAIN_SEZONLAR)].copy().reset_index(drop=True)
    val = df[df["Season"] == VALIDATION_SEZON].copy().reset_index(drop=True)
    oos = df[df["Season"] == OOS_SEZON].copy().reset_index(drop=True)
    print(f"  Train: {len(train)}, Val: {len(val)}, OOS: {len(oos)}")
    return train, val, oos


def bant_tara(df_bolme, bant_min, bant_max):
    """Bir oran bandı için N, Win, ROI hesapla."""
    mask = (df_bolme["fav_oran"] >= bant_min) & (df_bolme["fav_oran"] < bant_max)
    alt = df_bolme[mask]
    n = len(alt)
    if n == 0:
        return None

    kazandi = (alt["FTR"].values == alt["fav_taraf"].values)
    odds = alt["fav_oran"].values
    pl = np.where(kazandi, odds - 1.0, -1.0)

    return {
        "bant_min": bant_min,
        "bant_max": bant_max,
        "N": n,
        "Win": int(kazandi.sum()),
        "Win_pct": round(kazandi.sum() / n * 100, 2),
        "ROI": float(pl.mean()),
        "Net_PL": float(pl.sum()),
    }


def bantlari_tara(df_bolme, etiket):
    bantlar = np.arange(BANT_MIN, BANT_MAX, BANT_ADIM)
    sonuclar = []
    for b in bantlar:
        r = bant_tara(df_bolme, round(b, 2), round(b + BANT_ADIM, 2))
        if r is not None:
            r["Bolme"] = etiket
            sonuclar.append(r)
    return pd.DataFrame(sonuclar)


# ==================================================================
# İSTATİSTİK
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
    return float(np.quantile(boot, p_lo)), float(np.quantile(boot, p_hi))


def permutation_sign(pl, n_perm=N_PERMUTATION, seed=RANDOM_SEED):
    n = len(pl)
    if n < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = pl.mean()
    abs_pl = np.abs(pl)
    ekstrem = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs((abs_pl * signs).mean()) >= abs(obs):
            ekstrem += 1
    return (1 + ekstrem) / (n_perm + 1)


def max_drawdown(pl, tarih):
    if len(pl) == 0:
        return 0.0
    idx = np.argsort(tarih, kind="stable")
    cum = np.cumsum(pl[idx])
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max() / BANKROLL)


# ==================================================================
# BH-FDR
# ==================================================================
def bh_fdr(p_values, q=0.05):
    n = len(p_values)
    idx_valid = [i for i, p in enumerate(p_values) if p is not None and not np.isnan(p)]
    m = len(idx_valid)
    q_out = [None] * n
    rej_out = [False] * n
    if m == 0:
        return q_out, rej_out
    sorted_idx = sorted(idx_valid, key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in sorted_idx]
    q_vals = [None] * m
    prev = 1.0
    for k in range(m - 1, -1, -1):
        qk = sorted_p[k] * m / (k + 1)
        qk = min(qk, prev)
        q_vals[k] = qk
        prev = qk
    for k, orig_i in enumerate(sorted_idx):
        q_out[orig_i] = q_vals[k]
        rej_out[orig_i] = q_vals[k] < q
    return q_out, rej_out


# ==================================================================
# ANA AKIŞ
# ==================================================================
def main():
    print("=" * 80)
    print("H14 FROZEN v1.0 — Optimal Odds Band Discovery + OOS Confirmation")
    print(f"Bant taraması: {BANT_MIN} - {BANT_MAX}, adım {BANT_ADIM}")
    print("Discovery filtresi: N>=100, Train_ROI>0, Val_ROI>0")
    print("Seçim: en yüksek Val_ROI (OOS'a BAKMADAN)")
    print("E0 (10).csv KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    df = veri_yukle()
    train, val, oos = bolmeleri_ayir(df)

    # --- 1. KEŞİF (Train + Val) ---
    print("\n--- KEŞİF (Train taraması) ---")
    tab_train = bantlari_tara(train, "Train")
    print(f"  {len(tab_train)} bant tarandı")

    print("\n--- KEŞİF (Validation taraması) ---")
    tab_val = bantlari_tara(val, "Validation")
    print(f"  {len(tab_val)} bant tarandı")

    # Birleştir (aynı bant_min için)
    merged = tab_train[["bant_min", "bant_max", "N", "ROI"]].merge(
        tab_val[["bant_min", "N", "ROI"]],
        on="bant_min", suffixes=("_train", "_val")
    )

    # Discovery filtresi
    discovery = merged[
        (merged["N_train"] >= N_MIN) &
        (merged["ROI_train"] > 0) &
        (merged["ROI_val"] > 0)
    ].copy()

    print(f"\n--- DISCOVERY SONUCU ---")
    print(f"  Filtreyi geçen bant: {len(discovery)}")
    if len(discovery) > 0:
        print(discovery.to_string(index=False))

    if len(discovery) == 0:
        print("\n  ⚠ Discovery filtresini geçen bant YOK.")
        print("  OOS doğrulaması yapılamaz.")
        print("  KARAR: NOT SUPPORTED")
        with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
            tab_train.to_excel(w, sheet_name="Train", index=False)
            tab_val.to_excel(w, sheet_name="Validation", index=False)
            pd.DataFrame([{
                "Protokol": "H14 FROZEN v1.0",
                "Karar": "NOT SUPPORTED",
                "Sebep": "Discovery filtresini gecen bant yok",
                "Discovery_filtre": f"N>={N_MIN}, Train_ROI>0, Val_ROI>0",
            }]).to_excel(w, sheet_name="Karar", index=False)
        print(f"  {CIKTI} yazıldı.")
        return

    # En yüksek Val ROI'li bandı seç
    secili = discovery.sort_values("ROI_val", ascending=False).iloc[0]
    print(f"\n--- SEÇİLEN BANT (OOS'a bakmadan) ---")
    print(f"  Bant: [{secili['bant_min']:.2f}, {secili['bant_max']:.2f})")
    print(f"  Train: N={secili['N_train']}, ROI={secili['ROI_train']:.6f}")
    print(f"  Val:   N={secili['N_val']}, ROI={secili['ROI_val']:.6f}")

    # --- 2. OOS DOĞRULAMA ---
    print(f"\n--- OOS DOĞRULAMA ---")
    oos_band = bant_tara(oos, float(secili["bant_min"]), float(secili["bant_max"]))
    if oos_band is None or oos_band["N"] < 5:
        print(f"  OOS'ta bu bantta yeterli maç yok.")
        karar = "YETERSIZ_VERI"
    else:
        # Detaylı istatistik
        mask = (oos["fav_oran"] >= float(secili["bant_min"])) & (oos["fav_oran"] < float(secili["bant_max"]))
        alt = oos[mask]
        kazandi = (alt["FTR"].values == alt["fav_taraf"].values)
        odds = alt["fav_oran"].values
        pl = np.where(kazandi, odds - 1.0, -1.0)
        tarih = alt["Date_p"].values

        ci_lo, ci_hi = bca_bootstrap_roi(pl)
        p_perm = permutation_sign(pl)
        mdd = max_drawdown(pl, tarih)

        print(f"  OOS: N={oos_band['N']}, Win%={oos_band['Win_pct']}")
        print(f"       ROI={oos_band['ROI']:.6f}")
        print(f"       BCa GA=[{ci_lo:.6f}, {ci_hi:.6f}]")
        print(f"       Perm_p={p_perm:.6f}")
        print(f"       MDD={mdd:.6f}")

        # Karar
        k_roi = oos_band["ROI"] > 0
        k_ga = ci_lo > 0
        k_n = oos_band["N"] >= N_MIN
        k_p = p_perm < ALPHA
        k_mdd = mdd <= MDD_ESIK

        if k_roi and k_ga and k_n and k_p and k_mdd:
            karar = "SUPPORTED"
        elif oos_band["ROI"] > 0:
            karar = "WEAK / UNCERTAIN"
        else:
            karar = "NOT SUPPORTED"

        print(f"\n  Kriterler: ROI>0={k_roi}, GA_lo>0={k_ga}, N>={N_MIN}={k_n}, p<{ALPHA}={k_p}, MDD<={MDD_ESIK}={k_mdd}")
        print(f"  KARAR: {karar}")

        # BH-FDR: keşifte bulunan tüm bantlar üzerinden q değeri
        # Sadece seçili bantın OOS p'si tek; ama discovery'de kaç bant vardı?
        # BH-FDR için OOS'taki tüm bantların p'sini hesaplayalım (tüm testler)
        print(f"\n  NOT: BH-FDR için OOS'ta tüm {BANT_MIN}-{BANT_MAX} bantları test ediliyor...")
        tum_bantlar = np.arange(BANT_MIN, BANT_MAX, BANT_ADIM)
        p_list = []
        for b in tum_bantlar:
            bb = bant_tara(oos, round(b, 2), round(b + BANT_ADIM, 2))
            if bb is None or bb["N"] < 5:
                p_list.append(None)
                continue
            m2 = (oos["fav_oran"] >= round(b, 2)) & (oos["fav_oran"] < round(b + BANT_ADIM, 2))
            a2 = oos[m2]
            k2 = (a2["FTR"].values == a2["fav_taraf"].values)
            o2 = a2["fav_oran"].values
            pl2 = np.where(k2, o2 - 1.0, -1.0)
            p_list.append(permutation_sign(pl2))

        q_values, rejected = bh_fdr(p_list, q=ALPHA)
        secili_idx = None
        for i, b in enumerate(tum_bantlar):
            if abs(b - float(secili["bant_min"])) < 1e-6:
                secili_idx = i
                break

        if secili_idx is not None and q_values[secili_idx] is not None:
            q_secili = q_values[secili_idx]
            k_q = q_secili < ALPHA
            print(f"  Seçili bant BH-FDR q = {q_secili:.6f} → {'PASS' if k_q else 'FAIL'}")
            if karar == "SUPPORTED" and not k_q:
                karar = "WEAK / UNCERTAIN"
                print(f"  → BH-FDR FAIL, karar WEAK'e düşürüldü")
        else:
            q_secili = None
            print(f"  Seçili bant BH-FDR q hesaplanamadı")

    # --- 3. Excel ---
    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        tab_train.to_excel(w, sheet_name="Train_Tarama", index=False)
        tab_val.to_excel(w, sheet_name="Val_Tarama", index=False)
        discovery.to_excel(w, sheet_name="Discovery", index=False)

        pd.DataFrame([{
            "Protokol": "H14 FROZEN v1.0",
            "Tarih": "2026-10-08",
            "Yontem": "Discovery (Train+Val) + OOS Confirmation",
            "Bant_araligi": f"{BANT_MIN}-{BANT_MAX}, adim {BANT_ADIM}",
            "Discovery_filtre": f"N_train>={N_MIN}, Train_ROI>0, Val_ROI>0",
            "Secim": "en yuksek Val_ROI (OOS'a bakmadan)",
            "Secilen_bant": f"[{secili['bant_min']:.2f}, {secili['bant_max']:.2f})",
            "Train_N_secili": int(secili["N_train"]),
            "Train_ROI_secili": float(secili["ROI_train"]),
            "Val_N_secili": int(secili["N_val"]),
            "Val_ROI_secili": float(secili["ROI_val"]),
            "OOS_N": oos_band["N"] if oos_band else 0,
            "OOS_ROI": oos_band["ROI"] if oos_band else None,
            "KARAR": karar,
            "2026_27": "KILITLI - okunmadi",
        }]).to_excel(w, sheet_name="Karar", index=False)

    print(f"\n{CIKTI} yazıldı.")

    print("\n" + "=" * 80)
    print("VERİ BÜTÜNLÜĞÜ VE KANIT")
    print("=" * 80)
    print(f"  Toplam: {len(df)}")
    print(f"  Okuma sayacı: {dict(_okuma_sayaci)}")
    print(f"  Yasaklılardan hiçbiri okunmadı: "
          f"{all(d not in _okuma_sayaci for d in YASAKLI_DOSYALAR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()