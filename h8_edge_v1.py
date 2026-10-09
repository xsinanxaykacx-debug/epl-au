# -*- coding: utf-8 -*-
"""
h8_edge_v1.py
H8-EDGE v1.0 FROZEN PROTOCOL — birebir implementasyon
Tarih: 2026-10-08

Kapsam:
- Sinyal: Elo_Diff > 100 AND B365H < 2.50 → HOME bahsi
- Fiyat: B365 açılış (B365H)
- Stake: 1 birim sabit
- Settlement: Home win → +(B365H - 1); Draw/Loss → -1
- Train:       2019/20 + 2020/21 + 2021/22 + 2022/23
- Validation:  2023/24
- Blind OOS:   2024/25 (tek seferlik)

DÜZELTME (bu sürüm):
- Permutation testi artık FTR sonuç etiketlerini permüte ediyor.
  B365H oranları sabit kalıyor, P/L her permütasyonda yeniden hesaplanıyor.
  (Önceki sign-flip yöntemi protokoldeki "sonuç etiketleri" ifadesiyle
  uyuşmuyordu.)

KİLİTLER:
- E0 (10).csv ASLA OKUNMAZ.
- Yalnızca E0 (4) – E0 (9) okunur.
- AH ailesi filtresi: 2280 → 2278.
- Sinyal, bahis yönü, eşikler, istatistiksel yöntemler DEĞİŞTİRİLEMEZ.
- Bootstrap: BCa 10.000, seed=20261008
- Permutation: 10.000, FTR sonuç etiketleri, iki taraflı, seed=20261008
- MDD: Bankroll 100, kronolojik sıralama

KARAR:
- SUPPORTED: ROI>0 + GA alt sınırı>0 + N>=100 + p<0.05 + MDD<=%20
- WEAK / UNCERTAIN: ROI>0 ama bir kriter eksik
- NOT SUPPORTED: ROI<=0 veya MDD>%20
- RED FLAG: Train pozitif, OOS negatif

ÇIKTI: h8_edge_v1_results.xlsx
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
H1_FEATURES = "h1_features.csv"
ENVANTER = "envanter_v2.xlsx"
ENVANTER_SHEET = "Eksik_Maclar"
CIKTI = "h8_edge_v1_results.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

MARKET_DOSYALARI = {
    "2019/20": "E0 (4).csv",
    "2020/21": "E0 (5).csv",
    "2021/22": "E0 (6).csv",
    "2022/23": "E0 (7).csv",
    "2023/24": "E0 (8).csv",
    "2024/25": "E0 (9).csv",
}
MARKET_SUTUNLAR = ["Season", "Date", "HomeTeam", "AwayTeam", "B365H"]

TRAIN_SEZONLAR = ["2019/20", "2020/21", "2021/22", "2022/23"]
VALIDATION_SEZON = "2023/24"
OOS_SEZON = "2024/25"

ELO_ESIK = 100
FIYAT_ESIK = 2.50

N_BOOTSTRAP = 10_000
N_PERMUTATION = 10_000
RANDOM_SEED = 20261008
BANKROLL = 100.0
MDD_ESIK = 0.20
ALPHA = 0.05
N_MIN = 100

BEKLENEN_TOP = 2278
BEKLENEN_TRAIN = 1519
BEKLENEN_VAL = 379
BEKLENEN_OOS = 380

BEKLENEN_SINYAL_TOP = 616
BEKLENEN_SINYAL_TRAIN = 406
BEKLENEN_SINYAL_VAL = 106
BEKLENEN_SINYAL_OOS = 104

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


# ==================================================================
# 2. VERİ YÜKLEME
# ==================================================================
def veri_yukle():
    print("\n--- Veri yükleniyor ---")

    if not os.path.exists(H1_FEATURES):
        raise RuntimeError(f"{H1_FEATURES} bulunamadı.")
    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")

    for c in ["FTR", "Home_Elo_Pre", "Away_Elo_Pre"]:
        if c not in feat.columns:
            raise RuntimeError(f"{H1_FEATURES} içinde {c} yok.")

    frames = []
    for sezon, dosya in MARKET_DOSYALARI.items():
        ad = os.path.basename(dosya)
        if ad in YASAKLI_DOSYALAR:
            raise RuntimeError(f"YASAKLI DOSYA: {ad}")
        df = _guvenli_oku(dosya)
        df["Season"] = sezon
        eksik = [s for s in MARKET_SUTUNLAR if s not in df.columns]
        if eksik:
            raise RuntimeError(f"{ad} içinde eksik sütunlar: {eksik}")
        frames.append(df[MARKET_SUTUNLAR].copy())
    market = pd.concat(frames, ignore_index=True)

    feat["_key"] = _mac_anahtari(feat)
    market["_key"] = _mac_anahtari(market)

    df = feat.merge(market[["_key", "B365H"]], on="_key", how="inner")
    print(f"  Merge sonrası: {len(df)}")

    dup = df["_key"].duplicated().sum()
    if dup > 0:
        raise RuntimeError(f"Duplicate bulundu: {dup}")

    eksik = pd.read_excel(ENVANTER, sheet_name=ENVANTER_SHEET)
    eksik["_key"] = _mac_anahtari(eksik)
    eksik_ah = set(eksik.loc[eksik["AH_Eksik"] == True, "_key"].dropna().astype(str))

    df["_eksik_ah"] = df["_key"].isin(eksik_ah)
    onceki = len(df)
    df = df[~df["_eksik_ah"]].copy().reset_index(drop=True)
    sonraki = len(df)
    print(f"  AH ailesi eksik filtresi: {onceki} → {sonraki} (dışlanan: {onceki - sonraki})")

    df["Elo_Diff"] = df["Home_Elo_Pre"] - df["Away_Elo_Pre"]
    df["Signal"] = (df["Elo_Diff"] > ELO_ESIK) & (df["B365H"] < FIYAT_ESIK)

    kritik = ["FTR", "Elo_Diff", "B365H"]
    df = df.dropna(subset=kritik).reset_index(drop=True)
    print(f"  NaN temizliği sonrası: {len(df)}")

    return df


# ==================================================================
# 3. SETTLEMENT
# ==================================================================
def settle_bets(df_signal):
    """Sinyal sağlayan maçlar için HOME bahsi (B365 açılış)."""
    if len(df_signal) == 0:
        return np.array([]), np.array([])

    ftr = df_signal["FTR"].values
    oran = df_signal["B365H"].values

    pl = np.where(ftr == "H", oran - 1.0, -1.0)

    tarih = pd.to_datetime(df_signal["Date"], dayfirst=True, format="mixed", errors="coerce")

    return pl, tarih.values


# ==================================================================
# 4. İSTATİSTİK
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
    prop_less = min(max(prop_less, 1.0/(n_boot+1)), n_boot/(n_boot+1))
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


def permutation_test_roi(df_signal, n_perm=N_PERMUTATION, seed=RANDOM_SEED):
    """
    DOĞRU PROTOKOL: FTR sonuç etiketleri permüte edilir.
    B365H oranları sabit kalır. Her permütasyonda P/L yeniden hesaplanır.
    İki taraflı.
    """
    if len(df_signal) < 5:
        return np.nan

    rng = np.random.default_rng(seed)

    ftr = df_signal["FTR"].to_numpy(copy=True)
    oran = df_signal["B365H"].to_numpy(copy=True)

    # Gözlenen
    obs_pl = np.where(ftr == "H", oran - 1.0, -1.0)
    obs = obs_pl.mean()

    n = len(ftr)
    ekstrem = 0

    for _ in range(n_perm):
        perm_ftr = rng.permutation(ftr)
        perm_pl = np.where(perm_ftr == "H", oran - 1.0, -1.0)
        if abs(perm_pl.mean()) >= abs(obs):
            ekstrem += 1

    return (1 + ekstrem) / (n_perm + 1)


def max_drawdown(pl):
    if len(pl) == 0:
        return 0.0
    cum = np.cumsum(pl)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max() / BANKROLL)


# ==================================================================
# 5. BÖLME ANALİZİ
# ==================================================================
def bolme_analiz(df_bolme, etiket):
    alt = df_bolme[df_bolme["Signal"]].copy().reset_index(drop=True)

    n_sinyal = len(alt)
    if n_sinyal == 0:
        return {
            "Bolme": etiket, "N": 0, "Win": 0, "Win_pct": None,
            "Ort_oran": None, "Net_PL": None, "ROI": None,
            "ROI_GA_lo": None, "ROI_GA_hi": None,
            "Perm_p": None, "MDD": None,
        }

    pl, tarih = settle_bets(alt)
    idx = np.argsort(tarih)
    pl = pl[idx]
    alt_sorted = alt.iloc[idx].reset_index(drop=True)

    win = int((pl > 0).sum())
    ort_oran = float(alt["B365H"].mean())
    net = float(pl.sum())
    roi = net / n_sinyal

    ci_lo, ci_hi = bca_bootstrap_roi(pl)
    p_perm = permutation_test_roi(alt_sorted)  # DÜZELTME: alt DataFrame
    mdd = max_drawdown(pl)

    return {
        "Bolme": etiket, "N": n_sinyal,
        "Win": win, "Win_pct": round(win/n_sinyal*100, 2),
        "Ort_oran": round(ort_oran, 4),
        "Net_PL": round(net, 4), "ROI": round(roi, 4),
        "ROI_GA_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else None,
        "ROI_GA_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else None,
        "Perm_p": round(p_perm, 4) if not np.isnan(p_perm) else None,
        "MDD": round(mdd, 4),
    }


def karar_uygula(oos_metrik, train_metrik, val_metrik):
    roi = oos_metrik["ROI"]
    ga_lo = oos_metrik["ROI_GA_lo"]
    n = oos_metrik["N"]
    p = oos_metrik["Perm_p"]
    mdd = oos_metrik["MDD"]

    k_roi = roi is not None and roi > 0
    k_ga = ga_lo is not None and ga_lo > 0
    k_n = n >= N_MIN
    k_p = p is not None and p < ALPHA
    k_mdd = mdd is not None and mdd <= MDD_ESIK

    kirmizi = False
    if train_metrik["ROI"] is not None and oos_metrik["ROI"] is not None:
        if train_metrik["ROI"] > 0 and oos_metrik["ROI"] < 0:
            kirmizi = True

    if k_roi and k_ga and k_n and k_p and k_mdd:
        karar = "SUPPORTED"
    elif roi is not None and roi <= 0:
        karar = "NOT SUPPORTED"
    elif mdd is not None and mdd > MDD_ESIK:
        karar = "NOT SUPPORTED"
    elif k_roi:
        karar = "WEAK / UNCERTAIN"
    else:
        karar = "NOT SUPPORTED"

    return {
        "K_ROI": k_roi, "K_GA": k_ga, "K_N": k_n, "K_P": k_p, "K_MDD": k_mdd,
        "Karar": karar, "Kirmizi_bayrak": kirmizi,
    }


# ==================================================================
# 6. ANA AKIŞ
# ==================================================================
def main():
    print("=" * 80)
    print("H8-EDGE v1.0 — FROZEN")
    print("Sinyal: Elo_Diff > 100 AND B365H < 2.50 → HOME")
    print("E0 (10).csv KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    df = veri_yukle()

    print("\n--- SANITY CHECK (veri) ---")
    if len(df) != BEKLENEN_TOP:
        raise RuntimeError(f"Toplam satır: {len(df)} != {BEKLENEN_TOP}")
    print(f"  Toplam satır = {BEKLENEN_TOP} ✓")

    train = df[df["Season"].isin(TRAIN_SEZONLAR)].copy().reset_index(drop=True)
    val = df[df["Season"] == VALIDATION_SEZON].copy().reset_index(drop=True)
    oos = df[df["Season"] == OOS_SEZON].copy().reset_index(drop=True)

    if len(train) != BEKLENEN_TRAIN:
        raise RuntimeError(f"Train N: {len(train)} != {BEKLENEN_TRAIN}")
    if len(val) != BEKLENEN_VAL:
        raise RuntimeError(f"Validation N: {len(val)} != {BEKLENEN_VAL}")
    if len(oos) != BEKLENEN_OOS:
        raise RuntimeError(f"OOS N: {len(oos)} != {BEKLENEN_OOS}")
    print(f"  Train N = {BEKLENEN_TRAIN} ✓")
    print(f"  Validation N = {BEKLENEN_VAL} ✓")
    print(f"  OOS N = {BEKLENEN_OOS} ✓")

    print("\n--- SANITY CHECK (sinyal) ---")
    n_sinyal_train = int(train["Signal"].sum())
    n_sinyal_val = int(val["Signal"].sum())
    n_sinyal_oos = int(oos["Signal"].sum())
    n_sinyal_top = int(df["Signal"].sum())

    print(f"  Train sinyal: {n_sinyal_train} (beklenen {BEKLENEN_SINYAL_TRAIN})")
    print(f"  Val sinyal:   {n_sinyal_val} (beklenen {BEKLENEN_SINYAL_VAL})")
    print(f"  OOS sinyal:   {n_sinyal_oos} (beklenen {BEKLENEN_SINYAL_OOS})")
    print(f"  Toplam sinyal: {n_sinyal_top} (beklenen {BEKLENEN_SINYAL_TOP})")

    if n_sinyal_train != BEKLENEN_SINYAL_TRAIN:
        raise RuntimeError(f"Train sinyal sayısı beklenenden farklı")
    if n_sinyal_val != BEKLENEN_SINYAL_VAL:
        raise RuntimeError(f"Val sinyal sayısı beklenenden farklı")
    if n_sinyal_oos != BEKLENEN_SINYAL_OOS:
        raise RuntimeError(f"OOS sinyal sayısı beklenenden farklı")
    print(f"  → TÜM SANITY CHECK'LER PASS")

    print("\n--- BÖLME ANALİZİ ---")
    train_metrik = bolme_analiz(train, "Train")
    val_metrik = bolme_analiz(val, "Validation")
    oos_metrik = bolme_analiz(oos, "OOS")

    ozet_df = pd.DataFrame([train_metrik, val_metrik, oos_metrik])
    print(ozet_df.to_string(index=False))

    print("\n--- KARAR ---")
    karar_dict = karar_uygula(oos_metrik, train_metrik, val_metrik)

    print(f"  OOS ROI: {oos_metrik['ROI']}")
    print(f"  OOS ROI GA: [{oos_metrik['ROI_GA_lo']}, {oos_metrik['ROI_GA_hi']}]")
    print(f"  OOS N: {oos_metrik['N']}")
    print(f"  OOS Perm_p: {oos_metrik['Perm_p']}")
    print(f"  OOS MDD: {oos_metrik['MDD']}")
    print(f"")
    print(f"  K_ROI: {karar_dict['K_ROI']}")
    print(f"  K_GA:  {karar_dict['K_GA']}")
    print(f"  K_N:   {karar_dict['K_N']}")
    print(f"  K_P:   {karar_dict['K_P']}")
    print(f"  K_MDD: {karar_dict['K_MDD']}")
    print(f"")
    print(f"  KARAR: {karar_dict['Karar']}")
    if karar_dict["Kirmizi_bayrak"]:
        print(f"  ⚠ KIRMIZI BAYRAK: Train pozitif, OOS negatif")

    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        ozet_df.to_excel(w, sheet_name="Ozet", index=False)

        for etiket, df_b in [("Train", train), ("Validation", val), ("OOS", oos)]:
            alt = df_b[df_b["Signal"]].copy().reset_index(drop=True)
            if len(alt) == 0:
                continue
            pl, tarih = settle_bets(alt)
            idx = np.argsort(tarih)
            alt = alt.iloc[idx].reset_index(drop=True)
            pl = pl[idx]
            alt["P/L"] = pl
            alt["Kumulatif_PL"] = np.cumsum(pl)
            cols = ["Season", "Date", "HomeTeam", "AwayTeam",
                    "Elo_Diff", "B365H", "FTR", "P/L", "Kumulatif_PL"]
            alt[cols].to_excel(w, sheet_name=f"Bahis_{etiket}", index=False)

        pd.DataFrame([{
            "Sinyal": f"Elo_Diff > {ELO_ESIK} AND B365H < {FIYAT_ESIK}",
            "Bahis_yonu": "HOME",
            "Fiyat": "B365 acilis (B365H)",
            "Stake": 1,
            "OOS_N": oos_metrik["N"],
            "OOS_Win_pct": oos_metrik["Win_pct"],
            "OOS_Ort_oran": oos_metrik["Ort_oran"],
            "OOS_Net_PL": oos_metrik["Net_PL"],
            "OOS_ROI": oos_metrik["ROI"],
            "OOS_ROI_GA_lo": oos_metrik["ROI_GA_lo"],
            "OOS_ROI_GA_hi": oos_metrik["ROI_GA_hi"],
            "OOS_Perm_p": oos_metrik["Perm_p"],
            "OOS_MDD": oos_metrik["MDD"],
            "K_ROI": karar_dict["K_ROI"],
            "K_GA": karar_dict["K_GA"],
            "K_N": karar_dict["K_N"],
            "K_P": karar_dict["K_P"],
            "K_MDD": karar_dict["K_MDD"],
            "KARAR": karar_dict["Karar"],
            "Kirmizi_bayrak": karar_dict["Kirmizi_bayrak"],
        }]).to_excel(w, sheet_name="Karar", index=False)

        pd.DataFrame([{
            "Protokol": "H8-EDGE FROZEN PROTOCOL v1.0",
            "Tarih": "2026-10-08",
            "Sinyal": f"Elo_Diff > {ELO_ESIK} AND B365H < {FIYAT_ESIK}",
            "Bahis_yonu": "HOME",
            "Fiyat_kaynagi": "B365 acilis (B365H)",
            "Stake": "1 birim sabit",
            "Settlement": "Home win: +(B365H-1); Draw/Loss: -1",
            "Bootstrap": "BCa 10.000, seed=20261008",
            "Permutation": "10.000, FTR sonuc etiketleri, iki tarafli, seed=20261008",
            "MDD_bankroll": 100,
            "N_min": N_MIN,
            "Karar_agaci": "SUPPORTED/WEAK/NOT SUPPORTED",
            "Kirmizi_bayrak": "Train pozitif, OOS negatif",
            "Train": "2019/20 - 2022/23",
            "Validation": "2023/24",
            "OOS": "2024/25",
            "2026_27": "KILITLI - okunmadi",
            "KARAR": karar_dict["Karar"],
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} başarıyla yazıldı.")

    print("\n" + "=" * 80)
    print("VERİ BÜTÜNLÜĞÜ VE KANIT")
    print("=" * 80)
    print(f"  Toplam satır: {len(df)}")
    print(f"  Train N: {len(train)}")
    print(f"  Validation N: {len(val)}")
    print(f"  OOS N: {len(oos)}")
    print(f"  Sinyal: Train={n_sinyal_train}, Val={n_sinyal_val}, OOS={n_sinyal_oos}")

    print("\n--- Dosya okuma sayacı ---")
    for ad in sorted(_okuma_sayaci.keys()):
        print(f"  {ad}: READ {_okuma_sayaci[ad]} TIMES")
    for ad in YASAKLI_DOSYALAR:
        print(f"  [YASAKLI] {ad}: READ {_okuma_sayaci.get(ad, 0)} TIMES")


if __name__ == "__main__":
    main()