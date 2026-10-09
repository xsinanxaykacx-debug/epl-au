# -*- coding: utf-8 -*-
"""
h13_corner_market.py
H13 FROZEN v1.0 — Corner Market Inefficiency (düzeltilmiş)
Tarih: 2026-10-08

DÜZELTMELER:
1. Same-day leakage: Bir takımın formuna yalnızca "kesinlikle önceki
   günlerden" gelen maçlar dahil edilir. Aynı Date içindeki hiçbir maç
   birbirinin formuna katkı yapmaz.
2. Permutation testi: Sign-permutation (FTR label permütasyonu değil,
   çünkü korner oranı CSV'de yok). H0: kazanç/kayıp işaretleri rastgele.
3. Break-even oranı rapora eklendi: 1/1.90 = %52.6316.
4. Form başlangıçları raporlanıyor (Train/Val/OOS).
5. Leakage kanıtı Excel'e eklendi.

Hipotez (netleştirilmiş):
İki takımın son 5 maçtaki korner üretim ortalaması toplamı,
maç korner toplamının 10.5 çizgisinin yönünü tahmin edebilir mi?

Test yapısı:
- Sinyal: HomeForm(N=5) + AwayForm(N=5) > 10.5 ise Over, < ise Under
- Bahis: Over/Under, yaklaşık 1.90 oran
- Stake: 1 birim
- Settlement: doğru → +0.90; yanlış → -1
- Break-even: Win% > 52.6316

KİLİTLER:
- E0 (10).csv ASLA OKUNMAZ.
- Çizgi 10.5 FROZEN.
- N=5 FROZEN.
- Yaklaşık oran 1.90 FROZEN.
- BCa 10.000, seed=20261008
- Permutation 10.000, iki taraflı, seed=20261008 (sign-permutation)
- MDD: bankroll 100, stable kronolojik
- N_MIN=100, ALPHA=0.05, MDD_ESIK=0.20

ÇIKTI: h13_corner_market_results.xlsx

UYARI:
Bu script gerçek korner ROI'si değil, sinyal doğruluk testidir.
Korner oranları CSV'de yoktur; yaklaşık 1.90 kullanılır.
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
CIKTI = "h13_corner_market_results.xlsx"

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

CORNER_LINE = 10.5
N_FORM = 5
YAKLASIK_ORAN = 1.90
BREAK_EVEN_WIN = 1.0 / YAKLASIK_ORAN  # 0.526316

N_BOOTSTRAP = 10_000
N_PERMUTATION = 10_000
RANDOM_SEED = 20261008
BANKROLL = 100.0
MDD_ESIK = 0.20
ALPHA = 0.05
N_MIN = 100

BEKLENEN_TOPLAM = 3800

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
    GEREKLI = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC"]

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
    for c in ["HC", "AC", "Date", "HomeTeam", "AwayTeam"]:
        n_nan = int(df[c].isna().sum())
        print(f"    {c:12s} NaN={n_nan}")

    gecerli = (
        df["HC"].notna() & df["AC"].notna() &
        df["Date"].notna() & df["HomeTeam"].notna() & df["AwayTeam"].notna()
    )
    n_dusen = int((~gecerli).sum())
    print(f"  Toplam düşen satır: {n_dusen}")
    df = df[gecerli].copy().reset_index(drop=True)
    print(f"  Temizlik sonrası: {len(df)}")

    df["TotalCorners"] = df["HC"] + df["AC"]
    df["Date_p"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")

    return df


# ==================================================================
# 2. FORM (same-day leakage ÖNLEYİCİ)
# ==================================================================
def form_hesapla(df):
    """
    Her maç için ev sahibi ve deplasman son N maç korner ortalaması.

    KRİTİK: Aynı takımın aynı gün iki maçı olamaz ama aynı gün başka
    maçların sonucu formu etkileyebilir. Bu yüzden form yalnızca
    "kesinlikle önceki günlerden" gelen maçları kullanır.

    Yöntem:
    - Tüm maçları tarihe göre sırala
    - Her gün için, o günün maçlarını işlemeden ÖNCE o güne kadar
      birikmiş geçmişi kullan
    - Sonra o günün maçlarını geçmişe ekle
    """
    df = df.copy().sort_values("Date_p", kind="stable").reset_index(drop=True)

    takim_gecmis = defaultdict(list)  # takım -> [(date, korner), ...]

    home_avg = np.full(len(df), np.nan)
    away_avg = np.full(len(df), np.nan)

    # Günlere göre grupla
    df["_date_only"] = df["Date_p"].dt.normalize()

    for date_only, grup in df.groupby("_date_only", sort=True):
        idx_list = grup.index.tolist()

        # ÖNCE: bu günün maçları için form hesapla (geçmiş = önceki günler)
        for i in idx_list:
            row = df.loc[i]
            ev = row["HomeTeam"]
            dep = row["AwayTeam"]

            ev_list = [k for (d, k) in takim_gecmis.get(ev, [])][-N_FORM:]
            dep_list = [k for (d, k) in takim_gecmis.get(dep, [])][-N_FORM:]

            if len(ev_list) == N_FORM and len(dep_list) == N_FORM:
                home_avg[i] = float(np.mean(ev_list))
                away_avg[i] = float(np.mean(dep_list))

        # SONRA: bu günün maçlarını geçmişe ekle (sonraki günler için)
        for i in idx_list:
            row = df.loc[i]
            takim_gecmis[row["HomeTeam"]].append((date_only, row["HC"]))
            takim_gecmis[row["AwayTeam"]].append((date_only, row["AC"]))

    df["HomeForm"] = home_avg
    df["AwayForm"] = away_avg
    df["BeklenenKorner"] = df["HomeForm"] + df["AwayForm"]

    df = df.drop(columns=["_date_only"])
    return df


# ==================================================================
# 3. BÖLME
# ==================================================================
def bolmeleri_ayir(df):
    train = df[df["Season"].isin(TRAIN_SEZONLAR)].copy().reset_index(drop=True)
    val = df[df["Season"] == VALIDATION_SEZON].copy().reset_index(drop=True)
    oos = df[df["Season"] == OOS_SEZON].copy().reset_index(drop=True)

    print(f"\n  Bölme boyutları:")
    print(f"    Train: {len(train)}")
    print(f"    Val:   {len(val)}")
    print(f"    OOS:   {len(oos)}")

    return train, val, oos


# ==================================================================
# 4. BCa BOOTSTRAP
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
# 5. PERMUTATION (SIGN-PERMUTATION)
# ==================================================================
def permutation_test_sign(pl, n_perm=N_PERMUTATION, seed=RANDOM_SEED):
    """
    SIGN-PERMUTATION.
    H0: kazanç/kayıp işaretleri rastgele (yön seçimi şans eseri).
    Her permütasyonda |pl| değerleri rastgele +/- işaret alır.
    İki taraflı.
    """
    n = len(pl)
    if n < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = pl.mean()
    abs_pl = np.abs(pl)
    ekstrem = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_mean = (abs_pl * signs).mean()
        if abs(perm_mean) >= abs(obs):
            ekstrem += 1
    return (1 + ekstrem) / (n_perm + 1)


# ==================================================================
# 6. MDD (stable)
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
# 7. BAHİS SİMÜLASYONU
# ==================================================================
def bahis_simule(df_bolme, etiket):
    alt = df_bolme.dropna(subset=["BeklenenKorner"]).copy().reset_index(drop=True)

    if len(alt) == 0:
        return None

    # Over / Under kararı
    over_mask = (alt["BeklenenKorner"] > CORNER_LINE).values
    under_mask = (alt["BeklenenKorner"] < CORNER_LINE).values
    equal_mask = (alt["BeklenenKorner"] == CORNER_LINE).values

    total = alt["TotalCorners"].values
    pl = np.zeros(len(alt))

    for i in range(len(alt)):
        if over_mask[i]:
            kazandi = total[i] > CORNER_LINE
        elif under_mask[i]:
            kazandi = total[i] < CORNER_LINE
        else:
            pl[i] = 0.0
            continue
        pl[i] = (YAKLASIK_ORAN - 1.0) if kazandi else -1.0

    bahis_mask = ~equal_mask
    pl_bahis = pl[bahis_mask]
    n_bahis = len(pl_bahis)

    if n_bahis < 5:
        return None

    win = int((pl_bahis > 0).sum())
    win_pct = win / n_bahis
    roi = float(pl_bahis.mean())

    ci_lo, ci_hi = bca_bootstrap_roi(pl_bahis)
    p_perm = permutation_test_sign(pl_bahis)
    tarih = alt["Date_p"].values[bahis_mask]
    mdd = max_drawdown(pl_bahis, tarih)

    return {
        "Bolme": etiket,
        "N": n_bahis,
        "Win": win,
        "Win_pct": round(win_pct * 100, 2),
        "Break_even_pct": round(BREAK_EVEN_WIN * 100, 4),
        "Net_PL": round(float(pl_bahis.sum()), 4),
        "ROI": round(roi, 6),
        "ROI_GA_lo": round(ci_lo, 6) if not np.isnan(ci_lo) else None,
        "ROI_GA_hi": round(ci_hi, 6) if not np.isnan(ci_hi) else None,
        "Perm_p": round(p_perm, 6) if not np.isnan(p_perm) else None,
        "MDD": round(mdd, 6),
    }


# ==================================================================
# 8. KARAR
# ==================================================================
def karar_uygula(d_oos, d_val):
    if d_oos is None or d_val is None:
        return "YETERSIZ_VERI"

    roi = d_oos["ROI"]
    ga_lo = d_oos["ROI_GA_lo"]
    n = d_oos["N"]
    p = d_oos["Perm_p"]
    mdd = d_oos["MDD"]
    roi_v = d_val["ROI"]

    k_val = roi_v is not None and roi_v > 0
    k_n = n >= N_MIN
    k_roi = roi is not None and roi > 0
    k_ga = ga_lo is not None and ga_lo > 0
    k_p = p is not None and p < ALPHA
    k_mdd = mdd is not None and mdd <= MDD_ESIK

    if k_val and k_n and k_roi and k_ga and k_p and k_mdd:
        return "SUPPORTED"
    elif roi is not None and roi > 0:
        return "WEAK / UNCERTAIN"
    else:
        return "NOT SUPPORTED"


# ==================================================================
# 9. ANA AKIŞ
# ==================================================================
def main():
    print("=" * 80)
    print("H13 FROZEN v1.0 — Corner Market Inefficiency (düzeltilmiş)")
    print("Çizgi: 10.5 korner | Form: son 5 maç | Yaklaşık oran: 1.90")
    print(f"Break-even Win%: {BREAK_EVEN_WIN*100:.4f}")
    print("E0 (10).csv KESİNLİKLE OKUNMAYACAK.")
    print("Same-day leakage ÖNLEYİCİ form hesabı aktif.")
    print("Permutation: sign-permutation (10.000, iki taraflı)")
    print("=" * 80)

    df = veri_yukle()
    df = form_hesapla(df)

    print(f"\n  Toplam korner istatistikleri (tüm veri):")
    print(f"    Min: {df['TotalCorners'].min()}")
    print(f"    Max: {df['TotalCorners'].max()}")
    print(f"    Ort: {df['TotalCorners'].mean():.2f}")
    print(f"    Medyan: {df['TotalCorners'].median():.2f}")

    n_form = df["BeklenenKorner"].notna().sum()
    print(f"\n  Form hesaplanabilen maç: {n_form} / {len(df)}")

    train, val, oos = bolmeleri_ayir(df)

    # Form başlangıçları (leakage kanıtı)
    print(f"\n  Form başlangıçları (leakage kanıtı):")
    for etiket, d in [("Train", train), ("Validation", val), ("OOS", oos)]:
        first = d.dropna(subset=["BeklenenKorner"])
        if len(first) > 0:
            print(f"    {etiket:11s} ilk form tarihi: {first['Date_p'].min().date()}")
            print(f"    {etiket:11s} form dolu: {len(first)} / {len(d)}")
        else:
            print(f"    {etiket:11s} form dolu: 0")

    print("\n--- Bahis Simülasyonu ---")
    d_train = bahis_simule(train, "Train")
    d_val = bahis_simule(val, "Validation")
    d_oos = bahis_simule(oos, "OOS")

    ozet = []
    for d in [d_train, d_val, d_oos]:
        if d is None:
            ozet.append({"Bolme": "YOK", "N": 0})
            continue
        ozet.append(d)

    ozet_df = pd.DataFrame(ozet)
    print(ozet_df.to_string(index=False))

    print("\n--- KARAR ---")
    karar = karar_uygula(d_oos, d_val)
    print(f"  {karar}")

    kirmizi = False
    if d_oos is not None and d_train is not None:
        if d_train["ROI"] > 0 and d_oos["ROI"] < 0:
            kirmizi = True
            print(f"  ⚠ KIRMIZI BAYRAK: Train ROI > 0, OOS ROI < 0")

    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        ozet_df.to_excel(w, sheet_name="Ozet", index=False)

        pd.DataFrame([{
            "Protokol": "H13 FROZEN v1.0 — Corner Market",
            "Tarih": "2026-10-08",
            "Hipotez": "Son 5 mac korner ortalamalari toplami, mac korner toplaminin 10.5 yonunu tahmin eder",
            "Corner_line": CORNER_LINE,
            "N_form": N_FORM,
            "Yaklasik_oran": YAKLASIK_ORAN,
            "Break_even_win_pct": round(BREAK_EVEN_WIN * 100, 4),
            "UYARI": "Korner oranlari CSV'de YOK. Yaklasik 1.90 kullanildi. Gercek ROI degil, sinyal testi.",
            "Same_day_leakage": "ONLENDI - form yalnizca onceki gunlerden",
            "Permutation": "sign-permutation, 10.000, iki tarafli, seed=20261008",
            "BCa": "10.000, seed=20261008",
            "MDD": "bankroll=100, stable kronolojik",
            "Train": "2015/16 - 2022/23",
            "Validation": "2023/24",
            "OOS": "2024/25",
            "2026_27": "KILITLI - okunmadi",
            "Karar": karar,
            "Kirmizi_bayrak": kirmizi,
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} yazıldı.")

    print("\n" + "=" * 80)
    print("VERİ BÜTÜNLÜĞÜ VE KANIT")
    print("=" * 80)
    print(f"  Toplam: {len(df)}")
    print(f"  Train N: {len(train)}")
    print(f"  Val N:   {len(val)}")
    print(f"  OOS N:   {len(oos)}")
    print(f"  Okuma sayacı: {dict(_okuma_sayaci)}")
    print(f"  Yasaklılardan hiçbiri okunmadı: "
          f"{all(d not in _okuma_sayaci for d in YASAKLI_DOSYALAR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()