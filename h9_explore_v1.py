# -*- coding: utf-8 -*-
"""
h9_explore_v1.py
H9 FROZEN PROTOCOL v1.0 — Value Betting (Multinomial Logistic)
Tarih: 2026-10-08

Kapsam:
- Model: Multinomial Logistic Regression (FTR: H/D/A)
- Feature: 49 sayısal pre-match (h1_features_v2.csv)
- Preprocessing: Train median + Train standardizasyon
- Piyasa: B365 açılış, normalize olasılık
- Value: P_model × odds − 1 > 0
- Bahis seçimi: En yüksek pozitif value (tek sonuç)
- Stake: 1 birim sabit
- Settlement: Doğru → +(odds−1); Yanlış → −1
- Train: 2015/16 – 2022/23 (3039)
- Validation: 2023/24 (379)
- Blind OOS: 2024/25 (380)

KİLİTLER:
- E0 (10).csv ASLA OKUNMAZ.
- Model yalnızca Train'de fit edilir; Validation/OOS'ta refit YOK.
- Median ve standardizasyon parametreleri yalnızca Train'den üretilir.
- Bootstrap: BCa 10.000, seed=20261008
- Permutation: 10.000, FTR etiketleri, iki taraflı, seed=20261008
- MDD: Bankroll 100, kronolojik
- Post-hoc değişiklik YASAK.

DÜZELTMELER (bu sürüm):
1. "_eksik_ah" CIKARILACAK listesine eklendi (yardımcı sütun).
2. multi_class="multinomial" parametresi kaldırıldı (sklearn >= 1.5 uyumu).
   solver="lbfgs" ile multinomial davranış zaten otomatik.
3. Line 678 syntax hatası düzeltildi (eksik parantez kapatıldı).
4. Dosya sonu tamamlandı (main guard + kapanış print'leri).

KARAR:
- SUPPORTED: OOS ROI>0 + GA alt sınırı>0 + N>=100 + p<0.05 + MDD<=%20
- WEAK / UNCERTAIN: ROI>0 ama bir kriter eksik
- NOT SUPPORTED: ROI<=0 veya MDD>%20
- KIRMIZI BAYRAK: Train ROI>0 ve OOS ROI<0

ÇIKTI: h9_explore_v1_results.xlsx
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ==================================================================
# 0. SABİTLER (KİLİTLİ)
# ==================================================================
H1_FEATURES = "h1_features_v2.csv"
ENVANTER = "envanter_v2.xlsx"
ENVANTER_SHEET = "Eksik_Maclar"
CIKTI = "h9_explore_v1_results.xlsx"

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

MARKET_SUTUNLAR = ["B365H", "B365D", "B365A"]

# "_eksik_ah" yardımcı sütun olarak dışlandı
CIKARILACAK = ["Season", "Date", "HomeTeam", "AwayTeam",
               "FTHG", "FTAG", "FTR", "_key", "_eksik_ah"]

TRAIN_SEZONLAR = ["2015/16", "2016/17", "2017/18", "2018/19",
                  "2019/20", "2020/21", "2021/22", "2022/23"]
VALIDATION_SEZON = "2023/24"
OOS_SEZON = "2024/25"

# Model parametreleri (FROZEN)
MODEL_SOLVER = "lbfgs"
MODEL_C = 1.0
MODEL_MAX_ITER = 5000
RANDOM_STATE = 20261008

# İstatistiksel parametreler (FROZEN)
N_BOOTSTRAP = 10_000
N_PERMUTATION = 10_000
RANDOM_SEED = 20261008
BANKROLL = 100.0
MDD_ESIK = 0.20
ALPHA = 0.05
N_MIN = 100

BEKLENEN_TOP = 3800
BEKLENEN_TRAIN = 3039
BEKLENEN_VAL = 379
BEKLENEN_OOS = 380
BEKLENEN_TOPLAM_FILTRE = 3798

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
    # 2016/17 tarihleri DD/MM/YY biçiminde; format inference tüm seride
    # 380 tarihi NaT yapabiliyor. Her satırı mixed-format olarak ayrıştır.
    try:
        parsed = pd.to_datetime(
            df["Date"], dayfirst=True, format="mixed", errors="coerce"
        )
    except (TypeError, ValueError):
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
# 2. VERİ YÜKLEME
# ==================================================================
def veri_yukle():
    print("\n--- Veri yükleniyor ---")

    if not os.path.exists(H1_FEATURES):
        raise RuntimeError(f"{H1_FEATURES} bulunamadı.")
    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")

    for c in ["FTR", "Season", "Date", "HomeTeam", "AwayTeam"]:
        if c not in feat.columns:
            raise RuntimeError(f"{H1_FEATURES} içinde {c} yok.")

    # Market yükleme
    frames = []
    for dosya, sezon in TUM_DOSYALAR:
        if not os.path.exists(dosya):
            raise RuntimeError(f"EKSİK DOSYA: {dosya}")
        df = _guvenli_oku(dosya)
        df["Season"] = sezon
        eksik = [s for s in MARKET_SUTUNLAR if s not in df.columns]
        if eksik:
            raise RuntimeError(f"{dosya}: eksik sütunlar {eksik}")
        frames.append(df[["Season", "Date", "HomeTeam", "AwayTeam"] + MARKET_SUTUNLAR].copy())
    market = pd.concat(frames, ignore_index=True)

    feat["_key"] = _mac_anahtari(feat)
    market["_key"] = _mac_anahtari(market)

    df = feat.merge(market[["_key", "B365H", "B365D", "B365A"]], on="_key", how="inner")
    print(f"  Merge sonrası: {len(df)}")
    if len(df) != BEKLENEN_TOP:
        raise RuntimeError(f"Merge sonrası {len(df)} != {BEKLENEN_TOP}")

    dup = df["_key"].duplicated().sum()
    if dup > 0:
        raise RuntimeError(f"Duplicate: {dup}")

    # AH ailesi eksik filtresi
    eksik = pd.read_excel(ENVANTER, sheet_name=ENVANTER_SHEET)
    eksik["_key"] = _mac_anahtari(eksik)
    eksik_ah = set(eksik.loc[eksik["AH_Eksik"] == True, "_key"].dropna().astype(str))

    df["_eksik_ah"] = df["_key"].isin(eksik_ah)
    onceki = len(df)
    df = df[~df["_eksik_ah"]].copy().reset_index(drop=True)
    sonraki = len(df)
    print(f"  AH ailesi eksik filtresi: {onceki} → {sonraki} (dışlanan: {onceki - sonraki})")

    if len(df) != BEKLENEN_TOPLAM_FILTRE:
        raise RuntimeError(f"Filtre sonrası {len(df)} != {BEKLENEN_TOPLAM_FILTRE}")

    return df


# ==================================================================
# 3. FEATURE HAZIRLIĞI (Train median + Train standardizasyon)
# ==================================================================
def feature_hazirla(df):
    print("\n--- Feature hazırlığı ---")

    feature_sutunlar = [c for c in df.columns if c not in CIKARILACAK
                        and c not in ["B365H", "B365D", "B365A"]]

    print(f"  Feature sayısı: {len(feature_sutunlar)}")
    if len(feature_sutunlar) != 49:
        raise RuntimeError(f"Feature sayısı {len(feature_sutunlar)} != 49")

    # Bölmeler
    train = df[df["Season"].isin(TRAIN_SEZONLAR)].copy().reset_index(drop=True)
    val = df[df["Season"] == VALIDATION_SEZON].copy().reset_index(drop=True)
    oos = df[df["Season"] == OOS_SEZON].copy().reset_index(drop=True)

    print(f"  Train: {len(train)}, Val: {len(val)}, OOS: {len(oos)}")

    if len(train) != BEKLENEN_TRAIN:
        raise RuntimeError(f"Train N: {len(train)} != {BEKLENEN_TRAIN}")
    if len(val) != BEKLENEN_VAL:
        raise RuntimeError(f"Val N: {len(val)} != {BEKLENEN_VAL}")
    if len(oos) != BEKLENEN_OOS:
        raise RuntimeError(f"OOS N: {len(oos)} != {BEKLENEN_OOS}")

    # Train median (yalnızca Train'den)
    medians = train[feature_sutunlar].median()
    for s in (train, val, oos):
        s[feature_sutunlar] = s[feature_sutunlar].fillna(medians)

    # Train standardizasyon (yalnızca Train'den)
    scaler = StandardScaler()
    scaler.fit(train[feature_sutunlar])

    X_train = scaler.transform(train[feature_sutunlar])
    X_val = scaler.transform(val[feature_sutunlar])
    X_oos = scaler.transform(oos[feature_sutunlar])

    y_train = train["FTR"].values
    y_val = val["FTR"].values
    y_oos = oos["FTR"].values

    return {
        "train": train, "val": val, "oos": oos,
        "X_train": X_train, "X_val": X_val, "X_oos": X_oos,
        "y_train": y_train, "y_val": y_val, "y_oos": y_oos,
        "feature_sutunlar": feature_sutunlar,
    }


# ==================================================================
# 4. MODEL EĞİTİMİ (yalnızca Train)
# ==================================================================
def model_egit(X_train, y_train):
    print("\n--- Model eğitiliyor (yalnızca Train) ---")

    # sklearn >= 1.5: multi_class parametresi kaldırıldı.
    # solver="lbfgs" ile multinomial davranış otomatiktir.
    model = LogisticRegression(
        solver=MODEL_SOLVER,
        C=MODEL_C,
        max_iter=MODEL_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    print(f"  Model fit edildi. Classes: {model.classes_}")
    return model


# ==================================================================
# 5. VALUE VE BAHİS HESABI
# ==================================================================
def value_ve_bahis(df_bolme, X_bolme, model):
    """
    Her maç için:
    - P_model (H/D/A)
    - P_market normalize
    - Value = P_model × odds − 1
    - En yüksek pozitif value'lu sonuca 1 birim bahis
    """
    probs = model.predict_proba(X_bolme)
    classes = model.classes_  # sklearn alfabetik sıralar: ['A', 'D', 'H']

    idx_H = list(classes).index("H")
    idx_D = list(classes).index("D")
    idx_A = list(classes).index("A")

    p_H = probs[:, idx_H]
    p_D = probs[:, idx_D]
    p_A = probs[:, idx_A]

    # Market olasılıkları (normalize)
    imp_H = 1.0 / df_bolme["B365H"].values
    imp_D = 1.0 / df_bolme["B365D"].values
    imp_A = 1.0 / df_bolme["B365A"].values
    toplam = imp_H + imp_D + imp_A
    pm_H = imp_H / toplam
    pm_D = imp_D / toplam
    pm_A = imp_A / toplam

    # Value
    val_H = p_H * df_bolme["B365H"].values - 1.0
    val_D = p_D * df_bolme["B365D"].values - 1.0
    val_A = p_A * df_bolme["B365A"].values - 1.0

    # Bahis seçimi: en yüksek pozitif value (tek sonuç)
    values = np.vstack([val_H, val_D, val_A])  # 3 × N
    secim = np.argmax(values, axis=0)  # 0=H, 1=D, 2=A
    max_val = values[secim, np.arange(len(secim))]

    bahis_mask = max_val > 0  # Value > 0

    return {
        "p_H": p_H, "p_D": p_D, "p_A": p_A,
        "pm_H": pm_H, "pm_D": pm_D, "pm_A": pm_A,
        "val_H": val_H, "val_D": val_D, "val_A": val_A,
        "secim": secim, "max_val": max_val,
        "bahis_mask": bahis_mask,
    }


def bahis_pl(df_bolme, vbilgi):
    """
    Bahis P/L hesapla.
    Settlement: Doğru → +(odds−1); Yanlış → −1
    """
    n = len(df_bolme)
    ftr = df_bolme["FTR"].values
    b365h = df_bolme["B365H"].values
    b365d = df_bolme["B365D"].values
    b365a = df_bolme["B365A"].values

    pl = np.zeros(n)
    oran = np.zeros(n)

    for i in range(n):
        if not vbilgi["bahis_mask"][i]:
            continue
        sec = vbilgi["secim"][i]
        if sec == 0:  # H
            kazandi = ftr[i] == "H"
            o = b365h[i]
        elif sec == 1:  # D
            kazandi = ftr[i] == "D"
            o = b365d[i]
        else:  # A
            kazandi = ftr[i] == "A"
            o = b365a[i]

        oran[i] = o
        pl[i] = (o - 1.0) if kazandi else -1.0

    return pl, oran


# ==================================================================
# 6. İSTATİSTİK
# ==================================================================
def bca_bootstrap_roi(pl, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """BCa %95 GA — ROI (P/L ortalaması)."""
    if len(pl) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(pl)
    theta_hat = pl.mean()

    # Bootstrap
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = pl[idx].mean()

    # Bias correction z0
    prop_less = np.mean(boot < theta_hat)
    prop_less = min(max(prop_less, 1.0/(n_boot+1)), n_boot/(n_boot+1))
    z0 = norm.ppf(prop_less)

    # Acceleration (jackknife)
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


def permutation_test_roi(df_bolme, vbilgi, n_perm=N_PERMUTATION, seed=RANDOM_SEED):
    """
    FTR etiketleri permüte edilir.
    Model, fiyatlar, bahis seçimleri sabit kalır.
    P/L yeniden hesaplanır. İki taraflı.
    """
    if len(df_bolme) < 5:
        return np.nan

    rng = np.random.default_rng(seed)

    ftr = df_bolme["FTR"].to_numpy(copy=True)
    b365h = df_bolme["B365H"].to_numpy(copy=True)
    b365d = df_bolme["B365D"].to_numpy(copy=True)
    b365a = df_bolme["B365A"].to_numpy(copy=True)
    secim = vbilgi["secim"]
    bahis_mask = vbilgi["bahis_mask"]

    def _hesapla_pl(ftr_arr):
        pl = np.zeros(len(ftr_arr))
        for i in range(len(ftr_arr)):
            if not bahis_mask[i]:
                continue
            sec = secim[i]
            if sec == 0:
                pl[i] = (b365h[i] - 1.0) if ftr_arr[i] == "H" else -1.0
            elif sec == 1:
                pl[i] = (b365d[i] - 1.0) if ftr_arr[i] == "D" else -1.0
            else:
                pl[i] = (b365a[i] - 1.0) if ftr_arr[i] == "A" else -1.0
        return pl

    obs_pl = _hesapla_pl(ftr)
    n_bahis = int(bahis_mask.sum())
    obs = obs_pl.sum() / n_bahis if n_bahis > 0 else 0.0

    ekstrem = 0
    for _ in range(n_perm):
        perm_ftr = rng.permutation(ftr)
        perm_pl = _hesapla_pl(perm_ftr)
        perm_roi = perm_pl.sum() / n_bahis if n_bahis > 0 else 0.0
        if abs(perm_roi) >= abs(obs):
            ekstrem += 1

    return (1 + ekstrem) / (n_perm + 1)


def max_drawdown(pl, tarih):
    """Kronolojik sıralı P/L üzerinden MDD (bankroll=100)."""
    if len(pl) == 0:
        return 0.0
    idx = np.argsort(tarih)
    pl_sorted = pl[idx]
    cum = np.cumsum(pl_sorted)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max() / BANKROLL)


# ==================================================================
# 7. BÖLME ANALİZİ
# ==================================================================
def bolme_analiz(df_bolme, X_bolme, model, etiket):
    vbilgi = value_ve_bahis(df_bolme, X_bolme, model)
    bahis_mask = vbilgi["bahis_mask"]
    n_bahis = int(bahis_mask.sum())

    if n_bahis == 0:
        return {
            "Bolme": etiket, "N": 0, "Win": 0, "Win_pct": None,
            "Ort_oran": None, "Net_PL": None, "ROI": None,
            "ROI_GA_lo": None, "ROI_GA_hi": None,
            "Perm_p": None, "MDD": None,
        }

    pl, oran = bahis_pl(df_bolme, vbilgi)

    # Sadece bahis yapılanlar
    mask = bahis_mask
    pl_bahis = pl[mask]
    oran_bahis = oran[mask]
    tarih = pd.to_datetime(df_bolme["Date"], dayfirst=True, errors="coerce").values[mask]

    win = int((pl_bahis > 0).sum())
    ort_oran = float(oran_bahis.mean())
    net = float(pl_bahis.sum())
    roi = net / n_bahis

    ci_lo, ci_hi = bca_bootstrap_roi(pl_bahis)

    # Permutation için sadece bahis yapılan maçlar
    df_bahis = df_bolme[mask].reset_index(drop=True)
    vbilgi_bahis = {
        "secim": vbilgi["secim"][mask],
        "bahis_mask": vbilgi["bahis_mask"][mask],
    }
    p_perm = permutation_test_roi(df_bahis, vbilgi_bahis)

    mdd = max_drawdown(pl_bahis, tarih)

    return {
        "Bolme": etiket, "N": n_bahis,
        "Win": win, "Win_pct": round(win/n_bahis*100, 2),
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
# 8. ANA AKIŞ
# ==================================================================
def main():
    print("=" * 80)
    print("H9 EXPLORE v1.0 — FROZEN (Value Betting)")
    print("E0 (10).csv KESİNLİKLE OKUNMAYACAK.")
    print("Model yalnızca Train'de fit edilir; Validation/OOS refit YOK.")
    print("=" * 80)

    # 1. Veri yükle
    df = veri_yukle()

    # 2. Feature hazırla
    data = feature_hazirla(df)

    # 3. Model eğit (yalnızca Train)
    model = model_egit(data["X_train"], data["y_train"])

    # 4. Bölme analizi
    print("\n--- Bölme analizi ---")
    train_metrik = bolme_analiz(data["train"], data["X_train"], model, "Train")
    val_metrik = bolme_analiz(data["val"], data["X_val"], model, "Validation")
    oos_metrik = bolme_analiz(data["oos"], data["X_oos"], model, "OOS")

    ozet_df = pd.DataFrame([train_metrik, val_metrik, oos_metrik])
    print(ozet_df.to_string(index=False))

    # 5. Karar
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
        print(f"  ⚠ KIRMIZI BAYRAK: Train ROI > 0, OOS ROI < 0")

    # 6. Excel
    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        ozet_df.to_excel(w, sheet_name="Ozet", index=False)

        # Her bölme için bahis listesi
        for etiket, df_b, X_b in [
            ("Train", data["train"], data["X_train"]),
            ("Validation", data["val"], data["X_val"]),
            ("OOS", data["oos"], data["X_oos"]),
        ]:
            vbilgi = value_ve_bahis(df_b, X_b, model)
            mask = vbilgi["bahis_mask"]
            if mask.sum() == 0:
                continue

            pl, oran = bahis_pl(df_b, vbilgi)
            alt = df_b[mask].copy().reset_index(drop=True)
            alt["P_model_H"] = vbilgi["p_H"][mask]
            alt["P_model_D"] = vbilgi["p_D"][mask]
            alt["P_model_A"] = vbilgi["p_A"][mask]
            alt["Val_H"] = vbilgi["val_H"][mask]
            alt["Val_D"] = vbilgi["val_D"][mask]
            alt["Val_A"] = vbilgi["val_A"][mask]
            alt["Max_Value"] = vbilgi["max_val"][mask]
            alt["Bahis_Taraf"] = np.where(vbilgi["secim"][mask] == 0, "H",
                                  np.where(vbilgi["secim"][mask] == 1, "D", "A"))
            alt["Oran"] = oran[mask]
            alt["P/L"] = pl[mask]
            alt["Kumulatif_PL"] = np.cumsum(pl[mask])
            cols = ["Season", "Date", "HomeTeam", "AwayTeam",
                    "B365H", "B365D", "B365A", "FTR",
                    "P_model_H", "P_model_D", "P_model_A",
                    "Val_H", "Val_D", "Val_A", "Max_Value",
                    "Bahis_Taraf", "Oran", "P/L", "Kumulatif_PL"]
            alt[cols].to_excel(w, sheet_name=f"Bahis_{etiket}", index=False)

        # Karar detayı
        pd.DataFrame([{
            "Model": "Multinomial Logistic Regression",
            "Feature_sayisi": 49,
            "Solver": MODEL_SOLVER,
            "C": MODEL_C,
            "Max_iter": MODEL_MAX_ITER,
            "Value_esik": 0,
            "Bahis_secimi": "En yuksek pozitif value (tek sonuc)",
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

        # Notlar
        pd.DataFrame([{
            "Protokol": "H9 FROZEN PROTOCOL v1.0",
            "Tarih": "2026-10-08",
            "Model": "Multinomial Logistic Regression",
            "Feature": "49 sayisal pre-match (h1_features_v2.csv)",
            "Preprocessing": "Train median + Train standardizasyon",
            "Market": "B365 acilis, normalize",
            "Value": "P_model × odds - 1 > 0",
            "Bahis_secimi": "En yuksek pozitif value (tek sonuc)",
            "Stake": "1 birim sabit",
            "Settlement": "Dogru: +(odds-1); Yanlis: -1",
            "Bootstrap": "BCa 10.000, seed=20261008",
            "Permutation": "10.000, FTR etiketleri, iki tarafli, seed=20261008",
            "MDD_bankroll": 100,
            "N_min": N_MIN,
            "Train": "2015/16 - 2022/23 (3039)",
            "Validation": "2023/24 (379)",
            "OOS": "2024/25 (380)",
            "2026_27": "KILITLI - okunmadi",
            "KARAR": karar_dict["Karar"],
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} başarıyla yazıldı.")

    print("\n" + "=" * 80)
    print("VERİ BÜTÜNLÜĞÜ VE KANIT")
    print("=" * 80)
    print(f"  Toplam: {len(df)}")
    print(f"  Train N: {len(data['train'])}")
    print(f"  Val N:   {len(data['val'])}")
    print(f"  OOS N:   {len(data['oos'])}")
    print(f"  Feature sayısı: {len(data['feature_sutunlar'])}")
    print(f"  Okuma sayacı: {dict(_okuma_sayaci)}")
    print(f"  Yasaklı dosyalar (okunmadı): {sorted(YASAKLI_DOSYALAR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()