# -*- coding: utf-8 -*-
"""
h1_edge_v1.py
H1 EDGE TEST PROTOKOLÜ v1.0 (FROZEN) — birebir implementasyon
Tarih: 2026-10-08

Kapsam:
- Train:       2019/20 + 2020/21 + 2021/22 + 2022/23
- Validation:  2023/24
- OOS:         2024/25 (yalnızca Validation'da GÜÇLÜ KANIT veren hipotezler için)

YASAKLAR:
- E0 (10).csv (2026/27) ASLA OKUNMAZ.
- Post-hoc eşik araması, kombinasyon patlaması, Kelly/staking YOK.

KİLİTLİ KARARLAR:
- Market movement = AHCh − AHh, eşik ±0.25
- Elo kategorileri: <0 (DÜŞÜK), 0–100 (ORTA), >100 (YÜKSEK)
- 9 hipotez: 3 Elo × 3 Market
- Form5 yalnızca tanımlayıcı raporlama
- Bahis yönü: EV→HOME, DEP→AWAY, NÖTR→bahis yok
- B365 açılış (executable) + B365 kapanış (benchmark)
- Quarter-AH split-stake settlement
- 1 birim sabit stake
- N_min = 100, alpha = 0.05, MDD_esik = 0.20, bankroll = 100
- BCa bootstrap: 10.000, seed = 20261008
- Permutation: 10.000, sonuç etiketleri, iki taraflı, seed = 20261008
- BH-FDR: q < 0.05, 9 hipotez

DÜZELTME (bu sürüm):
- fdr_uygula() içindeki erken return kaldırıldı.
  Artık FDR yalnızca geçerli p-değerleri için uygulanır; karar döngüsü
  her satır için bağımsız çalışır. Böylece:
    n=0 → "Test yok"
    n=1,2,...(<100) → "Kanıt yetersiz (N<100)"
  doğru etiketler üretilir.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

# ==================================================================
# 0. SABİTLER (KİLİTLİ)
# ==================================================================
H1_FEATURES = "h1_features.csv"
ENVANTER = "envanter_v2.xlsx"
ENVANTER_SHEET = "Eksik_Maclar"
CIKTI = "h1_edge_v1_results.xlsx"

YASAKLI_DOSYALAR = {"E0 (10).csv"}

MARKET_DOSYALARI = {
    "2019/20": "E0 (4).csv",
    "2020/21": "E0 (5).csv",
    "2021/22": "E0 (6).csv",
    "2022/23": "E0 (7).csv",
    "2023/24": "E0 (8).csv",
    "2024/25": "E0 (9).csv",
}
MARKET_SUTUNLARI = ["Season", "Date", "HomeTeam", "AwayTeam",
                    "AHh", "AHCh",
                    "B365AHH", "B365AHA", "B365CAHH", "B365CAHA"]

TRAIN_SEZONLAR = ["2019/20", "2020/21", "2021/22", "2022/23"]
VALIDATION_SEZON = "2023/24"
OOS_SEZON = "2024/25"

MOVEMENT_ESIK = 0.25

def elo_kategori(elo_diff):
    if pd.isna(elo_diff):
        return None
    if elo_diff < 0:
        return "DUSUK"
    elif elo_diff <= 100:
        return "ORTA"
    else:
        return "YUKSEK"

def market_sinif(movement):
    if pd.isna(movement):
        return None
    if movement > MOVEMENT_ESIK:
        return "EV_YONLU"
    elif movement < -MOVEMENT_ESIK:
        return "DEPLASMAN_YONLU"
    else:
        return "NOTR"

def bahis_yonu(market_sinif):
    if market_sinif == "EV_YONLU":
        return "HOME"
    elif market_sinif == "DEPLASMAN_YONLU":
        return "AWAY"
    else:
        return None

N_BOOTSTRAP = 10_000
N_PERMUTATION = 10_000
RANDOM_SEED = 20261008
BANKROLL = 100.0
MDD_ESIK = 0.20
ALPHA = 0.05
N_MIN = 100


# ==================================================================
# 1. QUARTER-AH SETTLEMENT BİRİM TESTLERİ
# ==================================================================
def _ah_settle_yarim(handicap, gol_fark, taraf):
    """Tek yarım çizgi: +1 win, 0 push, -1 loss."""
    if taraf == "Home":
        adj = gol_fark + handicap
    else:
        adj = -gol_fark - handicap
    if adj > 0:
        return 1.0
    elif adj == 0:
        return 0.0
    else:
        return -1.0


def ah_settlement(handicap, gol_fark, taraf, oran):
    """Quarter-AH split-stake. Dönüş: net P/L (stake=1)."""
    if handicap is None or pd.isna(handicap):
        return 0.0
    is_quarter = (abs(handicap * 4) % 2 == 1)
    if is_quarter:
        h1 = handicap - 0.25 if handicap > 0 else handicap + 0.25
        h2 = handicap + 0.25 if handicap > 0 else handicap - 0.25
        r1 = _ah_settle_yarim(h1, gol_fark, taraf)
        r2 = _ah_settle_yarim(h2, gol_fark, taraf)
        pl = 0.0
        for r in (r1, r2):
            if r == 1.0:
                pl += 0.5 * (oran - 1.0)
            elif r == -1.0:
                pl += 0.5 * (-1.0)
        return pl
    else:
        r = _ah_settle_yarim(handicap, gol_fark, taraf)
        if r == 1.0:
            return oran - 1.0
        elif r == -1.0:
            return -1.0
        else:
            return 0.0


def _test_ah_settlement():
    """Quarter-AH settlement birim testleri (KİLİTLİ)."""
    ORA = 2.00
    NET = ORA - 1.0

    testler = [
        (-0.25, +1, "Home", +1.0 * NET,   "AHh=-0.25, ev +1 farkla → +1.00×net"),
        (+0.25,  0, "Home", +0.5 * NET,   "AHh=+0.25, beraberlik → +0.50×net"),
        (-0.75, +1, "Home", +0.5 * NET,   "AHh=-0.75, ev +1 farkla → +0.50×net"),
        (-0.75, +2, "Home", +1.0 * NET,   "AHh=-0.75, ev +2 farkla → +1.00×net"),
        (-0.75,  0, "Home", -1.0 * 1.0,   "AHh=-0.75, beraberlik → -1.00×stake"),
        (+0.25, -1, "Home", -1.0 * 1.0,   "AHh=+0.25, ev kaybetti → -1.00×stake"),
        ( 0.0, +1, "Home", +1.0 * NET,    "AHh=0, ev kazandı → +1.00×net"),
        ( 0.0,  0, "Home", 0.0,           "AHh=0, beraberlik → push"),
        (-0.5, +1, "Home", +1.0 * NET,    "AHh=-0.5, ev kazandı → +1.00×net"),
        (-0.5,  0, "Home", -1.0 * 1.0,    "AHh=-0.5, beraberlik → -1.00×stake"),
        (-1.0, +1, "Home", 0.0,           "AHh=-1.0, ev +1 farkla → push"),
        (-1.0, +2, "Home", +1.0 * NET,    "AHh=-1.0, ev +2 farkla → +1.00×net"),
    ]

    hatalar = []
    for i, (ah, gf, taraf, beklenen, aciklama) in enumerate(testler, 1):
        sonuc = ah_settlement(ah, gf, taraf, ORA)
        ok = abs(sonuc - beklenen) < 1e-9
        durum = "PASS" if ok else "FAIL"
        print(f"  Test {i:2d}: {durum} | AHh={ah:+.2f}, gol_fark={gf:+d}, "
              f"taraf={taraf:5s} | beklenen={beklenen:+.4f}, sonuç={sonuc:+.4f} | {aciklama}")
        if not ok:
            hatalar.append((i, beklenen, sonuc))

    if hatalar:
        msg = "Quarter-AH settlement birim testleri BAŞARISIZ:\n"
        for i, b, s in hatalar:
            msg += f"  Test {i}: beklenen={b}, sonuç={s}\n"
        raise RuntimeError(msg)

    print("  → TÜM QUARTER-AH TESTLERİ PASS")


# ==================================================================
# 2. VERİ YÜKLEME
# ==================================================================
def _mac_anahtari(df):
    tarih = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
    return (df["Season"].astype(str).str.strip()
            + "|" + tarih.fillna("NA")
            + "|" + df["HomeTeam"].astype(str).str.strip()
            + "|" + df["AwayTeam"].astype(str).str.strip())


def _market_yukle():
    frames = []
    for sezon, dosya in MARKET_DOSYALARI.items():
        ad = os.path.basename(dosya)
        if ad in YASAKLI_DOSYALAR:
            raise RuntimeError(f"YASAKLI DOSYA: {ad}")
        df = pd.read_csv(dosya, encoding="utf-8-sig")
        df["Season"] = sezon
        eksik = [s for s in MARKET_SUTUNLARI if s not in df.columns]
        if eksik:
            raise RuntimeError(f"{ad} içinde eksik sütunlar: {eksik}")
        frames.append(df[MARKET_SUTUNLARI].copy())
    return pd.concat(frames, ignore_index=True)


def veri_yukle():
    if not os.path.exists(H1_FEATURES):
        raise RuntimeError(f"{H1_FEATURES} bulunamadı.")
    feat = pd.read_csv(H1_FEATURES, encoding="utf-8-sig")

    market = _market_yukle()

    feat["_key"] = _mac_anahtari(feat)
    market["_key"] = _mac_anahtari(market)

    df = feat.merge(
        market[["_key", "AHh", "AHCh", "B365AHH", "B365AHA", "B365CAHH", "B365CAHA"]],
        on="_key", how="inner"
    )
    df["Movement"] = df["AHCh"] - df["AHh"]
    df["Elo_Kategori"] = df["Elo_Diff"].apply(elo_kategori)
    df["Market_Sinif"] = df["Movement"].apply(market_sinif)
    df["Bahis_Yon"] = df["Market_Sinif"].apply(bahis_yonu)
    df["_tarih"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df = df.sort_values("_tarih").reset_index(drop=True)
    return df


# ==================================================================
# 3. P/L HESABI
# ==================================================================
def hesapla_pl(bets):
    return np.array([ah_settlement(b["handikap"], b["gol_fark"], b["yon"], b["oran"])
                     for b in bets], dtype=float)


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
    z_lo = norm.ppf(0.025); z_hi = norm.ppf(0.975)
    def _p(z):
        nn = z0 + z; dd = 1 - a * nn
        return norm.cdf(z0 + nn / dd) if dd != 0 else np.nan
    p_lo = min(max(_p(z_lo), 0.0), 1.0)
    p_hi = min(max(_p(z_hi), 0.0), 1.0)
    return float(np.quantile(boot, p_lo)), float(np.quantile(boot, p_hi))


def permutation_test(bets, n_perm=N_PERMUTATION, seed=RANDOM_SEED):
    if len(bets) < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    n = len(bets)

    handikaplar = np.array([b["handikap"] for b in bets], dtype=float)
    yonler      = [b["yon"] for b in bets]
    oranlar     = np.array([b["oran"] for b in bets], dtype=float)
    gol_farklar = np.array([b["gol_fark"] for b in bets], dtype=int)

    def _roi(perm_idx):
        pl = []
        for i in range(n):
            pl.append(ah_settlement(handikaplar[i], gol_farklar[perm_idx[i]],
                                    yonler[i], oranlar[i]))
        return np.mean(pl)

    identity = np.arange(n)
    obs = _roi(identity)
    ekstrem = 0
    for _ in range(n_perm):
        perm_idx = rng.permutation(n)
        if abs(_roi(perm_idx)) >= abs(obs):
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
# 5. HİPOTEZ ANALİZİ
# ==================================================================
def hipotez_calistir(df, sezonlar):
    alt = df[df["Season"].isin(sezonlar)].copy()
    sonuclar = []

    elo_kats = ["DUSUK", "ORTA", "YUKSEK"]
    market_sinifs = ["EV_YONLU", "NOTR", "DEPLASMAN_YONLU"]

    for ek in elo_kats:
        for mk in market_sinifs:
            kod = f"H1-{elo_kats.index(ek)+1 + 3*market_sinifs.index(mk)}"
            if mk == "NOTR":
                sonuclar.append({
                    "kod": kod, "elo": ek, "market": mk, "bahis_yon": None,
                    "n": 0, "win": 0, "push": 0, "win_pct": None,
                    "ort_oran": None, "net_pl": None, "roi": None,
                    "roi_ci_lo": None, "roi_ci_hi": None, "perm_p": None,
                    "mdd": None, "form5_ort": None, "form5_med": None,
                    "form5_poz_oran": None,
                })
                continue

            hucre = alt[(alt["Elo_Kategori"] == ek) & (alt["Market_Sinif"] == mk)].copy()
            hucre = hucre.dropna(subset=["Elo_Diff", "Movement",
                                          "Home_Elo_Pre", "Away_Elo_Pre"])
            if len(hucre) == 0:
                sonuclar.append({
                    "kod": kod, "elo": ek, "market": mk, "bahis_yon": None,
                    "n": 0, "win": 0, "push": 0, "win_pct": None,
                    "ort_oran": None, "net_pl": None, "roi": None,
                    "roi_ci_lo": None, "roi_ci_hi": None, "perm_p": None,
                    "mdd": None, "form5_ort": None, "form5_med": None,
                    "form5_poz_oran": None,
                })
                continue

            yon = "HOME" if mk == "EV_YONLU" else "AWAY"
            handikap_kol = "AHh"
            oran_kol = "B365AHH" if yon == "HOME" else "B365AHA"
            hucre = hucre.dropna(subset=[handikap_kol, oran_kol, "FTHG", "FTAG"])
            hucre["gol_fark"] = hucre["FTHG"] - hucre["FTAG"]

            bets = []
            for _, r in hucre.iterrows():
                bets.append({
                    "handikap": r[handikap_kol],
                    "gol_fark": int(r["gol_fark"]),
                    "yon": yon,
                    "oran": float(r[oran_kol]),
                })

            if len(bets) == 0:
                sonuclar.append({
                    "kod": kod, "elo": ek, "market": mk, "bahis_yon": yon,
                    "n": 0, "win": 0, "push": 0, "win_pct": None,
                    "ort_oran": None, "net_pl": None, "roi": None,
                    "roi_ci_lo": None, "roi_ci_hi": None, "perm_p": None,
                    "mdd": None, "form5_ort": None, "form5_med": None,
                    "form5_poz_oran": None,
                })
                continue

            pl = hesapla_pl(bets)
            n = len(pl)
            win = int((pl > 0).sum())
            push = int((pl == 0).sum())
            oranlar = [b["oran"] for b in bets]
            net = float(pl.sum())
            roi = net / n
            ci_lo, ci_hi = bca_bootstrap_roi(pl)
            p_perm = permutation_test(bets)
            mdd = max_drawdown(pl)

            f5 = hucre["Form5_Diff"].dropna()
            f5_ort = float(f5.mean()) if len(f5) > 0 else None
            f5_med = float(f5.median()) if len(f5) > 0 else None
            f5_poz = float((f5 > 0).mean()) if len(f5) > 0 else None

            sonuclar.append({
                "kod": kod, "elo": ek, "market": mk, "bahis_yon": yon,
                "n": n, "win": win, "push": push,
                "win_pct": round(win/n*100, 2),
                "ort_oran": round(float(np.mean(oranlar)), 4),
                "net_pl": round(net, 4), "roi": round(roi, 4),
                "roi_ci_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else None,
                "roi_ci_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else None,
                "perm_p": round(p_perm, 4) if not np.isnan(p_perm) else None,
                "mdd": round(mdd, 4),
                "form5_ort": round(f5_ort, 4) if f5_ort is not None else None,
                "form5_med": round(f5_med, 4) if f5_med is not None else None,
                "form5_poz_oran": round(f5_poz, 4) if f5_poz is not None else None,
            })
    return pd.DataFrame(sonuclar)


def fdr_uygula(df):
    """
    BH-FDR: 9 hipotez üzerinde.
    Boş satırlar (n=0 veya perm_p NaN) FDR ailesine DAHİL EDİLMEZ,
    ancak karar döngüsü HER satır için bağımsız çalışır.

    DÜZELTME: Erken return kaldırıldı. Artık:
      n=0 → "Test yok"
      n<100 → "Kanıt yetersiz (N<100)"
    doğru etiketlenir.
    """
    p_vals = df["perm_p"].copy()
    gecerli = p_vals.notna() & (df["n"] > 0)

    if not gecerli.any():
        # FDR uygulanamaz — ama karar döngüsü yine de çalışır.
        df["q_fdr"] = np.nan
    else:
        p_arr = p_vals[gecerli].values
        _, q_arr, _, _ = multipletests(p_arr, alpha=ALPHA, method="fdr_bh")
        df.loc[gecerli, "q_fdr"] = q_arr

    kararlar = []
    for _, r in df.iterrows():
        if r["n"] == 0:
            kararlar.append("Test yok")
            continue
        if r["n"] < N_MIN:
            kararlar.append("Kanıt yetersiz (N<100)")
            continue
        if r["roi"] is None or r["roi"] <= 0:
            kararlar.append("Reddedildi (ROI≤0)")
            continue
        if r["roi_ci_lo"] is None or r["roi_ci_lo"] <= 0:
            kararlar.append("Kanıt yetersiz (GA≤0)")
            continue
        if pd.isna(r["q_fdr"]) or r["q_fdr"] >= ALPHA:
            kararlar.append("Kanıt yetersiz (q_FDR≥0.05)")
            continue
        if r["mdd"] > MDD_ESIK:
            kararlar.append("Kanıt yetersiz (MDD>%20)")
            continue
        kararlar.append("GÜÇLÜ KANIT")
    df["karar"] = kararlar
    return df


# ==================================================================
# 6. ANA AKIŞ
# ==================================================================
def main():
    print("=" * 80)
    print("H1 EDGE TEST v1.0 — FROZEN (bug fix sürümü)")
    print("E0 (10).csv KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    # -------- QUARTER-AH SETTLEMENT BİRİM TESTLERİ --------
    print("\n--- Quarter-AH settlement birim testleri ---")
    _test_ah_settlement()

    # -------- VERİ --------
    print("\n--- Veri yükleniyor ---")
    df = veri_yukle()
    print(f"  Birleşik satır: {len(df)}")
    print(f"  Sezonlar: {sorted(df['Season'].unique())}")

    # -------- TRAIN --------
    print(f"\n--- TRAIN ({TRAIN_SEZONLAR}) ---")
    train_df = hipotez_calistir(df, TRAIN_SEZONLAR)
    print(train_df[["kod", "elo", "market", "n", "roi", "perm_p", "mdd"]].to_string(index=False))

    # -------- VALIDATION --------
    print(f"\n--- VALIDATION ({VALIDATION_SEZON}) ---")
    val_df = hipotez_calistir(df, [VALIDATION_SEZON])
    val_df = fdr_uygula(val_df)
    print(val_df[["kod", "elo", "market", "n", "roi", "roi_ci_lo", "q_fdr", "mdd", "karar"]].to_string(index=False))

    guclu = val_df[val_df["karar"] == "GÜÇLÜ KANIT"]
    print(f"\n  Validation GÜÇLÜ KANIT: {len(guclu)} / 9")

    # -------- OOS --------
    oos_df = None
    if len(guclu) > 0:
        print(f"\n--- OOS ({OOS_SEZON}) — yalnızca GÜÇLÜ KANIT veren hipotezler ---")
        guclu_kodlar = guclu["kod"].tolist()
        oos_tum = hipotez_calistir(df, [OOS_SEZON])
        oos_df = oos_tum[oos_tum["kod"].isin(guclu_kodlar)].copy()
        print(oos_df[["kod", "n", "roi", "roi_ci_lo", "perm_p", "mdd"]].to_string(index=False))
    else:
        print("\n--- OOS: Validation'da GÜÇLÜ KANIT veren hipotez yok, OOS çalıştırılmadı ---")

    # -------- EXCEL --------
    print(f"\n--- {CIKTI} yazılıyor ---")
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        train_df.to_excel(w, sheet_name="Train", index=False)
        val_df.to_excel(w, sheet_name="Validation", index=False)
        guclu.to_excel(w, sheet_name="Karar_Ozeti", index=False)
        if oos_df is not None:
            oos_df.to_excel(w, sheet_name=f"OOS_{OOS_SEZON.replace('/','_')}", index=False)
        else:
            pd.DataFrame([{"Not": "Validation'da GÜÇLÜ KANIT yok"}]).to_excel(
                w, sheet_name=f"OOS_{OOS_SEZON.replace('/','_')}", index=False)
        pd.DataFrame([{
            "Protokol": "H1 EDGE TEST v1.0 (FROZEN)",
            "Uygulama": "bug fix sürümü",
            "Tarih": "2026-10-08",
            "Market_movement": "AHCh - AHh, esik ±0.25",
            "Elo_kat": "DUSUK<0, ORTA 0-100, YUKSEK>100",
            "Hipotez": 9,
            "Form5_rol": "Yalnizca tanimlayici",
            "Bahis_yon": "EV->HOME, DEP->AWAY, NOTR->yok",
            "Fiyat": "B365 acilis (executable)",
            "Settlement": "Quarter-AH split-stake",
            "Stake": 1,
            "N_min": N_MIN,
            "Bootstrap": "BCa 10.000, seed=20261008",
            "Permutation": "10.000, sonuc etiketleri, iki tarafli",
            "FDR": "BH q<0.05, 9 hipotez",
            "MDD_esik": MDD_ESIK,
            "Bankroll": BANKROLL,
            "Train": "2019/20 - 2022/23",
            "Validation": "2023/24",
            "OOS": "2024/25",
            "2026_27": "KILITLI - okunmadi",
        }]).to_excel(w, sheet_name="Notlar", index=False)

    print(f"\n{CIKTI} başarıyla yazıldı.")


if __name__ == "__main__":
    main()