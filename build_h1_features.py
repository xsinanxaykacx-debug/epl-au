# -*- coding: utf-8 -*-
"""
build_h1_features.py
H1 Feature Mühendisliği — Pre-match takım gücü snapshot'ları

Amaç:
- 2015/16 – 2024/25 (3.800 maç) için her maçın başında BİLİNEBİLECEK takım gücü değişkenlerini üret.
- Look-ahead güvenliğini MAKİNEYLE KANITLA.

Düzeltmeler (bu sürüm):
1. Form pencere audit off-by-one → 1./4./6./11. maçlar.
2. Ev/deplasman GD eklendi.
3. Gol istatistikleri (GoalAvg/GoalAgainstAvg/GD) → sezonlar arası TAŞINIR (sezon=None).
   Form → sezon içinde sıfırlanır (sezon=sezon).

Kurallar:
- 2026/27 (E0 (10).csv) KESİNLİKLE OKUNMAZ.
- Bahis testi yok, ROI yok, sinyal yok, eşik yok.
- Yetersiz geçmiş → NaN. Maç atlanmaz.
- Aynı gün maçları arasında sızıntı yasak.
"""

import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

# ==================================================================
# 0. SABİTLER (KİLİTLİ)
# ==================================================================
YASAKLI_DOSYALAR = {"E0 (10).csv"}   # 2026/27

DOSYA_SEZON = {
    "E0.csv":       "2015/16",
    "E0 (1).csv":   "2016/17",
    "E0 (2).csv":   "2017/18",
    "E0 (3).csv":   "2018/19",
    "E0 (4).csv":   "2019/20",
    "E0 (5).csv":   "2020/21",
    "E0 (6).csv":   "2021/22",
    "E0 (7).csv":   "2022/23",
    "E0 (8).csv":   "2023/24",
    "E0 (9).csv":   "2024/25",
    "E0 (10).csv":  "2026/27",
}

ELO_BASLANGIC = 1500.0
ELO_K = 20.0
ELO_HA = 100.0

FORM_PENCERELERI = [3, 5, 10]
GOL_PENCERELERI = [5, 10]

CIKTI = "h1_features.csv"

AUDIT_SEED = 20261008
AUDIT_N = 100

_okuma_sayaci = defaultdict(int)


# ==================================================================
# 1. VERİ YÜKLEME
# ==================================================================
def _guvenli_oku(dosya: str) -> pd.DataFrame:
    ad = os.path.basename(dosya)
    if ad in YASAKLI_DOSYALAR:
        raise RuntimeError(f"YASAKLI DOSYA OKUNAMAZ: {ad}")
    _okuma_sayaci[ad] += 1
    return pd.read_csv(dosya, encoding="utf-8-sig")


def veri_yukle() -> pd.DataFrame:
    frames = []
    for dosya in sorted(glob.glob("E0*.csv")):
        ad = os.path.basename(dosya)
        if ad in YASAKLI_DOSYALAR:
            print(f"  [ATLANDI] {ad} — yasaklı (2026/27)")
            continue
        sezon = DOSYA_SEZON.get(ad, "?")
        df = _guvenli_oku(dosya)
        df["Season"] = sezon
        frames.append(df)
        print(f"  {ad:15s} → {sezon}  ({len(df)} satır)")

    df = pd.concat(frames, ignore_index=True)
    df["_tarih"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df = df.sort_values(["_tarih", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    df["_mac_id"] = df.index
    return df


# ==================================================================
# 2. YARDIMCI FONKSİYONLAR
# ==================================================================
def _aday_listesi(gecmis_liste, pencere, ev_mi, sezon, target_id):
    """
    sezon = None → sezonlar arası taşınır (gol istatistikleri)
    sezon = "2023/24" → yalnızca o sezon (form)
    """
    if sezon is not None:
        adaylar = [g for g in gecmis_liste if g["sezon"] == sezon]
    else:
        adaylar = gecmis_liste
    if ev_mi is not None:
        adaylar = [g for g in adaylar if g["ev_mi"] == ev_mi]

    if len(adaylar) < pencere:
        return [], False
    son = adaylar[-pencere:]
    if any(g.get("mac_id") == target_id for g in son):
        raise RuntimeError(
            f"LOOK-AHEAD İHLALİ: target {target_id} kendi feature'ında kullanıldı!"
        )
    return son, True


def _form(gecmis_liste, pencere, ev_mi, sezon, target_id):
    adaylar, yeterli = _aday_listesi(gecmis_liste, pencere, ev_mi, sezon, target_id)
    if not yeterli:
        return np.nan, []
    return float(np.mean([g["puan"] for g in adaylar])), adaylar


def _ort(gecmis_liste, pencere, alan, sezon, ev_mi, target_id):
    """sezon = None → sezonlar arası taşınır (gol istatistikleri)."""
    adaylar, yeterli = _aday_listesi(gecmis_liste, pencere, ev_mi, sezon, target_id)
    if not yeterli:
        return np.nan, []
    return float(np.mean([g[alan] for g in adaylar])), adaylar


def _fark(gecmis_liste, pencere, sezon, ev_mi, target_id):
    """sezon = None → sezonlar arası taşınır (gol istatistikleri)."""
    adaylar, yeterli = _aday_listesi(gecmis_liste, pencere, ev_mi, sezon, target_id)
    if not yeterli:
        return np.nan, []
    return float(np.mean([g["atti"] - g["yedi"] for g in adaylar])), adaylar


def _rest_days(gecmis_liste, bu_tarih):
    if not gecmis_liste:
        return np.nan, None
    son = gecmis_liste[-1]
    return float((bu_tarih - son["tarih"]).days), son


# ==================================================================
# 3. ANA HESAPLAMA
# ==================================================================
def hesapla_features(df: pd.DataFrame):
    elo = defaultdict(lambda: ELO_BASLANGIC)
    gecmis = defaultdict(list)

    kayitlar = []
    audit_log = []

    df = df.copy()
    df["_gun"] = df["_tarih"].dt.date

    for gun, grup in df.groupby("_gun", sort=True):
        gun_kayitlari = []
        gun_audit = []

        # ------- 1) GÜN N — PRE-MATCH FEATURE'LARI HESAPLA -------
        for _, r in grup.iterrows():
            ev = r["HomeTeam"]; dep = r["AwayTeam"]
            sezon = r["Season"]
            target_id = r["_mac_id"]
            target_tarih = r["_tarih"]

            kullanilan_gecmisler = []

            # --- Elo (pre) ---
            ev_elo_pre = elo[ev]; dep_elo_pre = elo[dep]
            elo_diff = ev_elo_pre - dep_elo_pre

            # --- Form genel (SEZON İÇİ — sezon=sezon) ---
            ev_form = {}; dep_form = {}
            for p in FORM_PENCERELERI:
                v, a = _form(gecmis[ev], p, None, sezon, target_id)
                ev_form[p] = v; kullanilan_gecmisler.extend(a)
                v, a = _form(gecmis[dep], p, None, sezon, target_id)
                dep_form[p] = v; kullanilan_gecmisler.extend(a)

            # --- Form ev/dep (SEZON İÇİ) ---
            ev_ev_form = {}; dep_dep_form = {}
            for p in FORM_PENCERELERI:
                v, a = _form(gecmis[ev], p, True, sezon, target_id)
                ev_ev_form[p] = v; kullanilan_gecmisler.extend(a)
                v, a = _form(gecmis[dep], p, False, sezon, target_id)
                dep_dep_form[p] = v; kullanilan_gecmisler.extend(a)

            # --- Gol genel (SEZONLAR ARASI — sezon=None) ---
            ev_ga = {}; dep_ga = {}
            ev_gya = {}; dep_gya = {}
            ev_gd = {}; dep_gd = {}
            for p in GOL_PENCERELERI:
                v, a = _ort(gecmis[ev], p, "atti", None, None, target_id)
                ev_ga[p] = v; kullanilan_gecmisler.extend(a)
                v, a = _ort(gecmis[dep], p, "atti", None, None, target_id)
                dep_ga[p] = v; kullanilan_gecmisler.extend(a)
                v, a = _ort(gecmis[ev], p, "yedi", None, None, target_id)
                ev_gya[p] = v; kullanilan_gecmisler.extend(a)
                v, a = _ort(gecmis[dep], p, "yedi", None, None, target_id)
                dep_gya[p] = v; kullanilan_gecmisler.extend(a)
                v, a = _fark(gecmis[ev], p, None, None, target_id)
                ev_gd[p] = v; kullanilan_gecmisler.extend(a)
                v, a = _fark(gecmis[dep], p, None, None, target_id)
                dep_gd[p] = v; kullanilan_gecmisler.extend(a)

            # --- Gol ev/dep (SEZONLAR ARASI) ---
            ev_ev_ga = {}; dep_dep_ga = {}
            ev_ev_gya = {}; dep_dep_gya = {}
            ev_ev_gd = {}; dep_dep_gd = {}
            for p in GOL_PENCERELERI:
                v, a = _ort(gecmis[ev], p, "atti", None, True, target_id)
                ev_ev_ga[p] = v; kullanilan_gecmisler.extend(a)
                v, a = _ort(gecmis[dep], p, "atti", None, False, target_id)
                dep_dep_ga[p] = v; kullanilan_gecmisler.extend(a)
                v, a = _ort(gecmis[ev], p, "yedi", None, True, target_id)
                ev_ev_gya[p] = v; kullanilan_gecmisler.extend(a)
                v, a = _ort(gecmis[dep], p, "yedi", None, False, target_id)
                dep_dep_gya[p] = v; kullanilan_gecmisler.extend(a)
                v, a = _fark(gecmis[ev], p, None, True, target_id)
                ev_ev_gd[p] = v; kullanilan_gecmisler.extend(a)
                v, a = _fark(gecmis[dep], p, None, False, target_id)
                dep_dep_gd[p] = v; kullanilan_gecmisler.extend(a)

            # --- Rest days ---
            ev_rest, ev_son = _rest_days(gecmis[ev], target_tarih)
            dep_rest, dep_son = _rest_days(gecmis[dep], target_tarih)
            if ev_son is not None: kullanilan_gecmisler.append(ev_son)
            if dep_son is not None: kullanilan_gecmisler.append(dep_son)

            ev_mp = len(gecmis[ev]); dep_mp = len(gecmis[dep])

            kayit = {
                "Season": sezon, "Date": r["Date"],
                "HomeTeam": ev, "AwayTeam": dep,
                "FTHG": r["FTHG"], "FTAG": r["FTAG"], "FTR": r["FTR"],
                "Home_Elo_Pre": ev_elo_pre, "Away_Elo_Pre": dep_elo_pre, "Elo_Diff": elo_diff,
                # Form genel
                **{f"Home_Form{p}_Pre": ev_form[p]  for p in FORM_PENCERELERI},
                **{f"Away_Form{p}_Pre": dep_form[p] for p in FORM_PENCERELERI},
                **{f"Form{p}_Diff":    ev_form[p] - dep_form[p] for p in FORM_PENCERELERI},
                # Form ev/dep
                **{f"Home_HomeForm{p}_Pre": ev_ev_form[p]  for p in FORM_PENCERELERI},
                **{f"Away_AwayForm{p}_Pre": dep_dep_form[p] for p in FORM_PENCERELERI},
                **{f"HomeForm{p}_Diff":    ev_ev_form[p] - dep_dep_form[p] for p in FORM_PENCERELERI},
                # Gol genel
                **{f"Home_GoalAvg{p}_Pre": ev_ga[p]  for p in GOL_PENCERELERI},
                **{f"Away_GoalAvg{p}_Pre": dep_ga[p] for p in GOL_PENCERELERI},
                **{f"Home_GoalAgainstAvg{p}_Pre": ev_gya[p]  for p in GOL_PENCERELERI},
                **{f"Away_GoalAgainstAvg{p}_Pre": dep_gya[p] for p in GOL_PENCERELERI},
                **{f"Home_GD{p}_Pre": ev_gd[p]  for p in GOL_PENCERELERI},
                **{f"Away_GD{p}_Pre": dep_gd[p] for p in GOL_PENCERELERI},
                # Gol ev/dep
                **{f"Home_HomeGoalAvg{p}_Pre":  ev_ev_ga[p]  for p in GOL_PENCERELERI},
                **{f"Away_AwayGoalAvg{p}_Pre":  dep_dep_ga[p] for p in GOL_PENCERELERI},
                **{f"Home_HomeGoalAgainstAvg{p}_Pre":  ev_ev_gya[p]  for p in GOL_PENCERELERI},
                **{f"Away_AwayGoalAgainstAvg{p}_Pre": dep_dep_gya[p] for p in GOL_PENCERELERI},
                **{f"Home_HomeGD{p}_Pre":  ev_ev_gd[p]  for p in GOL_PENCERELERI},
                **{f"Away_AwayGD{p}_Pre":  dep_dep_gd[p] for p in GOL_PENCERELERI},
                # Rest
                "Home_RestDays": ev_rest, "Away_RestDays": dep_rest,
                # Maç sayısı
                "Home_MatchesPlayed": ev_mp, "Away_MatchesPlayed": dep_mp,
                "_mac_id": target_id,
                "_target_tarih": target_tarih,
            }
            gun_kayitlari.append(kayit)

            gun_audit.append({
                "mac_id": target_id,
                "target_tarih": target_tarih,
                "kullanilan_tarihler": [g["tarih"] for g in kullanilan_gecmisler],
                "kullanilan_mac_idler": [g.get("mac_id") for g in kullanilan_gecmisler],
            })

        # ------- 2) GÜN N — SONUÇLARI İŞLE -------
        for _, r in grup.iterrows():
            ev = r["HomeTeam"]; dep = r["AwayTeam"]
            sezon = r["Season"]
            fthg = r["FTHG"]; ftag = r["FTAG"]; ftr = r["FTR"]
            tarih = r["_tarih"]; mac_id = r["_mac_id"]

            if ftr == "H": ev_puan, dep_puan = 3.0, 0.0
            elif ftr == "D": ev_puan, dep_puan = 1.0, 1.0
            else: ev_puan, dep_puan = 0.0, 3.0

            ev_elo = elo[ev]; dep_elo = elo[dep]
            E_ev = 1.0 / (1.0 + 10 ** ((dep_elo - ev_elo - ELO_HA) / 400.0))
            S_ev = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
            elo[ev] = ev_elo + ELO_K * (S_ev - E_ev)
            elo[dep] = dep_elo + ELO_K * ((1 - S_ev) - (1 - E_ev))

            gecmis[ev].append({"tarih": tarih, "sezon": sezon, "puan": ev_puan,
                               "ev_mi": True, "atti": fthg, "yedi": ftag, "mac_id": mac_id})
            gecmis[dep].append({"tarih": tarih, "sezon": sezon, "puan": dep_puan,
                                "ev_mi": False, "atti": ftag, "yedi": fthg, "mac_id": mac_id})

        kayitlar.extend(gun_kayitlari)
        audit_log.extend(gun_audit)

    return pd.DataFrame(kayitlar), audit_log


# ==================================================================
# 4. GERÇEK AUDIT
# ==================================================================
def audit_lookahead_real(audit_log, n=AUDIT_N, seed=AUDIT_SEED):
    rng = np.random.default_rng(seed)
    n_total = len(audit_log)
    if n_total == 0:
        return {"n": 0, "lookahead_violations": -1, "self_violations": -1,
                "same_day_violations": -1}

    idx = rng.choice(n_total, size=min(n, n_total), replace=False)

    lookahead_violations = 0
    self_violations = 0
    same_day_violations = 0

    for i in idx:
        kayit = audit_log[i]
        target_tarih = kayit["target_tarih"]
        target_id = kayit["mac_id"]

        for t in kayit["kullanilan_tarihler"]:
            if t is None or pd.isna(t):
                continue
            if t >= target_tarih:
                lookahead_violations += 1
                break

        if target_id in kayit["kullanilan_mac_idler"]:
            self_violations += 1

        for t in kayit["kullanilan_tarihler"]:
            if t is not None and not pd.isna(t) and t == target_tarih:
                same_day_violations += 1
                break

    return {
        "n": len(idx),
        "lookahead_violations": lookahead_violations,
        "self_violations": self_violations,
        "same_day_violations": same_day_violations,
    }


def audit_form_pencere(feat: pd.DataFrame):
    """
    Off-by-one düzeltildi:
    - 1. maç  → Form3/5/10 = NaN
    - 4. maç  → Form3 dolu
    - 6. maç  → Form5 dolu
    - 11. maç → Form10 dolu
    """
    feat = feat.copy()
    feat["_tarih"] = pd.to_datetime(feat["Date"], dayfirst=True, format="mixed", errors="coerce")

    ev_df = feat[["Season", "_tarih", "HomeTeam", "Home_Form3_Pre", "Home_Form5_Pre", "Home_Form10_Pre"]].copy()
    ev_df.columns = ["Season", "_tarih", "Takim", "Form3", "Form5", "Form10"]
    dep_df = feat[["Season", "_tarih", "AwayTeam", "Away_Form3_Pre", "Away_Form5_Pre", "Away_Form10_Pre"]].copy()
    dep_df.columns = ["Season", "_tarih", "Takim", "Form3", "Form5", "Form10"]
    tum = pd.concat([ev_df, dep_df], ignore_index=True)
    tum = tum.sort_values(["Takim", "Season", "_tarih"]).reset_index(drop=True)
    tum["mac_sira"] = tum.groupby(["Takim", "Season"]).cumcount() + 1

    m1  = tum[tum["mac_sira"] == 1]
    m4  = tum[tum["mac_sira"] == 4]
    m6  = tum[tum["mac_sira"] == 6]
    m11 = tum[tum["mac_sira"] == 11]

    return {
        "mac1_Form3_NaN":       int(m1["Form3"].isna().sum()),
        "mac1_Form3_toplam":    len(m1),
        "mac1_Form5_NaN":       int(m1["Form5"].isna().sum()),
        "mac1_Form5_toplam":    len(m1),
        "mac1_Form10_NaN":      int(m1["Form10"].isna().sum()),
        "mac1_Form10_toplam":   len(m1),
        "mac4_Form3_dolu":      int(m4["Form3"].notna().sum()),
        "mac4_Form3_toplam":    len(m4),
        "mac6_Form5_dolu":      int(m6["Form5"].notna().sum()),
        "mac6_Form5_toplam":    len(m6),
        "mac11_Form10_dolu":    int(m11["Form10"].notna().sum()),
        "mac11_Form10_toplam":  len(m11),
    }


# ==================================================================
# 5. ANA AKIŞ
# ==================================================================
def main():
    print("=" * 80)
    print("BUILD H1 FEATURES — Pre-match takım gücü snapshot")
    print("2026/27 (E0 (10).csv) KESİNLİKLE OKUNMAYACAK.")
    print("=" * 80)

    print("\n--- Veri yükleniyor ---")
    df = veri_yukle()
    print(f"  Toplam satır: {len(df)}")

    print("\n--- Feature'lar hesaplanıyor (gün gün) ---")
    feat, audit_log = hesapla_features(df)
    print(f"  Üretilen satır: {len(feat)}")

    # ==========================================================
    # AUDIT
    # ==========================================================
    print("\n" + "=" * 80)
    print("AUDIT")
    print("=" * 80)

    # A — Satır sayısı
    print("\nA — Sezon bazında satır sayısı")
    sezon_sayim = feat.groupby("Season").size()
    for s in sorted(sezon_sayim.index):
        print(f"  {s}: {sezon_sayim[s]}")
    print(f"  TOPLAM: {sezon_sayim.sum()}")

    # B — Yasaklı veri
    print("\nB — Yasaklı veri kontrolü")
    for ad in YASAKLI_DOSYALAR:
        print(f"  {ad}: READ {_okuma_sayaci[ad]} TIMES")

    # C — Gerçek look-ahead audit
    print("\nC — Look-ahead audit (kullanılan geçmiş maçların tarihleri)")
    audit = audit_lookahead_real(audit_log, n=AUDIT_N, seed=AUDIT_SEED)
    print(f"  Seçilen maç: {audit['n']}")
    print(f"  Look-ahead violations: {audit['lookahead_violations']}")
    print(f"  Self-reference violations: {audit['self_violations']}")
    print(f"  Same-day leakage: {audit['same_day_violations']}")

    # D — Form pencere audit
    print("\nD — Form pencere audit (off-by-one düzeltildi)")
    fp = audit_form_pencere(feat)
    for k, v in fp.items():
        print(f"  {k}: {v}")

    # F — İlk maç NaN kontrolü
    print("\nF — İlk maç NaN kontrolü (global ilk maç)")
    ilk = feat.iloc[0]
    print(f"  Home_Elo_Pre: {ilk['Home_Elo_Pre']}")
    print(f"  Away_Elo_Pre: {ilk['Away_Elo_Pre']}")
    print(f"  Home_Form5_Pre: {ilk['Home_Form5_Pre']}")
    print(f"  Away_Form5_Pre: {ilk['Away_Form5_Pre']}")
    print(f"  Home_RestDays: {ilk['Home_RestDays']}")
    print(f"  Away_RestDays: {ilk['Away_RestDays']}")

    # ==========================================================
    # ÇIKTI
    # ==========================================================
    feat_out = feat.drop(columns=["_mac_id", "_target_tarih"], errors="ignore")

    print(f"\n--- {CIKTI} yazılıyor ---")
    feat_out.to_csv(CIKTI, index=False, encoding="utf-8-sig")
    print(f"  {CIKTI} yazıldı.")
    print(f"  Satır: {len(feat_out)}, Sütun: {len(feat_out.columns)}")

    print("\n--- Sütun listesi ---")
    for c in feat_out.columns:
        print(f"  {c}")

    print("\n--- NaN oranı (%) — ilk 15 sütun ---")
    for c in feat_out.columns[:15]:
        nan_oran = feat_out[c].isna().sum() / len(feat_out) * 100
        print(f"  {c}: {nan_oran:.2f}")


if __name__ == "__main__":
    main()