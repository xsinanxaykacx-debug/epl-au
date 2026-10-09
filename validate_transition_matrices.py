# -*- coding: utf-8 -*-
"""
Sentetik 5x5 Markov geçiş matrisleri doğrulama betiği.

Ana ESS:
    Geyer Initial Positive Sequence (IPS)

Destekleyici ESS:
    Batch Means BM50 ve BM100
İkincil ESS:
    Batch Means BM20

İçerik:
    - Yapılandırma ve olasılık matrisi doğrulaması
    - Doğrusal ve iteratif durağan dağılım karşılaştırması
    - Bağımsız ısınma senaryoları
    - IPS ESS ve gecikme/kesme raporlaması
    - BM ESS, ham ve sınırlandırılmış değerler
    - Geçiş matrisi ki-kare tanılaması
    - Bonferroni çoklu test düzeltmesi
    - ESS yöntemleri arası ayrışma analizi
    - Ayrıntılı uyarılar ve hata kontrolleri

ÖNEMLİ:
    Bu betik sentetik Markov matrislerini inceler.
    Tek başına gerçek verilerdeki bir modelin geçerliliğini,
    tahmin başarısını veya ekonomik kârlılığını kanıtlamaz.
    Otomatik istatistiksel PASS kararı üretmez.
"""

import sys
import numpy as np


# ============================================================
# 1. AYARLAR
# ============================================================

DURUMLAR = [
    "DÜŞÜK-ÜST",
    "DÜŞÜK-ALT",
    "YÜKSEK-ÜST",
    "YÜKSEK-ALT",
    "EŞİTLİK",
]

ISINMA_TOHUMLARI = {
    1000: [42, 123, 456],
    2000: [1042, 1123, 1456],
}

ISINMA_SENARYOLARI = [1000, 2000]

HEDEF_UZUNLUK = 98000
MAX_LAG = 1000

ALPHA_AILE = 0.05

ESS_AYRISMA_ALT = 0.5
ESS_AYRISMA_UST = 2.0

MATRIS_TOL = 1e-10
DURAGAN_TOL = 1e-12
DURAGAN_MAX_ITER = 100000


P_low = np.array(
    [
        [0.25, 0.20, 0.20, 0.20, 0.15],
        [0.20, 0.25, 0.20, 0.20, 0.15],
        [0.20, 0.20, 0.25, 0.20, 0.15],
        [0.20, 0.20, 0.20, 0.25, 0.15],
        [0.20, 0.20, 0.20, 0.20, 0.20],
    ],
    dtype=float,
)

P_mid = np.array(
    [
        [0.40, 0.15, 0.20, 0.15, 0.10],
        [0.15, 0.40, 0.15, 0.20, 0.10],
        [0.20, 0.15, 0.40, 0.15, 0.10],
        [0.15, 0.20, 0.15, 0.40, 0.10],
        [0.15, 0.15, 0.15, 0.15, 0.40],
    ],
    dtype=float,
)

P_high = np.array(
    [
        [0.55, 0.10, 0.15, 0.10, 0.10],
        [0.10, 0.55, 0.10, 0.15, 0.10],
        [0.15, 0.10, 0.55, 0.10, 0.10],
        [0.10, 0.15, 0.10, 0.55, 0.10],
        [0.10, 0.10, 0.10, 0.10, 0.60],
    ],
    dtype=float,
)

MATRISLER = {
    "P_low": P_low,
    "P_mid": P_mid,
    "P_high": P_high,
}


# ============================================================
# 2. GENEL YARDIMCILAR
# ============================================================

def tam_sayi_mi(deger):
    """bool değerlerini tam sayı olarak kabul etmez."""
    return (
        isinstance(deger, (int, np.integer))
        and not isinstance(deger, (bool, np.bool_))
    )


def yazdir(deger, basamak=4, nan_metin="TANIMSIZ"):
    """Sonlu sayıları güvenli biçimde biçimlendirir."""
    try:
        if deger is None:
            return nan_metin

        sayi = float(deger)

        if not np.isfinite(sayi):
            return nan_metin

        return f"{sayi:.{basamak}f}"

    except (TypeError, ValueError, OverflowError):
        return nan_metin


def guvenli_ortalama(degerler):
    """Boş olmayan ve sonlu değerlerden ortalama hesaplar."""
    x = np.asarray(degerler, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.mean(x))


# ============================================================
# 3. YAPILANDIRMA DOĞRULAMASI
# ============================================================

def yapilandirmayi_dogrula():
    if not DURUMLAR:
        raise ValueError("En az bir durum tanımlanmalı.")

    if len(DURUMLAR) != 5:
        raise ValueError("Bu betik tam olarak beş durum bekliyor.")

    if len(set(DURUMLAR)) != len(DURUMLAR):
        raise ValueError("Durum isimlerinde tekrar var.")

    if not tam_sayi_mi(HEDEF_UZUNLUK) or HEDEF_UZUNLUK < 4:
        raise ValueError(
            "HEDEF_UZUNLUK en az 4 olan pozitif bir tam sayı olmalı."
        )

    if not tam_sayi_mi(MAX_LAG) or MAX_LAG < 1:
        raise ValueError("MAX_LAG pozitif bir tam sayı olmalı.")

    if not np.isfinite(ALPHA_AILE) or not 0 < ALPHA_AILE < 1:
        raise ValueError("ALPHA_AILE, 0 ile 1 arasında olmalı.")

    if (
        not np.isfinite(ESS_AYRISMA_ALT)
        or not np.isfinite(ESS_AYRISMA_UST)
        or ESS_AYRISMA_ALT <= 0
        or ESS_AYRISMA_UST <= ESS_AYRISMA_ALT
    ):
        raise ValueError("ESS ayrışma sınırları geçersiz.")

    if set(ISINMA_TOHUMLARI) != set(ISINMA_SENARYOLARI):
        raise ValueError(
            "Isınma senaryoları ile tohum sözlüğünün anahtarları uyuşmuyor."
        )

    if len(ISINMA_SENARYOLARI) != len(set(ISINMA_SENARYOLARI)):
        raise ValueError("Yinelenen ısınma senaryosu var.")

    for isinma in ISINMA_SENARYOLARI:
        if not tam_sayi_mi(isinma) or isinma < 0:
            raise ValueError(
                f"Geçersiz ısınma uzunluğu: {isinma}"
            )

    tum_tohumlar = []

    for isinma in ISINMA_SENARYOLARI:
        liste = ISINMA_TOHUMLARI[isinma]

        if not isinstance(liste, (list, tuple)) or len(liste) == 0:
            raise ValueError(
                f"Isınma={isinma}: tohum listesi boş veya geçersiz."
            )

        for tohum in liste:
            if not tam_sayi_mi(tohum):
                raise ValueError(
                    f"Isınma={isinma}: tohum tam sayı olmalı: {tohum}"
                )

            if not 0 <= tohum <= 2**32 - 1:
                raise ValueError(f"Geçersiz tohum: {tohum}")

            tum_tohumlar.append(int(tohum))

    if len(tum_tohumlar) != len(set(tum_tohumlar)):
        raise ValueError(
            "Isınma senaryoları arasında yinelenen temel tohum var."
        )

    tohum_sayilari = {
        len(ISINMA_TOHUMLARI[x])
        for x in ISINMA_SENARYOLARI
    }

    if len(tohum_sayilari) != 1:
        raise ValueError(
            "Her ısınma senaryosunda aynı sayıda tohum olmalı."
        )

    if not MATRISLER:
        raise ValueError("En az bir matris gerekli.")

    if len(MATRISLER) != len(set(MATRISLER)):
        raise ValueError("Matris isimlerinde tekrar var.")

    for ad, P in MATRISLER.items():
        if not isinstance(ad, str) or not ad.strip():
            raise ValueError("Matris ismi boş olamaz.")

        matris_dogrula(P, ad)

    # Etkin tohumların da benzersiz ve geçerli olduğunu doğrula.
    etkin_tohumlar = []

    for matris_no, _ in enumerate(MATRISLER.items()):
        for isinma in ISINMA_SENARYOLARI:
            for temel_tohum in ISINMA_TOHUMLARI[isinma]:
                etkin_tohum = (
                    int(temel_tohum) + matris_no * 100000
                )

                if not 0 <= etkin_tohum <= 2**32 - 1:
                    raise ValueError(
                        f"Etkin tohum sınırı aşıldı: {etkin_tohum}"
                    )

                etkin_tohumlar.append(etkin_tohum)

    if len(etkin_tohumlar) != len(set(etkin_tohumlar)):
        raise ValueError("Etkin tohumlar arasında çakışma var.")

    # Her matris × her pencere × her başlangıç satırı.
    toplam_test = (
        len(MATRISLER)
        * sum(
            len(ISINMA_TOHUMLARI[x])
            for x in ISINMA_SENARYOLARI
        )
        * len(DURUMLAR)
    )

    if toplam_test < 1:
        raise ValueError("Test ailesi boş olamaz.")

    return toplam_test


# ============================================================
# 4. DURAĞAN DAĞILIM VE MATRİS DOĞRULAMA
# ============================================================

def duragan_dagilim_dogrusal(P):
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    A = P.T - np.eye(n)
    A[-1, :] = 1.0

    b = np.zeros(n)
    b[-1] = 1.0

    return np.linalg.solve(A, b)


def duragan_dagilim_iteratif(
    P,
    tol=DURAGAN_TOL,
    max_iter=DURAGAN_MAX_ITER,
):
    P = np.asarray(P, dtype=float)

    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("İteratif tolerans pozitif ve sonlu olmalı.")

    if not tam_sayi_mi(max_iter) or max_iter < 1:
        raise ValueError("max_iter pozitif tam sayı olmalı.")

    pi = np.full(P.shape[0], 1.0 / P.shape[0])

    for _ in range(max_iter):
        yeni = pi @ P

        if np.max(np.abs(yeni - pi)) < tol:
            return yeni

        pi = yeni

    raise RuntimeError(
        "İteratif durağan dağılım yakınsamadı."
    )


def matris_dogrula(P, ad, tol=MATRIS_TOL):
    try:
        P = np.asarray(P, dtype=float)
    except (TypeError, ValueError) as hata:
        raise ValueError(
            f"{ad}: Matris sayısal değerlere dönüştürülemedi."
        ) from hata

    if P.shape != (5, 5):
        raise ValueError(f"{ad}: Matris 5x5 olmalı.")

    if not np.all(np.isfinite(P)):
        raise ValueError(
            f"{ad}: Sonlu olmayan matris değeri var."
        )

    if np.any(P < 0):
        raise ValueError(
            f"{ad}: Negatif olasılık var."
        )

    if not np.allclose(
        P.sum(axis=1),
        1.0,
        atol=tol,
        rtol=0,
    ):
        raise ValueError(
            f"{ad}: Satır toplamları 1 değil."
        )

    # Bu betik pozitif geçiş olasılıklı matrisler bekler.
    if not np.all(P > 0):
        raise ValueError(
            f"{ad}: Tüm geçiş olasılıkları pozitif olmalı."
        )

    try:
        pi = duragan_dagilim_dogrusal(P)
    except np.linalg.LinAlgError as hata:
        raise ValueError(
            f"{ad}: Doğrusal durağan dağılım çözülemedi."
        ) from hata

    if not np.all(np.isfinite(pi)):
        raise ValueError(
            f"{ad}: Durağan dağılım sonlu değil."
        )

    if np.any(pi < -tol):
        raise ValueError(
            f"{ad}: Durağan dağılımda negatif olasılık var."
        )

    if not np.isclose(
        pi.sum(),
        1.0,
        atol=tol,
        rtol=0,
    ):
        raise ValueError(
            f"{ad}: Durağan dağılım toplamı 1 değil."
        )

    if not np.allclose(
        pi @ P,
        pi,
        atol=tol,
        rtol=0,
    ):
        raise ValueError(
            f"{ad}: pi @ P = pi sağlanmıyor."
        )

    pi_iteratif = duragan_dagilim_iteratif(P)

    if not np.allclose(
        pi,
        pi_iteratif,
        atol=tol,
        rtol=0,
    ):
        raise ValueError(
            f"{ad}: Durağan dağılım yöntemleri uyuşmuyor."
        )

    return pi


# Yapılandırmayı doğrula ve çoklu test eşiğini hesapla.
TOPLAM_TEST = yapilandirmayi_dogrula()
BONFERRONI_ESIK = ALPHA_AILE / TOPLAM_TEST


# ============================================================
# 5. SİMÜLASYON
# ============================================================

def matris_hafif_dogrula(P, tol=MATRIS_TOL):
    """
    Simülasyon çağrılarında kullanılan hafif doğrulama.
    Durağan dağılım hesabı yapmaz.

    Kontroller:
        - 5x5 boyut
        - sonlu değerler
        - negatif olmayan olasılıklar
        - satır toplamlarının 1 olması

    Hatalıysa ValueError yükseltir.
    """
    try:
        P_arr = np.asarray(P, dtype=float)
    except (TypeError, ValueError) as hata:
        raise ValueError(
            "Simülasyon matrisi sayısal değerlere dönüştürülemedi."
        ) from hata

    if P_arr.shape != (5, 5):
        raise ValueError(
            "Simülasyon matrisi 5x5 olmalı."
        )

    if not np.all(np.isfinite(P_arr)):
        raise ValueError(
            "Simülasyon matrisi sonlu olmayan değer içeriyor."
        )

    if np.any(P_arr < 0):
        raise ValueError(
            "Simülasyon matrisi negatif olasılık içeriyor."
        )

    if not np.allclose(
        P_arr.sum(axis=1),
        1.0,
        atol=tol,
        rtol=0,
    ):
        raise ValueError(
            "Simülasyon matrisinin satır toplamları 1 değil."
        )

    return P_arr


def simulasyon_seri_uzun(P, tohum, toplam_adim):
    # Hafif matris doğrulaması: hatalı matriste simülasyon başlamaz.
    P = matris_hafif_dogrula(P)

    if not tam_sayi_mi(tohum):
        raise ValueError("Simülasyon tohumu tam sayı olmalı.")

    if not 0 <= tohum <= 2**32 - 1:
        raise ValueError("Simülasyon tohumu geçersiz.")

    if not tam_sayi_mi(toplam_adim) or toplam_adim < 1:
        raise ValueError(
            "toplam_adim pozitif tam sayı olmalı."
        )

    rng = np.random.default_rng(int(tohum))

    durum = 0
    seri = np.empty(int(toplam_adim), dtype=np.int8)

    for t in range(int(toplam_adim)):
        seri[t] = durum
        durum = rng.choice(5, p=P[durum])

    return seri


# ============================================================
# 6. OTOKORELASYON
# ============================================================

def otokorelasyon_dizisi(
    seri,
    max_lag=1000,
    uyari_listesi=None,
):
    def uyari(mesaj):
        if uyari_listesi is not None:
            uyari_listesi.append(mesaj)

    x = np.asarray(seri, dtype=float)
    n = len(x)

    if n < 3:
        uyari(
            "Otokorelasyon: seri 3'ten kısa; "
            "yalnızca rho(0)=1 döndürülüyor."
        )
        return np.array([1.0])

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Otokorelasyon serisi sonlu olmayan değer içeriyor."
        )

    if not tam_sayi_mi(max_lag) or max_lag < 0:
        raise ValueError(
            "Otokorelasyon max_lag negatif olmayan tam sayı olmalı."
        )

    x = x - x.mean()
    varyans = np.dot(x, x) / n

    if not np.isfinite(varyans) or varyans <= 0:
        uyari(
            "Otokorelasyon: varyans sıfır veya geçersiz; "
            "yalnızca rho(0)=1 döndürülüyor."
        )
        return np.array([1.0])

    max_lag = min(int(max_lag), n - 1)

    fft_boyutu = 1 << (2 * n - 1).bit_length()

    fx = np.fft.rfft(x, n=fft_boyutu)

    otokovaryans = np.fft.irfft(
        fx * np.conjugate(fx),
        n=fft_boyutu,
    )[:max_lag + 1]

    otokovaryans /= n

    if (
        not np.isfinite(otokovaryans[0])
        or otokovaryans[0] <= 0
    ):
        uyari(
            "Otokorelasyon: otokovaryans(0) geçersiz; "
            "yalnızca rho(0)=1 döndürülüyor."
        )
        return np.array([1.0])

    rho = otokovaryans / otokovaryans[0]

    if not np.all(np.isfinite(rho)):
        raise RuntimeError(
            "Otokorelasyon hesabı sonlu olmayan sonuç üretti."
        )

    return rho


# ============================================================
# 7. IPS ESS
# ============================================================

def etkin_orneklem_boyutu(
    seri,
    max_lag=MAX_LAG,
    uyari_listesi=None,
    kesme_bilgisi=None,
):
    """
    Geyer Initial Positive Sequence (IPS) yaklaşımı.

    Gecikme çiftleri:
        rho(0)+rho(1)
        rho(2)+rho(3)
        ...

    Alanlar:
        durum:
            negatif_ilk_cift
            negatif_cift
            max_lag_sinirinda
            tanimsiz

        efektif_max_lag:
            Otokorelasyon fonksiyonuna verilen en büyük gecikme.

        son_hesaplanan_lag:
            Gerçekten hesaplanan son otokorelasyon gecikmesi.

        son_tam_cift_bitis_lag:
            Oluşturulabilen son tam çiftin bitiş gecikmesi.

        cift_baslangic_lag:
            İlk negatif çiftin başlangıç gecikmesi;
            negatif çift yoksa None.

        negatif_cift_bitis_lag:
            İlk negatif çiftin bitiş gecikmesi (k+1);
            negatif çift yoksa None.

        cift_bitis_lag:
            IPS'e eklenen son pozitif çiftin bitiş gecikmesi;
            hiç pozitif çift yoksa None.

        kullanilan_max_lag:
            IPS'e fiilen giren son gecikme;
            pozitif çift yoksa None.

        kullanilan_max_lag_durumu:
            normal / pozitif_cift_yok / tanimsiz.
    """

    bilgi = {
        "durum": "tanimsiz",
        "cift_baslangic_lag": None,
        "negatif_cift_bitis_lag": None,
        "cift_bitis_lag": None,
        "efektif_max_lag": None,
        "son_hesaplanan_lag": None,
        "son_tam_cift_bitis_lag": None,
        "kullanilan_max_lag": None,
        "kullanilan_max_lag_durumu": "tanimsiz",
        "hesaplanan_gecikme_sayisi": 0,
        "cift_olusturulabilen": 0,
        "toplam_cift": 0,
        "kullanilan_cift": 0,
    }

    def bilgi_yaz():
        if kesme_bilgisi is not None:
            kesme_bilgisi.update(bilgi)

    def uyari_yaz(mesaj):
        if uyari_listesi is not None:
            uyari_listesi.append(mesaj)

    # Parametre doğrulaması önce yapılır.
    if (
        isinstance(max_lag, (bool, np.bool_))
        or not isinstance(max_lag, (int, np.integer))
        or max_lag < 1
    ):
        uyari_yaz(
            f"Geçersiz max_lag ({max_lag}); IPS ESS tanımsız."
        )
        bilgi_yaz()
        return np.nan

    try:
        x = np.asarray(seri, dtype=float)
    except (TypeError, ValueError):
        uyari_yaz(
            "Seri sayısal değerlere dönüştürülemedi; IPS ESS tanımsız."
        )
        bilgi_yaz()
        return np.nan

    n = len(x)

    if (
        n < 3
        or not np.all(np.isfinite(x))
        or np.var(x) == 0
    ):
        uyari_yaz(
            "Seri kısa, sabit veya sonlu olmayan değer içeriyor; "
            "IPS ESS tanımsız."
        )
        bilgi_yaz()
        return np.nan

    efektif_max_lag = min(int(max_lag), n - 2)

    rho = otokorelasyon_dizisi(
        x,
        efektif_max_lag,
    )

    cift_sayisi = len(rho) // 2
    rho_ciftleri = rho[:2 * cift_sayisi]

    bilgi["efektif_max_lag"] = efektif_max_lag
    bilgi["son_hesaplanan_lag"] = len(rho) - 1
    bilgi["hesaplanan_gecikme_sayisi"] = len(rho)
    bilgi["cift_olusturulabilen"] = cift_sayisi
    bilgi["toplam_cift"] = cift_sayisi

    if cift_sayisi > 0:
        bilgi["son_tam_cift_bitis_lag"] = 2 * cift_sayisi - 1

    ciftler = []
    durum = "max_lag_sinirinda"

    for k in range(0, len(rho_ciftleri), 2):
        cift = float(
            rho_ciftleri[k] + rho_ciftleri[k + 1]
        )

        if not np.isfinite(cift):
            durum = "tanimsiz"
            uyari_yaz(
                "IPS çift toplamı sonlu değil; ESS tanımsız."
            )
            # Negatif çift bulunamadı; başlangıç alanı None kalır.
            break

        if cift <= 0:
            # Negatif veya sıfır çiftin başlangıç ve bitiş gecikmeleri.
            bilgi["cift_baslangic_lag"] = k
            bilgi["negatif_cift_bitis_lag"] = k + 1

            durum = (
                "negatif_ilk_cift"
                if not ciftler
                else "negatif_cift"
            )
            break

        # Geyer monoton azalan pozitif çift dizisi.
        if ciftler:
            cift = min(cift, ciftler[-1])

        ciftler.append(cift)
        # IPS'e eklenen son pozitif çiftin bitiş gecikmesi.
        bilgi["cift_bitis_lag"] = k + 1

    bilgi["durum"] = durum
    bilgi["kullanilan_cift"] = len(ciftler)

    if bilgi["cift_bitis_lag"] is not None:
        bilgi["kullanilan_max_lag"] = (
            bilgi["cift_bitis_lag"]
        )
        bilgi["kullanilan_max_lag_durumu"] = "normal"
    else:
        bilgi["kullanilan_max_lag"] = None
        bilgi["kullanilan_max_lag_durumu"] = (
            "pozitif_cift_yok"
        )

    bilgi_yaz()

    if durum == "max_lag_sinirinda":
        uyari_yaz(
            "IPS mevcut tam gecikme çiftlerinin sonuna ulaştı. "
            f"efektif_max_lag={efektif_max_lag}, "
            f"son_tam_cift_bitis_lag="
            f"{bilgi['son_tam_cift_bitis_lag']}, "
            f"kullanilan_max_lag="
            f"{bilgi['kullanilan_max_lag']}. "
            "ESS tahmininin gecikme sınırına duyarlılığı ayrıca "
            "incelenmeli."
        )

    if durum == "tanimsiz":
        return np.nan

    if not ciftler:
        uyari_yaz(
            "Kullanılabilir pozitif IPS çifti yok; ESS tanımsız."
        )
        return np.nan

    tau = max(
        1.0,
        -1.0 + 2.0 * sum(ciftler),
    )

    if not np.isfinite(tau) or tau <= 0:
        uyari_yaz(
            "IPS tau değeri geçersiz; ESS tanımsız."
        )
        return np.nan

    ess = n / tau

    if not np.isfinite(ess):
        uyari_yaz(
            "IPS ESS sonlu olmayan değer üretti."
        )
        return np.nan

    return float(min(n, max(1.0, ess)))


# ============================================================
# 8. BATCH MEANS ESS
# ============================================================

def batch_means_ess(
    seri,
    batch_sayisi,
    uyari_listesi=None,
):
    sonuc = {
        "ham": np.nan,
        "kirpilmis": np.nan,
        "kirpildi": None,
        "kullanilan_n": 0,
        "batch_sayisi": batch_sayisi,
        "batch_uzunluk": 0,
    }

    def uyari(mesaj):
        if uyari_listesi is not None:
            uyari_listesi.append(mesaj)

    if (
        isinstance(batch_sayisi, (bool, np.bool_))
        or not isinstance(batch_sayisi, (int, np.integer))
        or batch_sayisi < 2
    ):
        uyari(
            f"BM{batch_sayisi}: batch_sayisi en az 2 olmalı."
        )
        return sonuc

    try:
        x = np.asarray(seri, dtype=float)
    except (TypeError, ValueError):
        uyari(
            f"BM{batch_sayisi}: seri sayısal değil."
        )
        return sonuc

    n = len(x)

    if n < 4:
        uyari(
            f"BM{batch_sayisi}: yetersiz veri."
        )
        return sonuc

    if not np.all(np.isfinite(x)):
        uyari(
            f"BM{batch_sayisi}: sonlu olmayan değer."
        )
        return sonuc

    if np.var(x, ddof=1) == 0:
        uyari(
            f"BM{batch_sayisi}: seri sabit."
        )
        return sonuc

    batch_uzunluk = n // batch_sayisi

    if batch_uzunluk < 2:
        uyari(
            f"BM{batch_sayisi}: batch uzunluğu 2'den küçük."
        )
        return sonuc

    kullanilan_n = batch_uzunluk * batch_sayisi
    x_kullanilan = x[:kullanilan_n]

    batch_ortalamalar = x_kullanilan.reshape(
        batch_sayisi,
        batch_uzunluk,
    ).mean(axis=1)

    var_seri = np.var(
        x_kullanilan,
        ddof=1,
    )

    var_batch = np.var(
        batch_ortalamalar,
        ddof=1,
    )

    sonuc["kullanilan_n"] = kullanilan_n
    sonuc["batch_uzunluk"] = batch_uzunluk

    if not np.isfinite(var_batch) or var_batch <= 0:
        uyari(
            f"BM{batch_sayisi}: batch varyansı geçersiz."
        )
        return sonuc

    # B batch ortalaması için:
    # ESS = B * Var(X) / Var(batch ortalamaları)
    ess_ham = (
        batch_sayisi * var_seri / var_batch
    )

    if not np.isfinite(ess_ham) or ess_ham <= 0:
        uyari(
            f"BM{batch_sayisi}: ham ESS geçersiz."
        )
        return sonuc

    ess_kirpilmis = min(
        float(kullanilan_n),
        max(1.0, float(ess_ham)),
    )

    return {
        "ham": float(ess_ham),
        "kirpilmis": float(ess_kirpilmis),
        "kirpildi": bool(ess_ham > kullanilan_n),
        "kullanilan_n": kullanilan_n,
        "batch_sayisi": int(batch_sayisi),
        "batch_uzunluk": batch_uzunluk,
    }


# ============================================================
# 9. STANDART HATA VE Z TANILAMASI
# ============================================================

def guvenli_z(frekans, pi_teorik, ess):
    if (
        not np.isfinite(ess)
        or ess <= 0
        or not np.isfinite(frekans)
        or not np.isfinite(pi_teorik)
        or not 0 < pi_teorik < 1
    ):
        return np.nan

    se = np.sqrt(
        pi_teorik * (1.0 - pi_teorik) / ess
    )

    if not np.isfinite(se) or se <= 0:
        return np.nan

    return float(
        abs(frekans - pi_teorik) / se
    )


# ============================================================
# 10. GEÇİŞ MATRİSİ Kİ-KARE TANILAMASI
# ============================================================

def ki_kare_p_degeri_df4(ki_kare):
    """
    Serbestlik derecesi 4 olan ki-kare dağılımının
    sağ kuyruk olasılığı:

        P(X >= x) = exp(-x/2) * (1 + x/2)
    """
    if not np.isfinite(ki_kare) or ki_kare < 0:
        return np.nan

    x = float(ki_kare)

    return float(
        np.exp(-x / 2.0) * (1.0 + x / 2.0)
    )


def gecis_matrisi_dogrula(seri, P, ad):
    seri = np.asarray(seri)

    if seri.ndim != 1:
        raise ValueError(
            f"{ad}: Geçiş testi serisi tek boyutlu olmalı."
        )

    if len(seri) < 2:
        raise ValueError(
            f"{ad}: Geçiş testi için en az iki durum gerekli."
        )

    if not np.issubdtype(seri.dtype, np.integer):
        if (
            not np.all(np.isfinite(seri))
            or not np.all(seri == np.floor(seri))
        ):
            raise ValueError(
                f"{ad}: Seri durumları tam sayı olmalı."
            )

    seri = seri.astype(int)

    if np.any(seri < 0) or np.any(seri >= 5):
        raise ValueError(
            f"{ad}: Seri 0-4 aralığı dışında durum içeriyor."
        )

    P = np.asarray(P, dtype=float)

    if P.shape != (5, 5):
        raise ValueError(
            f"{ad}: Geçiş matrisi 5x5 olmalı."
        )

    gecis_sayaci = np.zeros(
        (5, 5),
        dtype=np.int64,
    )

    baslangic_sayaci = np.zeros(
        5,
        dtype=np.int64,
    )

    for t in range(len(seri) - 1):
        i = int(seri[t])
        j = int(seri[t + 1])

        gecis_sayaci[i, j] += 1
        baslangic_sayaci[i] += 1

    gozlenen = np.full(
        (5, 5),
        np.nan,
        dtype=float,
    )

    sonuclar = []

    for i in range(5):
        ni = int(baslangic_sayaci[i])

        if ni == 0:
            sonuclar.append(
                {
                    "satir": i,
                    "durum": "gozlem_yok",
                    "ki_kare": np.nan,
                    "df": 4,
                    "p_deger": np.nan,
                    "bonferroni_anlamli": None,
                    "min_beklenen": np.nan,
                    "uyari": (
                        "Başlangıç durumu gözlenmedi."
                    ),
                }
            )
            continue

        gozlenen[i] = (
            gecis_sayaci[i] / ni
        )

        beklenen = ni * P[i]
        goz = gecis_sayaci[i].astype(float)

        min_beklenen = float(
            np.min(beklenen)
        )

        if (
            not np.all(np.isfinite(beklenen))
            or np.any(beklenen <= 0)
        ):
            ki2 = np.nan
            p = np.nan
            anlamli = None

            uyari = (
                "Beklenen hücre sayısı sıfır, negatif "
                "veya sonlu değil."
            )

        else:
            ki2 = float(
                np.sum(
                    (goz - beklenen) ** 2 / beklenen
                )
            )

            p = ki_kare_p_degeri_df4(ki2)

            anlamli = (
                bool(p < BONFERRONI_ESIK)
                if np.isfinite(p)
                else None
            )

            uyari = ""

            if min_beklenen < 5:
                uyari = (
                    f"Minimum beklenen hücre="
                    f"{min_beklenen:.3f}; "
                    "ki-kare yaklaşımı güvenilir "
                    "olmayabilir."
                )

        sonuclar.append(
            {
                "satir": i,
                "durum": "ok",
                "ki_kare": ki2,
                "df": 4,
                "p_deger": p,
                "bonferroni_anlamli": anlamli,
                "min_beklenen": min_beklenen,
                "uyari": uyari,
            }
        )

    fark = np.abs(gozlenen - P)

    sonlu_farklar = fark[np.isfinite(fark)]

    max_fark = (
        float(np.max(sonlu_farklar))
        if len(sonlu_farklar)
        else np.nan
    )

    return {
        "ad": ad,
        "gozlenen": gozlenen,
        "teorik": P,
        "fark": fark,
        "max_fark": max_fark,
        "baslangic_sayaci": baslangic_sayaci,
        "gecis_sayaci": gecis_sayaci,
        "ki_kare": sonuclar,
        "toplam_test": TOPLAM_TEST,
        "bonferroni_esik": BONFERRONI_ESIK,
        "not": (
            "Ki-kare sonuçları tanısaldır. Her başlangıç "
            "satırında df=4 kullanılır. Bonferroni, bütün "
            "yapılandırmalardaki satır testlerine uygulanır. "
            "Bu düzeltme testler arası bağımsızlık gerektirmez; "
            "ancak test istatistiğinin asimptotik yaklaşımının "
            "uygunluğunu tek başına garanti etmez. Markov "
            "geçişlerinin bağımlılığı ve beklenen hücre "
            "büyüklükleri dikkate alınmalıdır. Durum frekansı "
            "z değerleri bu test ailesine dahil değildir."
        ),
    }


# ============================================================
# 11. RAPORLAMA YARDIMCILARI
# ============================================================

def bm_yazdir(bm):
    if not bm:
        return "TANIMSIZ"

    ham = bm.get("ham", np.nan)
    kirpilmis = bm.get("kirpilmis", np.nan)

    if not np.isfinite(ham):
        return "TANIMSIZ"

    if bm.get("kirpildi"):
        return (
            f"{ham:.0f}->{kirpilmis:.0f}(K)"
        )

    return f"{ham:.0f}"


def sonuc_bul(tum_sonuclar, ad, tohum, isinma):
    for s in tum_sonuclar:
        if (
            s["ad"] == ad
            and s["tohum"] == tohum
            and s["isinma"] == isinma
        ):
            return s

    raise KeyError(
        "İstenen simülasyon sonucu bulunamadı."
    )


# ============================================================
# 12. TEK PENCERE ANALİZİ
# ============================================================

def analiz(P, ad, tohum, isinma, seri_uzun):
    P = np.asarray(P, dtype=float)

    pi = duragan_dagilim_dogrusal(P)

    baslangic = int(isinma)
    bitis = baslangic + int(HEDEF_UZUNLUK)

    seri = seri_uzun[baslangic:bitis]

    if len(seri) != HEDEF_UZUNLUK:
        raise ValueError(
            f"{ad}, tohum={tohum}, ısınma={isinma}: "
            f"beklenen {HEDEF_UZUNLUK}, bulunan {len(seri)}."
        )

    frekans = np.array(
        [
            np.mean(seri == d)
            for d in range(5)
        ],
        dtype=float,
    )

    ess_ips = []

    ess_bm = {
        20: [],
        50: [],
        100: [],
    }

    se_listesi = []
    z_listesi = []
    kesme_bilgileri = []
    uyarilar = []

    for d in range(5):
        ikili = (
            seri == d
        ).astype(float)

        durum_uyarilari = []
        kesme = {}

        ess = etkin_orneklem_boyutu(
            ikili,
            MAX_LAG,
            durum_uyarilari,
            kesme,
        )

        ess_ips.append(ess)
        kesme_bilgileri.append(kesme)

        for batch in (20, 50, 100):
            bm = batch_means_ess(
                ikili,
                batch,
                durum_uyarilari,
            )

            ess_bm[batch].append(bm)

        p = float(pi[d])

        se = (
            float(
                np.sqrt(
                    p * (1.0 - p) / ess
                )
            )
            if np.isfinite(ess) and ess > 0
            else np.nan
        )

        se_listesi.append(se)

        z_listesi.append(
            guvenli_z(
                frekans[d],
                p,
                ess,
            )
        )

        for mesaj in durum_uyarilari:
            uyarilar.append(
                f"{DURUMLAR[d]}: {mesaj}"
            )

    gecis = gecis_matrisi_dogrula(
        seri,
        P,
        ad,
    )

    return {
        "ad": ad,
        "tohum": tohum,
        "isinma": isinma,
        "pi_teorik": pi,
        "frekans": frekans,
        "ess": ess_ips,
        "ess_bm_20": ess_bm[20],
        "ess_bm_50": ess_bm[50],
        "ess_bm_100": ess_bm[100],
        "se": se_listesi,
        "z": z_listesi,
        "uyarilar": uyarilar,
        "kesme": kesme_bilgileri,
        "gecis": gecis,
    }


# ============================================================
# 13. ANA PROGRAM
# ============================================================

def main():
    print("=" * 100)
    print("AŞAMA 1: MATRİS VE YAPILANDIRMA DOĞRULAMASI")
    print("=" * 100)

    print(f"Matris sayısı: {len(MATRISLER)}")
    print(f"Durum sayısı: {len(DURUMLAR)}")
    print(f"Isınma senaryoları: {ISINMA_SENARYOLARI}")
    print(f"Hedef pencere: {HEDEF_UZUNLUK:,}")
    print(f"IPS MAX_LAG: {MAX_LAG}")
    print(f"Toplam geçiş testi: {TOPLAM_TEST}")
    print(f"Bonferroni eşiği: {BONFERRONI_ESIK:.10g}")

    for ad, P in MATRISLER.items():
        pi = matris_dogrula(P, ad)

        print(
            f"OK | {ad} | "
            f"durağan dağılım={np.round(pi, 6)}"
        )

    print("\n" + "=" * 100)
    print("AŞAMA 2: SİMÜLASYON VE ESS")
    print("=" * 100)

    print(
        "IPS ana; BM50/BM100 destekleyici; BM20 ikincil."
    )
    print(
        "Her matris ve ısınma/tohum koşulu ayrı bir "
        "simülasyon akışı kullanır."
    )

    tum_sonuclar = []
    matris_adlari = list(MATRISLER.keys())

    for matris_no, (ad, P) in enumerate(
        MATRISLER.items()
    ):
        print("\n" + "=" * 100)
        print(f"MATRİS: {ad}")
        print("=" * 100)

        print(
            "Teorik durağan dağılım:",
            np.round(
                duragan_dagilim_dogrusal(P),
                6,
            ),
        )

        for isinma in ISINMA_SENARYOLARI:
            for temel_tohum in ISINMA_TOHUMLARI[isinma]:
                efektif_tohum = int(
                    temel_tohum + matris_no * 100000
                )

                seri = simulasyon_seri_uzun(
                    P,
                    efektif_tohum,
                    isinma + HEDEF_UZUNLUK,
                )

                s = analiz(
                    P,
                    ad,
                    efektif_tohum,
                    isinma,
                    seri,
                )

                tum_sonuclar.append(s)

                print(
                    f"\nTohum={efektif_tohum} | "
                    f"Isınma={isinma} | "
                    f"N={HEDEF_UZUNLUK:,}"
                )

                print(
                    f"{'Durum':<12} "
                    f"{'Frekans':>9} "
                    f"{'Teorik':>9} "
                    f"{'IPS':>10} "
                    f"{'BM50':>12} "
                    f"{'BM100':>12} "
                    f"{'BM20':>12} "
                    f"{'SE':>10} "
                    f"{'z':>9}"
                )

                for d in range(5):
                    print(
                        f"{DURUMLAR[d]:<12} "
                        f"{s['frekans'][d]:>9.5f} "
                        f"{s['pi_teorik'][d]:>9.5f} "
                        f"{yazdir(s['ess'][d], 0):>10} "
                        f"{bm_yazdir(s['ess_bm_50'][d]):>12} "
                        f"{bm_yazdir(s['ess_bm_100'][d]):>12} "
                        f"{bm_yazdir(s['ess_bm_20'][d]):>12} "
                        f"{yazdir(s['se'][d], 6):>10} "
                        f"{yazdir(s['z'][d], 3):>9}"
                    )

                g = s["gecis"]

                print("\nGEÇİŞ MATRİSİ TESTLERİ")

                for k in g["ki_kare"]:
                    isim = DURUMLAR[k["satir"]]

                    if k["durum"] == "gozlem_yok":
                        print(
                            f"  {isim}: GÖZLEM YOK"
                        )
                        continue

                    if k["bonferroni_anlamli"] is None:
                        etiket = "BELİRSİZ"
                    elif k["bonferroni_anlamli"]:
                        etiket = "ANLAMLI SAPMA"
                    else:
                        etiket = "ANLAMLI SAPMA YOK"

                    print(
                        f"  {isim}: "
                        f"ki²={yazdir(k['ki_kare'], 4)}, "
                        f"df={k['df']}, "
                        f"p={yazdir(k['p_deger'], 8)}, "
                        f"min_beklenen="
                        f"{yazdir(k['min_beklenen'], 2)}, "
                        f"{etiket}"
                    )

                    if k["uyari"]:
                        print(
                            f"    UYARI: {k['uyari']}"
                        )

                if s["uyarilar"]:
                    print("ESS UYARILARI:")

                    for mesaj in s["uyarilar"]:
                        print(f"  - {mesaj}")

    if not tum_sonuclar:
        raise RuntimeError(
            "Hiçbir simülasyon sonucu üretilemedi."
        )

    # --------------------------------------------------------
    # AŞAMA 3: BM ESS KIRPMA ÖZETİ
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("AŞAMA 3: BM ESS KIRPMA ÖZETİ")
    print("=" * 100)

    kirpma = {
        20: 0,
        50: 0,
        100: 0,
    }

    gecerli = {
        20: 0,
        50: 0,
        100: 0,
    }

    for s in tum_sonuclar:
        for d in range(5):
            for b in (20, 50, 100):
                bm = s[f"ess_bm_{b}"][d]

                if bm["kirpildi"] is not None:
                    gecerli[b] += 1
                    kirpma[b] += int(
                        bm["kirpildi"]
                    )

    for b in (50, 100, 20):
        if gecerli[b]:
            oran = (
                100.0 * kirpma[b] / gecerli[b]
            )

            print(
                f"BM{b}: {kirpma[b]}/{gecerli[b]} "
                f"kırpıldı (%{oran:.2f})"
            )
        else:
            print(
                f"BM{b}: geçerli hesap yok."
            )

    # --------------------------------------------------------
    # AŞAMA 4: ISINMA KARŞILAŞTIRMASI
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("AŞAMA 4: ISINMA SENARYOSU KARŞILAŞTIRMASI")
    print("=" * 100)

    print(
        "NOT: Senaryolar ayrı tohum grupları kullanır."
    )
    print(
        "Frekans farkları betimleyicidir; ısınmanın "
        "yeterliliğini tek başına kanıtlamaz."
    )

    for ad, P in MATRISLER.items():
        pi = duragan_dagilim_dogrusal(P)

        print(f"\n{ad}:")

        matris_no = matris_adlari.index(ad)

        for isinma in ISINMA_SENARYOLARI:
            farklar = []

            for temel_tohum in ISINMA_TOHUMLARI[isinma]:
                tohum = (
                    temel_tohum + matris_no * 100000
                )

                s = sonuc_bul(
                    tum_sonuclar,
                    ad,
                    tohum,
                    isinma,
                )

                farklar.append(
                    float(
                        np.max(
                            np.abs(
                                s["frekans"] - pi
                            )
                        )
                    )
                )

            print(
                f"  Isınma={isinma}: "
                f"farklar="
                f"{[round(x, 7) for x in farklar]}, "
                f"ortalama={np.mean(farklar):.8f}, "
                f"maksimum={np.max(farklar):.8f}"
            )

    # --------------------------------------------------------
    # AŞAMA 5: ESS YÖNTEMLERİ ARASI AYRIŞMA
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("AŞAMA 5: ESS YÖNTEMLERİ ARASI AYRIŞMA")
    print("=" * 100)

    print(
        f"Ayrışma sınırı: oran < {ESS_AYRISMA_ALT} "
        f"veya > {ESS_AYRISMA_UST}"
    )

    for b in (50, 100, 20):
        ham_n = 0
        ham_ayrisma = 0

        kirp_n = 0
        kirp_ayrisma = 0

        for s in tum_sonuclar:
            for d in range(5):
                ips = s["ess"][d]
                bm = s[f"ess_bm_{b}"][d]

                if (
                    not np.isfinite(ips)
                    or ips <= 0
                ):
                    continue

                for alan in ("ham", "kirpilmis"):
                    deger = bm.get(alan, np.nan)

                    if (
                        not np.isfinite(deger)
                        or deger <= 0
                    ):
                        continue

                    oran = ips / deger

                    ayrisiyor = (
                        oran < ESS_AYRISMA_ALT
                        or oran > ESS_AYRISMA_UST
                    )

                    if alan == "ham":
                        ham_n += 1
                        ham_ayrisma += int(
                            ayrisiyor
                        )
                    else:
                        kirp_n += 1
                        kirp_ayrisma += int(
                            ayrisiyor
                        )

        if ham_n:
            print(
                f"IPS vs BM{b} ham: "
                f"{ham_ayrisma}/{ham_n} "
                f"(%{100 * ham_ayrisma / ham_n:.2f})"
            )
        else:
            print(
                f"IPS vs BM{b} ham: karşılaştırma yok"
            )

        if kirp_n:
            print(
                f"IPS vs BM{b} kırpılmış: "
                f"{kirp_ayrisma}/{kirp_n} "
                f"(%{100 * kirp_ayrisma / kirp_n:.2f})"
            )
        else:
            print(
                f"IPS vs BM{b} kırpılmış: karşılaştırma yok"
            )

    # --------------------------------------------------------
    # AŞAMA 6: IPS KESME VE GECİKME RAPORU
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("AŞAMA 6: IPS KESME VE GECİKME RAPORU")
    print("=" * 100)

    print("Alan açıklamaları:")
    print("  durum                    : Kesme nedeni")
    print("  efektif_max_lag          : İstenen/izin verilen üst sınır")
    print("  son_hesaplanan_lag       : Hesaplanan son otokorelasyon gecikmesi")
    print("  son_tam_cift_bitis_lag   : Oluşturulabilen son tam çiftin bitişi")
    print("  cift_baslangic_lag       : İlk negatif çiftin başlangıcı")
    print("  negatif_cift_bitis_lag   : İlk negatif çiftin bitişi")
    print("  cift_bitis_lag           : IPS'e giren son pozitif çiftin bitişi")
    print("  kullanilan_max_lag       : IPS'e fiilen giren son gecikme")
    print("  kullanilan_max_lag_durumu: Veri/kullanım durumu")

    durumlar = {
        "negatif_cift": 0,
        "negatif_ilk_cift": 0,
        "max_lag_sinirinda": 0,
        "tanimsiz": 0,
    }

    gecikmeler = {
        "negatif_cift": [],
        "negatif_ilk_cift": [],
        "max_lag_sinirinda": [],
    }

    for s in tum_sonuclar:
        for kb in s["kesme"]:
            durum = kb.get(
                "durum",
                "tanimsiz",
            )

            if durum not in durumlar:
                durum = "tanimsiz"

            durumlar[durum] += 1

            son = kb.get(
                "cift_bitis_lag"
            )

            if (
                son is not None
                and durum in gecikmeler
            ):
                gecikmeler[durum].append(
                    son
                )

    print("\nKESME DURUMLARI:")

    for durum, sayi in durumlar.items():
        print(f"  {durum}: {sayi}")

    print(
        "\nIPS'E GİREN SON POZİTİF ÇİFTİN BİTİŞ GECİKMESİ:"
    )

    for durum, liste in gecikmeler.items():
        print(f"\n  {durum}:")

        if not liste:
            print(
                "    Son gecikme verisi yok."
            )
            continue

        arr = np.asarray(
            liste,
            dtype=float,
        )

        print(
            f"    n={len(arr)}, "
            f"min={arr.min():.0f}, "
            f"Q1={np.percentile(arr, 25):.1f}, "
            f"medyan={np.median(arr):.1f}, "
            f"ortalama={arr.mean():.2f}, "
            f"Q3={np.percentile(arr, 75):.1f}, "
            f"max={arr.max():.0f}"
        )

    veri_durumlari = {}

    for s in tum_sonuclar:
        for kb in s["kesme"]:
            vd = kb.get(
                "kullanilan_max_lag_durumu",
                "tanimsiz",
            )

            veri_durumlari[vd] = (
                veri_durumlari.get(vd, 0) + 1
            )

    print("\nVERİ MEVCUDİYETİ:")

    for vd, sayi in sorted(veri_durumlari.items()):
        print(f"  {vd}: {sayi}")

    print("\nGECİKME ALANLARI ÖRNEĞİ:")

    ornek_kb = None

    for s in tum_sonuclar:
        for kb in s["kesme"]:
            if kb.get("durum") in (
                "negatif_cift",
                "negatif_ilk_cift",
                "max_lag_sinirinda",
            ):
                ornek_kb = kb
                break

        if ornek_kb is not None:
            break

    if ornek_kb is not None:
        for alan in (
            "efektif_max_lag",
            "son_hesaplanan_lag",
            "son_tam_cift_bitis_lag",
            "kullanilan_max_lag",
            "hesaplanan_gecikme_sayisi",
            "cift_olusturulabilen",
            "kullanilan_cift",
            "cift_baslangic_lag",
            "negatif_cift_bitis_lag",
            "cift_bitis_lag",
            "durum",
            "kullanilan_max_lag_durumu",
        ):
            print(
                f"  {alan:<30} = "
                f"{ornek_kb.get(alan)}"
            )
    else:
        print(
            "  Örnek kayıt bulunamadı."
        )

    # --------------------------------------------------------
    # AŞAMA 7: GEÇİŞ TESTİ UYARI ÖZETİ
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("AŞAMA 7: GEÇİŞ TESTİ UYARI ÖZETİ")
    print("=" * 100)

    kucuk_beklenen = 0
    gozlem_yok = 0
    anlamli_sapma = 0
    tanimsiz_test = 0
    gecerli_test = 0

    for s in tum_sonuclar:
        for k in s["gecis"]["ki_kare"]:
            if k["durum"] == "gozlem_yok":
                gozlem_yok += 1
                continue

            if (
                np.isfinite(k["min_beklenen"])
                and k["min_beklenen"] < 5
            ):
                kucuk_beklenen += 1

            if k["bonferroni_anlamli"] is None:
                tanimsiz_test += 1
            else:
                gecerli_test += 1

                anlamli_sapma += int(
                    k["bonferroni_anlamli"]
                )

    print(
        f"Küçük beklenen hücre uyarısı: "
        f"{kucuk_beklenen}"
    )

    print(
        f"Gözlem olmayan satır: {gozlem_yok}"
    )

    print(
        f"Tanımsız test: {tanimsiz_test}"
    )

    print(
        f"Geçerli test: {gecerli_test}"
    )

    print(
        f"Bonferroni sonrası anlamlı sapma: "
        f"{anlamli_sapma}"
    )

    print(
        "NOT: Anlamlı sapma bulunmaması, matrisin bütün "
        "özelliklerinin doğrulandığı anlamına gelmez."
    )

    print(
        "NOT: Satır başına ki-kare testi asimptotik bir "
        "tanılamadır. Bonferroni düzeltmesi, çoklu "
        "karşılaştırmada ailevi hata oranını kontrol eder; "
        "ancak Markov bağımlılığı nedeniyle test "
        "istatistiğinin ki-kare(4) dağılımına yakınsamasını "
        "tek başına garanti etmez. Bu nedenle bu testler "
        "kesin doğrulama değil, tanısal kontrol olarak "
        "yorumlanmalıdır. Kesin doğrulama için bağımsız "
        "veri, kör OOS veya tam Bayesçi model "
        "karşılaştırması gerekir."
    )

    # --------------------------------------------------------
    # AŞAMA 8: ESS UYARI ÖZETİ
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("AŞAMA 8: ESS UYARI ÖZETİ")
    print("=" * 100)

    uyari_sayisi = 0

    for s in tum_sonuclar:
        if s["uyarilar"]:
            uyari_sayisi += len(
                s["uyarilar"]
            )

            print(
                f"\n{s['ad']} | "
                f"tohum={s['tohum']} | "
                f"ısınma={s['isinma']}"
            )

            for mesaj in s["uyarilar"]:
                print(f"  - {mesaj}")

    if uyari_sayisi == 0:
        print(
            "ESS fonksiyonları çalışma uyarısı üretmedi."
        )
    else:
        print(
            f"Toplam ESS uyarısı: {uyari_sayisi}"
        )

    # --------------------------------------------------------
    # SON ÖZET
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("SON ÖZET")
    print("=" * 100)

    print(
        f"Analiz penceresi: {len(tum_sonuclar)}"
    )

    print(
        f"Gözlem/pencere: {HEDEF_UZUNLUK:,}"
    )

    print(
        f"Geçiş testi ailesi: {TOPLAM_TEST}"
    )

    print(
        f"Bonferroni eşiği: {BONFERRONI_ESIK:.10g}"
    )

    print(
        f"ESS uyarısı: {uyari_sayisi}"
    )

    print(
        f"Anlamlı geçiş satırı sapması: {anlamli_sapma}"
    )

    print(
        "KARAR: OTOMATİK İSTATİSTİKSEL PASS VERİLMEDİ."
    )

    print(
        "Bu çıktı yalnızca sentetik matrisler için "
        "tanısal simülasyon kanıtıdır."
    )

    print("=" * 100)


# ============================================================
# 14. ÇALIŞTIRMA VE HATA YÖNETİMİ
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except (
        ValueError,
        RuntimeError,
        KeyError,
        IndexError,
        TypeError,
        OverflowError,
        np.linalg.LinAlgError,
        FloatingPointError,
    ) as hata:
        print(
            f"\nHATA: {hata}",
            file=sys.stderr,
        )
        raise