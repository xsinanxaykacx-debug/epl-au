
# -*- coding: utf-8 -*-
"""IPS ESS doğrulama ve entegrasyon testleri.

Bu dosya validate_transition_matrices.py dosyasını import eder;
ana betiğin içeriğini değiştirmez ve ana doğrulama akışını çalıştırmaz.

Çalıştırma:
    python test_ips.py

Deterministik birim testleri PASS/FAIL üretir. AR(1), alternating ve
Markov senaryoları ayrıca tanı çıktısı verir; bunlar tek başlarına
istatistiksel doğruluk kanıtı değildir.
"""

import unittest
from unittest.mock import patch

import numpy as np
import validate_transition_matrices as vtm


# ---------------------------------------------------------------------
# KONTROLLÜ IPS TEST VERİSİ
# ---------------------------------------------------------------------

RHO_SENTETIK = np.array(
    [1.00, 0.60, 0.40, 0.20, 0.10, 0.05, -0.02, -0.01],
    dtype=float,
)


# ---------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------------------

def sayi_yaz(deger, basamak=4):
    """Sayısal değerleri güvenli biçimde metne dönüştürür."""
    if deger is None:
        return "None"

    try:
        sayi = float(deger)
    except (TypeError, ValueError):
        return str(deger)

    if not np.isfinite(sayi):
        return "NaN/sonsuz"

    return f"{sayi:.{basamak}f}"


def yaz_bilgi(bilgi):
    """IPS kesme tanılarını standart sırayla yazdırır."""
    alanlar = (
        "durum",
        "efektif_max_lag",
        "son_hesaplanan_lag",
        "son_tam_cift_bitis_lag",
        "cift_baslangic_lag",
        "negatif_cift_bitis_lag",
        "cift_bitis_lag",
        "kullanilan_max_lag",
        "kullanilan_max_lag_durumu",
        "hesaplanan_gecikme_sayisi",
        "cift_olusturulabilen",
        "toplam_cift",
        "kullanilan_cift",
    )

    for alan in alanlar:
        print(f"  {alan:<30} = {bilgi.get(alan)}")


def uyari_yaz(uyarilar):
    """Uyarı listesini yazdırır."""
    for uyari in uyarilar:
        print(f"  Uyarı: {uyari}")


# ---------------------------------------------------------------------
# 1. IPS KESME MANTIĞI: DETERMINİSTİK BİRİM TESTLERİ
# ---------------------------------------------------------------------

class TestIPSKesmeMantigi(unittest.TestCase):
    """Yapay otokorelasyon dizisiyle IPS kesme mantığını sınar."""

    def test_kontrollu_rho_kesme_ve_ess(self):
        """Negatif çiftte doğru lag, çift sayısı ve ESS hesaplanmalı."""
        n = 1000

        # Sabit seri kullanılmamalı; fonksiyon sıfır varyansta
        # erken dönebileceği için değişken bir seri kullanıyoruz.
        seri = np.arange(n, dtype=float)

        uyarilar = []
        bilgi = {}

        def sahte_otokorelasyon(
            seri_gelen,
            max_lag=1000,
            uyari_listesi=None,
        ):
            """Gerçek ACF yerine kontrollü rho dizisini döndürür."""
            uzunluk = min(
                int(max_lag) + 1,
                len(RHO_SENTETIK),
            )
            return RHO_SENTETIK[:uzunluk].copy()

        with patch.object(
            vtm,
            "otokorelasyon_dizisi",
            side_effect=sahte_otokorelasyon,
        ):
            ess = vtm.etkin_orneklem_boyutu(
                seri,
                max_lag=len(RHO_SENTETIK) - 1,
                uyari_listesi=uyarilar,
                kesme_bilgisi=bilgi,
            )

        # IPS çiftleri:
        # Gamma_0 = rho[0] + rho[1] = 1.60
        # Gamma_1 = rho[2] + rho[3] = 0.60
        # Gamma_2 = rho[4] + rho[5] = 0.15
        # Gamma_3 = rho[6] + rho[7] = -0.03
        #
        # İlk negatif çift Gamma_3 olduğundan bu çift hesaba katılmaz.
        # Tau = -1 + 2 * (1.60 + 0.60 + 0.15) = 3.70
        tau_beklenen = -1.0 + 2.0 * (1.60 + 0.60 + 0.15)
        ess_beklenen = n / tau_beklenen

        # Sayısal sonuçlar.
        self.assertAlmostEqual(
            tau_beklenen,
            3.70,
            places=12,
        )
        self.assertAlmostEqual(
            ess,
            ess_beklenen,
            places=10,
        )

        # Kesme durumu ve negatif çiftin konumu.
        self.assertEqual(
            bilgi.get("durum"),
            "negatif_cift",
        )
        self.assertEqual(
            bilgi.get("cift_baslangic_lag"),
            6,
        )
        self.assertEqual(
            bilgi.get("negatif_cift_bitis_lag"),
            7,
        )

        # Kullanılan son pozitif çiftin bitiş lag'i.
        self.assertEqual(
            bilgi.get("cift_bitis_lag"),
            5,
        )
        self.assertEqual(
            bilgi.get("kullanilan_max_lag"),
            5,
        )
        self.assertEqual(
            bilgi.get("kullanilan_cift"),
            3,
        )

        # Sekiz rho değeri dört tam çift oluşturur.
        self.assertEqual(
            bilgi.get("son_tam_cift_bitis_lag"),
            7,
        )

        # Negatif çift bulunduğu için normal kesme beklenir.
        self.assertEqual(
            bilgi.get("kullanilan_max_lag_durumu"),
            "normal",
        )

        print("\n[DETAY] Kontrollü rho senaryosu")
        print(f"  Tau beklenen = {tau_beklenen:.6f}")
        print(f"  ESS beklenen = {ess_beklenen:.6f}")
        print(f"  ESS gözlenen = {ess:.6f}")

        yaz_bilgi(bilgi)

        if uyarilar:
            print("  Uyarılar:")
            uyari_yaz(uyarilar)

    def test_sabit_seri_nan_ve_tanimsiz_durum(self):
        """Sabit seride ESS tanımsız olmalı ve uyarı üretilmeli."""
        seri = np.full(1000, 0.5, dtype=float)

        uyarilar = []
        bilgi = {}

        ess = vtm.etkin_orneklem_boyutu(
            seri,
            max_lag=50,
            uyari_listesi=uyarilar,
            kesme_bilgisi=bilgi,
        )

        self.assertTrue(
            np.isnan(ess),
            "Sabit serinin ESS değeri NaN olmalı.",
        )
        self.assertEqual(
            bilgi.get("durum"),
            "tanimsiz",
        )
        self.assertGreaterEqual(
            len(uyarilar),
            1,
            "Sabit seri için en az bir uyarı bekleniyor.",
        )

        print("\n[DETAY] Sabit seri")
        print(f"  ESS = {sayi_yaz(ess)}")

        yaz_bilgi(bilgi)
        uyari_yaz(uyarilar)


# ---------------------------------------------------------------------
# 2. GERÇEK SERİLERDE ENTEGRASYON VE DAYANIKLILIK TESTLERİ
# ---------------------------------------------------------------------

class TestIPSEntegrasyonTanilari(unittest.TestCase):
    """Gerçek serilerde fonksiyonun çalışmasını ve ESS aralığını sınar."""

    def test_ar1_ess_gecerli_aralikta(self):
        """AR(1) serisinde ESS sonlu ve geçerli aralıkta olmalı."""
        phi = 0.7
        n = 20000

        rng = np.random.default_rng(20240501)
        seri = np.zeros(n, dtype=float)

        for t in range(1, n):
            seri[t] = (
                phi * seri[t - 1]
                + rng.normal()
            )

        uyarilar = []
        bilgi = {}

        ess = vtm.etkin_orneklem_boyutu(
            seri,
            max_lag=vtm.MAX_LAG,
            uyari_listesi=uyarilar,
            kesme_bilgisi=bilgi,
        )

        self.assertTrue(
            np.isfinite(ess),
            "AR(1) ESS sonlu olmalı.",
        )
        self.assertGreaterEqual(
            ess,
            1.0,
        )
        self.assertLessEqual(
            ess,
            float(n),
        )

        # AR(1) teorik entegre otokorelasyon süresi:
        # tau = (1 + phi) / (1 - phi)
        tau_teorik = (1.0 + phi) / (1.0 - phi)
        ess_teorik = n / tau_teorik
        tau_gozlenen = n / ess if ess > 0 else np.nan

        print("\n[DETAY] AR(1), phi=0.7")
        print("  Not: Bu senaryo tanısaldır; sabit yüzde 5 kabul eşiği yok.")
        print(f"  Tau teorik   = {tau_teorik:.6f}")
        print(f"  ESS teorik   = {ess_teorik:.3f}")
        print(f"  ESS gözlenen = {ess:.3f}")
        print(f"  Tau n/ESS    = {tau_gozlenen:.6f}")
        print(f"  IPS durumu   = {bilgi.get('durum')}")

        yaz_bilgi(bilgi)
        uyari_yaz(uyarilar)

    def test_alternating_seri_coker_degil(self):
        """Alternating seride hesaplama istisnasız tamamlanmalı."""
        n = 4000
        seri = np.tile(
            np.array([0.0, 1.0]),
            n // 2,
        )

        uyarilar = []
        bilgi = {}

        ess = vtm.etkin_orneklem_boyutu(
            seri,
            max_lag=50,
            uyari_listesi=uyarilar,
            kesme_bilgisi=bilgi,
        )

        # Uç durumda ESS NaN veya sonlu olabilir.
        # Testin amacı istisnasız tamamlanması ve tanının oluşmasıdır.
        self.assertTrue(
            np.isnan(ess) or np.isfinite(ess),
            "ESS NaN veya sonlu bir sayı olmalı.",
        )

        if np.isfinite(ess):
            self.assertGreaterEqual(
                ess,
                1.0,
            )
            self.assertLessEqual(
                ess,
                float(n),
            )

        print("\n[DETAY] Alternating seri [0,1,...]")
        print(f"  ESS          = {sayi_yaz(ess)}")
        print(f"  IPS durumu   = {bilgi.get('durum')}")
        print(
            "  Not: cift <= 0 kontrolü toleranssızdır; "
            "bu senaryo tanısaldır."
        )

        yaz_bilgi(bilgi)
        uyari_yaz(uyarilar)

    def test_markov_zinciri_ess_gecerli_aralikta(self):
        """Sentetik Markov zincirinde ESS geçerli aralıkta olmalı."""
        p = vtm.P_low
        n = 20000
        tohum = 314159

        seri = np.asarray(
            vtm.simulasyon_seri_uzun(
                p,
                tohum,
                n,
            )
        )

        # Hedef durum 0 için gösterge serisi:
        # hedef durumdaysa 1, değilse 0.
        ikili = (seri == 0).astype(float)

        uyarilar = []
        bilgi = {}

        ess = vtm.etkin_orneklem_boyutu(
            ikili,
            max_lag=vtm.MAX_LAG,
            uyari_listesi=uyarilar,
            kesme_bilgisi=bilgi,
        )

        self.assertTrue(
            np.isfinite(ess),
            "Markov ESS sonlu olmalı.",
        )
        self.assertGreaterEqual(
            ess,
            1.0,
        )
        self.assertLessEqual(
            ess,
            float(n),
        )

        print("\n[DETAY] Markov zinciri entegrasyonu")
        print(f"  n            = {n}")
        print("  Hedef durum  = 0")
        print(f"  ESS          = {ess:.3f}")
        print(f"  Tau n/ESS    = {n / ess:.6f}")

        yaz_bilgi(bilgi)
        uyari_yaz(uyarilar)


# ---------------------------------------------------------------------
# 3. TESTLERİ ÇALIŞTIR
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("IPS ESS TESTLERİ")
    print("Deterministik birim testleri ve gerçek seri entegrasyon tanıları")
    print("=" * 78)

    unittest.main(verbosity=2)

