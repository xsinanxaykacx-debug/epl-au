# -*- coding: utf-8 -*-
"""
test_invariant.py

validate_transition_matrices.py içindeki IPS ve BM ESS fonksiyonlarının
matematiksel değişmezlerini (invariant) bağımsız olarak sınar.

Bu dosya ana betiği DEĞİŞTİRMEZ; yalnızca import eder.

KAPSAM:
    - IPS: negatif çift başlangıç/bitiş ilişkisi, kullanılan gecikme,
      son tam çift bitişi.
    - BM ESS: ham / kirpilmis / kirpildi alanlarının tutarlılığı,
      her iki kırpma dalının kontrollü girdilerle kesin sınanması.
    - Uç durumlar: sabit, kısa, alternating seriler.

SINIR:
    Bu testlerin geçmesi, ana betikteki 90 pencerenin tamamının
    doğrulandığı anlamına GELMEZ. Yalnızca fonksiyonların
    değişmezleri sağladığını gösterir.

NOT:
    IPS testlerinde kullanılan rho dizileri YAPAYDIR.
    Gerçek bir otokorelasyon fonksiyonunda rho(0) = 1 olmak
    zorundadır; ancak burada amaç yalnızca kesme mantığını izole
    etmektir.
"""

import unittest
from unittest.mock import patch

import numpy as np

import validate_transition_matrices as vtm


# ============================================================
# YARDIMCILAR
# ============================================================

def sahte_otokorelasyon_uret(rho_dizisi):
    """
    Verilen rho dizisini döndüren sahte bir otokorelasyon
    fonksiyonu üretir. max_lag'e saygı gösterir.

    UYARI: Bu fonksiyon yalnızca test amaçlıdır. Döndürülen
    rho dizisi gerçek bir otokorelasyon tahmini değildir.
    """
    rho_arr = np.asarray(rho_dizisi, dtype=float)

    def _sahte(seri, max_lag=1000, uyari_listesi=None):
        istenen = min(int(max_lag) + 1, len(rho_arr))
        return rho_arr[:istenen].copy()

    return _sahte


def yapay_seri_uret(n=1000):
    """Sabit olmayan, varyansı sıfırdan büyük yapay seri."""
    return np.arange(n, dtype=float)


# ============================================================
# IPS DEĞİŞMEZLERİ
# ============================================================

class TestIPSInvariant(unittest.TestCase):

    RHO_NEGATIF_UCUNCU = np.array(
        [1.00, 0.60, 0.40, 0.20, 0.10, 0.05, -0.02, -0.01],
        dtype=float,
    )
    # YAPAY dizi. Γ_0=1.60, Γ_1=0.60, Γ_2=0.15, Γ_3=-0.03.
    # Beklenen: cift_baslangic_lag=6, negatif_cift_bitis_lag=7,
    #           cift_bitis_lag=5, kullanilan_max_lag=5,
    #           kullanilan_cift=3, cift_sayisi=4,
    #           son_tam_cift_bitis_lag=7, tau=3.70, ESS=1000/3.70.

    RHO_NEGATIF_ILK = np.array(
        [-0.50, 0.50, 0.40, 0.20, 0.10, 0.05, 0.02, 0.01],
        dtype=float,
    )
    # YAPAY dizi. Γ_0 = -0.50 + 0.50 = 0.00 <= 0
    # → negatif_ilk_cift dalı.
    # UYARI: Gerçek otokorelasyonda rho(0)=1'dir; bu dizi yalnızca
    # dalı tetiklemek için kurgulanmıştır.

    def _calistir(self, rho_dizisi, n_seri=1000):
        seri = yapay_seri_uret(n_seri)
        uyarilar = []
        bilgi = {}
        sahte = sahte_otokorelasyon_uret(rho_dizisi)

        with patch.object(
            vtm, "otokorelasyon_dizisi", side_effect=sahte
        ):
            ess = vtm.etkin_orneklem_boyutu(
                seri,
                max_lag=len(rho_dizisi) - 1,
                uyari_listesi=uyarilar,
                kesme_bilgisi=bilgi,
            )
        return ess, bilgi, uyarilar

    # --------------------------------------------------------
    # IPS testleri
    # --------------------------------------------------------

    def test_negatif_cift_baslangic_bitis_iliskisi(self):
        """Γ_3 negatif → durum=negatif_cift, bas=6, bit=7."""
        _, bilgi, _ = self._calistir(self.RHO_NEGATIF_UCUNCU)

        self.assertEqual(bilgi.get("durum"), "negatif_cift")
        self.assertEqual(bilgi.get("cift_baslangic_lag"), 6)
        self.assertEqual(bilgi.get("negatif_cift_bitis_lag"), 7)
        self.assertEqual(
            bilgi.get("negatif_cift_bitis_lag"),
            bilgi.get("cift_baslangic_lag") + 1,
        )

    def test_negatif_ilk_cift_dali(self):
        """
        YAPAY girdi: Γ_0 = 0.00 <= 0 → negatif_ilk_cift.
        Beklenen: bas=0, bit=1, kullanilan_cift=0, ess=NaN.
        """
        ess, bilgi, _ = self._calistir(self.RHO_NEGATIF_ILK)

        self.assertEqual(bilgi.get("durum"), "negatif_ilk_cift")
        self.assertEqual(bilgi.get("cift_baslangic_lag"), 0)
        self.assertEqual(bilgi.get("negatif_cift_bitis_lag"), 1)
        self.assertEqual(bilgi.get("kullanilan_cift"), 0)
        self.assertTrue(np.isnan(ess))

    def test_kullanilan_max_lag_bagimsiz_beklenen_deger(self):
        """
        cift_bitis_lag ve kullanilan_max_lag'ın HER BİRİ bağımsız
        olarak beklenen değere (5) eşit olmalı.
        """
        _, bilgi, _ = self._calistir(self.RHO_NEGATIF_UCUNCU)

        self.assertEqual(bilgi.get("cift_bitis_lag"), 5)
        self.assertEqual(bilgi.get("kullanilan_max_lag"), 5)
        self.assertEqual(bilgi.get("kullanilan_cift"), 3)
        self.assertEqual(
            bilgi.get("kullanilan_max_lag_durumu"), "normal"
        )

    def test_cift_bitis_lag_tek_sayi(self):
        """Kabul edilen çiftler (2k, 2k+1) olduğundan bitiş tek."""
        _, bilgi, _ = self._calistir(self.RHO_NEGATIF_UCUNCU)
        bit = bilgi.get("cift_bitis_lag")
        self.assertIsNotNone(bit)
        self.assertEqual(bit % 2, 1)

    def test_son_tam_cift_bitis_lag_formulu(self):
        """son_tam_cift_bitis_lag = 2*cift_sayisi - 1 = 7."""
        _, bilgi, _ = self._calistir(self.RHO_NEGATIF_UCUNCU)

        self.assertEqual(bilgi.get("cift_olusturulabilen"), 4)
        self.assertEqual(bilgi.get("son_tam_cift_bitis_lag"), 7)
        self.assertEqual(
            bilgi.get("son_tam_cift_bitis_lag"),
            2 * bilgi.get("cift_olusturulabilen") - 1,
        )

    def test_kullanilan_max_lag_pozitif_cift_yoksa_none(self):
        """
        Γ_0 <= 0 olduğunda kod ilk çiftte break eder; sonraki
        çiftler OKUNMAZ. Beklenen: kullanilan_max_lag=None,
        kullanilan_max_lag_durumu='pozitif_cift_yok'.
        """
        _, bilgi, _ = self._calistir(self.RHO_NEGATIF_ILK)

        self.assertIsNone(bilgi.get("kullanilan_max_lag"))
        self.assertEqual(
            bilgi.get("kullanilan_max_lag_durumu"),
            "pozitif_cift_yok",
        )
        self.assertIsNone(bilgi.get("cift_bitis_lag"))

    def test_tau_ve_ess_elle_hesapla_uyumlu(self):
        """tau = 3.70, ESS = 1000 / 3.70 ≈ 270.270270."""
        ess, _, _ = self._calistir(
            self.RHO_NEGATIF_UCUNCU, n_seri=1000
        )

        tau_beklenen = -1.0 + 2.0 * (1.60 + 0.60 + 0.15)
        ess_beklenen = 1000 / tau_beklenen

        self.assertAlmostEqual(tau_beklenen, 3.70, places=12)
        self.assertAlmostEqual(ess, ess_beklenen, places=10)

    def test_sabit_seri_tanimsiz(self):
        seri = np.full(1000, 0.5, dtype=float)
        uyarilar = []
        bilgi = {}
        ess = vtm.etkin_orneklem_boyutu(
            seri, max_lag=50,
            uyari_listesi=uyarilar, kesme_bilgisi=bilgi,
        )
        self.assertTrue(np.isnan(ess))
        self.assertEqual(bilgi.get("durum"), "tanimsiz")
        self.assertGreaterEqual(len(uyarilar), 1)

    def test_kisa_seri_tanimsiz(self):
        seri = np.array([0.0, 1.0], dtype=float)
        uyarilar = []
        bilgi = {}
        ess = vtm.etkin_orneklem_boyutu(
            seri, max_lag=50,
            uyari_listesi=uyarilar, kesme_bilgisi=bilgi,
        )
        self.assertTrue(np.isnan(ess))
        self.assertEqual(bilgi.get("durum"), "tanimsiz")


# ============================================================
# BM ESS DEĞİŞMEZLERİ
# ============================================================

class TestBMInvariant(unittest.TestCase):

    def _normal_seri(self, n=10000, tohum=42):
        rng = np.random.default_rng(tohum)
        return rng.normal(size=n)

    def _kirpilacak_seri(self, n=10000, batch_sayisi=100):
        """
        Her batch içinde varyansı yüksek, batch ortalamaları
        arasında varyansı çok düşük bir seri üretir.

        Mantık:
            temel  = [-1, +1, -1, +1, ...]  (batch_uzunluk kadar)
            Her batch'e çok küçük bir ofset eklenir.
            - Batch içi ortalama ≈ ofset (±1 birbirini götürür)
            - Batch ortalamaları arasındaki fark = ofset farkı
              (çok küçük)
            - Serinin kendi varyansı ≈ 1 (büyük)

        Böylece:
            Var(X)         ≈ 1        (büyük)
            Var(batch ort) ≈ çok küçük
            ess_ham = B * Var(X) / Var(batch ort) → çok büyük
            → kirpildi = True kesin.

        n / batch_sayisi tam sayı ve çift olmalıdır.
        """
        batch_uzunluk = n // batch_sayisi

        if n % batch_sayisi != 0 or batch_uzunluk % 2 != 0:
            raise ValueError(
                "Batch uzunluğu pozitif ve çift olmalı."
            )

        temel = np.tile(
            np.array([-1.0, 1.0]),
            batch_uzunluk // 2,
        )
        ofsetler = np.linspace(
            -1e-4, 1e-4, batch_sayisi
        )

        return (
            temel[np.newaxis, :] + ofsetler[:, np.newaxis]
        ).reshape(-1)

    def _seri_varyanslari(self, seri, batch_sayisi):
        """
        batch_means_ess'in içeride hesapladığı var_seri ve
        var_batch değerlerini test amaçlı yeniden hesaplar.
        Yalnızca hata mesajlarını zenginleştirmek için
        kullanılır; assertion kaynağı DEĞİLDİR.

        Geçersiz girdilerde ValueError yükseltir.
        """
        x = np.asarray(seri, dtype=float)
        n = len(x)

        if (
            n <= 0
            or isinstance(batch_sayisi, (bool, np.bool_))
            or not isinstance(batch_sayisi, (int, np.integer))
            or batch_sayisi <= 0
        ):
            raise ValueError(
                "n pozitif olmalı; batch_sayisi pozitif "
                "tam sayı olmalı."
            )

        if n % batch_sayisi != 0:
            raise ValueError(
                "n, batch_sayisi'na tam bölünmeli."
            )

        batch_uzunluk = n // batch_sayisi

        if batch_uzunluk < 2 or batch_uzunluk % 2 != 0:
            raise ValueError(
                "batch uzunluğu en az 2 ve çift sayı olmalı."
            )

        kullanilan_n = batch_uzunluk * batch_sayisi
        x_k = x[:kullanilan_n]

        var_seri = float(np.var(x_k, ddof=1))

        batch_ort = x_k.reshape(
            batch_sayisi, batch_uzunluk
        ).mean(axis=1)
        var_batch = float(np.var(batch_ort, ddof=1))

        return var_seri, var_batch, kullanilan_n

    # --------------------------------------------------------
    # Her iki kırpma dalını KESİN tetikleyen testler
    # --------------------------------------------------------

    def test_kirpildi_true_dali_kesin(self):
        """
        Kontrollü seriyle kırpma dalının çalışmasını doğrular.
        Hedef dal çalışmazsa test FAIL verir.
        """
        n = 10000

        for b in (20, 50, 100):
            with self.subTest(batch_sayisi=b):
                seri = self._kirpilacak_seri(
                    n=n, batch_sayisi=b
                )
                bm = vtm.batch_means_ess(seri, b)

                var_seri, var_batch, kul_n = (
                    self._seri_varyanslari(seri, b)
                )

                ek = (
                    f"var_seri={var_seri:.6g}, "
                    f"var_batch={var_batch:.6g}, "
                    f"ham={bm['ham']}, "
                    f"kullanilan_n={bm['kullanilan_n']}, "
                    f"kirpildi={bm['kirpildi']}"
                )

                self.assertIsNotNone(
                    bm["kirpildi"],
                    f"BM{b}: geçerli hesap bekleniyordu; {ek}"
                )
                self.assertIs(
                    bm["kirpildi"], True,
                    f"BM{b}: True bekleniyordu; {ek}"
                )
                self.assertGreater(
                    bm["ham"], bm["kullanilan_n"],
                    f"BM{b}: ham > kullanilan_n bekleniyordu; {ek}"
                )
                self.assertEqual(
                    bm["kirpilmis"],
                    float(bm["kullanilan_n"]),
                    f"BM{b}: kirpilmis == kullanilan_n "
                    f"bekleniyordu; {ek}"
                )

    def test_kirpildi_false_dali_kesin(self):
        """
        Artan doğrusal seriyle kırpılmayan dalı doğrular.
        Hedef dal çalışmazsa test FAIL verir.
        """
        n = 10000
        seri = np.arange(n, dtype=float)

        for b in (20, 50, 100):
            with self.subTest(batch_sayisi=b):
                bm = vtm.batch_means_ess(seri, b)

                var_seri, var_batch, kul_n = (
                    self._seri_varyanslari(seri, b)
                )

                ek = (
                    f"var_seri={var_seri:.6g}, "
                    f"var_batch={var_batch:.6g}, "
                    f"ham={bm['ham']}, "
                    f"kullanilan_n={bm['kullanilan_n']}, "
                    f"kirpildi={bm['kirpildi']}"
                )

                self.assertIsNotNone(
                    bm["kirpildi"],
                    f"BM{b}: geçerli hesap bekleniyordu; {ek}"
                )
                self.assertIs(
                    bm["kirpildi"], False,
                    f"BM{b}: False bekleniyordu; {ek}"
                )
                self.assertLessEqual(
                    bm["ham"], bm["kullanilan_n"],
                    f"BM{b}: ham <= kullanilan_n bekleniyordu; {ek}"
                )
                self.assertEqual(
                    bm["kirpilmis"],
                    max(1.0, bm["ham"]),
                    f"BM{b}: kirpilmis == max(1, ham) "
                    f"bekleniyordu; {ek}"
                )

    # --------------------------------------------------------
    # Genel tutarlılık testleri
    # --------------------------------------------------------

    def test_ham_kirpilmis_iliskisi_genel(self):
        """
        Farklı serilerde: ham ile kirpilmis arasındaki ilişki
        kirpildi alanıyla tutarlı olmalı.
        """
        seriler = [
            self._normal_seri(n=10000, tohum=42),
            self._kirpilacak_seri(n=10000, batch_sayisi=50),
            np.arange(10000, dtype=float),
        ]

        for idx, seri in enumerate(seriler):
            for b in (20, 50, 100):
                with self.subTest(seri=idx, batch_sayisi=b):
                    bm = vtm.batch_means_ess(seri, b)

                    if bm["kirpildi"] is None:
                        self.assertTrue(np.isnan(bm["ham"]))
                        self.assertTrue(
                            np.isnan(bm["kirpilmis"])
                        )
                        continue

                    if bm["kirpildi"] is True:
                        self.assertGreater(
                            bm["ham"], bm["kullanilan_n"]
                        )
                        self.assertEqual(
                            bm["kirpilmis"],
                            float(bm["kullanilan_n"]),
                        )
                    else:
                        self.assertLessEqual(
                            bm["ham"], bm["kullanilan_n"]
                        )
                        self.assertEqual(
                            bm["kirpilmis"],
                            max(1.0, bm["ham"]),
                        )

    def test_kirpilmis_alt_ust_sinir(self):
        seri = self._normal_seri(n=10000, tohum=7)
        for b in (20, 50, 100):
            bm = vtm.batch_means_ess(seri, b)
            if bm["kirpildi"] is None:
                continue
            self.assertGreater(bm["kirpilmis"], 0)
            self.assertLessEqual(
                bm["kirpilmis"], bm["kullanilan_n"]
            )

    # --------------------------------------------------------
    # Uç durumlar
    # --------------------------------------------------------

    def test_sabit_seri_gecersiz(self):
        seri = np.full(1000, 0.5, dtype=float)
        uyarilar = []
        bm = vtm.batch_means_ess(seri, 50, uyarilar)

        self.assertIsNone(bm["kirpildi"])
        self.assertTrue(np.isnan(bm["ham"]))
        self.assertTrue(np.isnan(bm["kirpilmis"]))
        self.assertGreaterEqual(len(uyarilar), 1)

    def test_kisa_seri_gecersiz(self):
        seri = np.array([0.0, 1.0], dtype=float)
        uyarilar = []
        bm = vtm.batch_means_ess(seri, 50, uyarilar)

        self.assertIsNone(bm["kirpildi"])
        self.assertTrue(np.isnan(bm["ham"]))
        self.assertGreaterEqual(len(uyarilar), 1)

    def test_gecersiz_batch_sayisi(self):
        seri = self._normal_seri(n=1000, tohum=1)
        uyarilar = []
        for b in (0, 1, -5):
            bm = vtm.batch_means_ess(seri, b, uyarilar)
            self.assertIsNone(bm["kirpildi"])
            self.assertTrue(np.isnan(bm["ham"]))

    def test_kullanilan_n_tam_batch_kati(self):
        n = 1000
        seri = self._normal_seri(n=n, tohum=11)
        for b in (20, 50, 100):
            bm = vtm.batch_means_ess(seri, b)
            beklenen = (n // b) * b
            self.assertEqual(bm["kullanilan_n"], beklenen)

    # --------------------------------------------------------
    # Alternating seri — geçersiz batch varyansı dalı
    # --------------------------------------------------------

    def test_alternating_seride_invariantlar(self):
        """
        n=4000, b∈{20,50,100} için batch uzunluğu çift.
        0,1,0,1,... serisinin her batch ortalaması tam 0.5,
        dolayısıyla Var(batch ort) = 0.

        Beklenen:
            kirpildi is None,
            ham is NaN,
            kirpilmis is NaN,
            en az bir uyarı üretilir.

        Bu test, geçersiz hesaplama dalını AÇIKÇA doğrular.
        """
        n = 4000
        seri = np.tile(np.array([0.0, 1.0]), n // 2)

        for b in (20, 50, 100):
            with self.subTest(batch_sayisi=b):
                uyarilar = []
                bm = vtm.batch_means_ess(seri, b, uyarilar)

                var_seri, var_batch, kul_n = (
                    self._seri_varyanslari(seri, b)
                )

                ek = (
                    f"var_seri={var_seri:.6g}, "
                    f"var_batch={var_batch:.6g}, "
                    f"kullanilan_n={kul_n}"
                )

                self.assertIsNone(
                    bm["kirpildi"],
                    f"BM{b}: kirpildi None bekleniyordu; {ek}"
                )
                self.assertTrue(
                    np.isnan(bm["ham"]),
                    f"BM{b}: ham NaN bekleniyordu; {ek}"
                )
                self.assertTrue(
                    np.isnan(bm["kirpilmis"]),
                    f"BM{b}: kirpilmis NaN bekleniyordu; {ek}"
                )
                self.assertGreaterEqual(
                    len(uyarilar), 1,
                    f"BM{b}: en az bir uyarı bekleniyordu; {ek}"
                )


if __name__ == "__main__":
    print("=" * 78)
    print("INVARIANT TESTLERİ — IPS ve BM ESS")
    print("Bu dosya yalnızca fonksiyon değişmezlerini sınar.")
    print("Ana betiğin 90 pencerelik doğrulaması DEĞİLDİR.")
    print("=" * 78)
    unittest.main(verbosity=2)