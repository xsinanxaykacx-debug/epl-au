
$klasor = "C:\Users\bzdye\Downloads\EPL"

# E0 (10).csv kesinlikle kapsam dışı.
$dosyalar = @(
    Get-ChildItem -LiteralPath $klasor -File -Filter "E0*.csv" |
    Where-Object {
        $_.Name -ne "E0 (10).csv" -and
        $_.Name -match '^E0(?: \(\d+\))?\.csv$'
    } |
    Sort-Object {
        if ($_.Name -eq "E0.csv") {
            0
        } else {
            [int]($_.Name -replace '^E0 \((\d+)\)\.csv$', '$1')
        }
    }
)

# Kontrol edilecek Alt/Üst sütunları.
$oranSutunlari = @(
    "B365>2.5",
    "B365<2.5",
    "Avg>2.5",
    "Avg<2.5",
    "B365C>2.5",
    "B365C<2.5",
    "AvgC>2.5",
    "AvgC<2.5"
)

$tarihFormatlari = @(
    "dd/MM/yyyy",
    "d/M/yyyy",
    "yyyy-MM-dd",
    "MM/dd/yyyy"
)

$kultur = [System.Globalization.CultureInfo]::InvariantCulture
$tarihStili = [System.Globalization.DateTimeStyles]::None

$envanter = @()
$eksikRaporu = @()

foreach ($dosya in $dosyalar) {

    Write-Host "Kontrol ediliyor: $($dosya.Name)"

    # Dosya başlığını ve CSV verilerini oku.
    $baslik = (Get-Content -LiteralPath $dosya.FullName -TotalCount 1)
    $sutunSayisi = @($baslik -split ',').Count
    $satirlar = @(Import-Csv -LiteralPath $dosya.FullName)

    $sutunlar = @()
    if ($satirlar.Count -gt 0) {
        $sutunlar = @($satirlar[0].PSObject.Properties.Name)
    }

    # Boş veri dosyası durumunda başlıktan sütunları belirle.
    if ($satirlar.Count -eq 0) {
        $sutunlar = @($baslik -split ',')
    }

    # Tarih aralığı ve hatalı tarih sayısı.
    $tarihler = @()
    $hataliTarih = 0

    foreach ($satir in $satirlar) {
        $tarihMetni = [string]$satir.Date
        $tarih = [datetime]::MinValue

        $gecerli = [datetime]::TryParseExact(
            $tarihMetni,
            $tarihFormatlari,
            $kultur,
            $tarihStili,
            [ref]$tarih
        )

        if ($gecerli) {
            $tarihler += $tarih.Date
        } else {
            $hataliTarih++
        }
    }

    $tarihIlk = ""
    $tarihSon = ""

    if ($tarihler.Count -gt 0) {
        $tarihIlk = ($tarihler | Measure-Object -Minimum).Minimum.ToString("dd/MM/yyyy")
        $tarihSon = ($tarihler | Measure-Object -Maximum).Maximum.ToString("dd/MM/yyyy")
    }

    # Div + Date + HomeTeam + AwayTeam ile yinelenen maç kontrolü.
    $macSayac = @{}
    $yinelenenGrup = 0
    $fazlaYinelenenSatir = 0

    foreach ($satir in $satirlar) {
        $tarihMetni = [string]$satir.Date
        $tarih = [datetime]::MinValue

        if ([datetime]::TryParseExact(
            $tarihMetni,
            $tarihFormatlari,
            $kultur,
            $tarihStili,
            [ref]$tarih
        )) {
            $tarihAnahtari = $tarih.ToString("yyyy-MM-dd")
        } else {
            $tarihAnahtari = $tarihMetni.Trim()
        }

        $anahtar = @(
            ([string]$satir.Div).Trim()
            $tarihAnahtari
            ([string]$satir.HomeTeam).Trim()
            ([string]$satir.AwayTeam).Trim()
        ) -join "|"

        if ($macSayac.ContainsKey($anahtar)) {
            $macSayac[$anahtar]++
        } else {
            $macSayac[$anahtar] = 1
        }
    }

    foreach ($anahtar in $macSayac.Keys) {
        if ($macSayac[$anahtar] -gt 1) {
            $yinelenenGrup++
            $fazlaYinelenenSatir += ($macSayac[$anahtar] - 1)
        }
    }

    # Genel envanter: dosya başına bir satır.
    $envanter += [pscustomobject]@{
        Dosya              = $dosya.Name
        Sutun              = $sutunSayisi
        Mac                = $satirlar.Count
        IlkTarih           = $tarihIlk
        SonTarih           = $tarihSon
        HataliTarih        = $hataliTarih
        YinelenenGrup      = $yinelenenGrup
        FazlaTekrarSatiri  = $fazlaYinelenenSatir
    }

    # Eksik oran değerleri: bulunmayan sütun "YOK" olarak gösterilir.
    $eksik = [ordered]@{ Dosya = $dosya.Name }

    foreach ($oran in $oranSutunlari) {
        if ($oran -notin $sutunlar) {
            $eksik[$oran] = "YOK"
        } else {
            $eksikAdedi = 0

            foreach ($satir in $satirlar) {
                $deger = $satir.PSObject.Properties[$oran].Value

                if ([string]::IsNullOrWhiteSpace([string]$deger)) {
                    $eksikAdedi++
                }
            }

            $eksik[$oran] = $eksikAdedi
        }
    }

    $eksikRaporu += [pscustomobject]$eksik
}

Write-Host "`n========== GENEL VERİ ENVANTERİ ==========" -ForegroundColor Cyan
$envanter | Format-Table -AutoSize

Write-Host "`n========== ALT/ÜST EKSİK DEĞER RAPORU ==========" -ForegroundColor Cyan
$eksikRaporu | Format-Table -AutoSize

Write-Host "`nKontrol tamamlandı. Hiçbir örüntü veya bahis stratejisi test edilmedi." -ForegroundColor Green
Write-Host "E0 (10).csv kapsam dışıdır."