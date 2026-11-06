param([switch]$CreateCertificate)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$authRoot = [IO.Path]::GetFullPath((Join-Path $taskRoot 'data/betfair/auth'))
if (-not $authRoot.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar)) { throw 'Invalid certificate directory' }
if (-not $CreateCertificate) { throw 'Use -CreateCertificate to prepare the local login certificate.' }
New-Item -ItemType Directory -Path $authRoot -Force | Out-Null
$certFile = Join-Path $authRoot 'client.crt'
$keyFile = Join-Path $authRoot 'client.key'
if ((Test-Path -LiteralPath $certFile) -or (Test-Path -LiteralPath $keyFile)) {
    if ((Test-Path -LiteralPath $certFile) -and (Test-Path -LiteralPath $keyFile)) {
        Write-Output "Existing certificate preserved: $certFile"
        exit 0
    }
    throw 'An incomplete certificate setup exists. Keep the existing files for review before creating another key.'
}
$rsa = [Security.Cryptography.RSA]::Create(2048)
try {
    $request = [Security.Cryptography.X509Certificates.CertificateRequest]::new(
        'CN=Horse Historical Data Client', $rsa,
        [Security.Cryptography.HashAlgorithmName]::SHA256,
        [Security.Cryptography.RSASignaturePadding]::Pkcs1)
    $oids = [Security.Cryptography.OidCollection]::new()
    $oids.Add([Security.Cryptography.Oid]::new('1.3.6.1.5.5.7.3.2')) | Out-Null
    $request.CertificateExtensions.Add([Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension]::new($oids, $false))
    $request.CertificateExtensions.Add([Security.Cryptography.X509Certificates.X509KeyUsageExtension]::new(
        [Security.Cryptography.X509Certificates.X509KeyUsageFlags]::DigitalSignature -bor
        [Security.Cryptography.X509Certificates.X509KeyUsageFlags]::KeyEncipherment, $false))
    $request.CertificateExtensions.Add([Security.Cryptography.X509Certificates.X509BasicConstraintsExtension]::new($false, $false, 0, $true))
    $certificate = $request.CreateSelfSigned([DateTimeOffset]::UtcNow.AddMinutes(-5), [DateTimeOffset]::UtcNow.AddYears(2))
    try {
        [IO.File]::WriteAllText($keyFile, $rsa.ExportRSAPrivateKeyPem())
        [IO.File]::WriteAllText($certFile, $certificate.ExportCertificatePem())
    } finally { $certificate.Dispose() }
} finally { $rsa.Dispose() }
Write-Output "Public certificate to upload to Betfair: $certFile"
Write-Output 'Keep client.key private. No Live App Key activation is required for historical downloads.'
