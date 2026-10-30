param([Parameter(Mandatory=$true)][string]$Study)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$studyPath = (Resolve-Path -LiteralPath $Study).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$venvConfig = Get-Content -LiteralPath (Join-Path $projectRoot '.venv\pyvenv.cfg')
$pythonBase = (($venvConfig | Where-Object { $_ -like 'home = *' }) -replace '^home = ', '').Trim()
if (-not (Test-Path -LiteralPath (Join-Path $pythonBase 'Lib\encodings'))) {
    $pythonVersion = (($venvConfig | Where-Object { $_ -like 'version = *' }) -replace '^version = ', '').Trim()
    $versionParts = $pythonVersion.Split('.')
    $pythonBase = Join-Path $env:LOCALAPPDATA ("Programs\Python\Python" + $versionParts[0] + $versionParts[1])
}
if (-not (Test-Path -LiteralPath (Join-Path $pythonBase 'Lib\encodings'))) {
    throw 'The virtual environment standard library is unavailable.'
}
$env:PYTHONHOME = $pythonBase
$env:PYTHONPATH = Join-Path $projectRoot '.venv\Lib\site-packages'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
Set-Location -LiteralPath $projectRoot
$started = Get-Date
$stopAt = $started.Date.AddHours(3)
if ($stopAt -le $started) { $stopAt = $stopAt.AddDays(1) }
$sessionMinutes = [Math]::Max(1, [Math]::Ceiling(($stopAt - $started).TotalMinutes))
& $pythonExe -B -u (Join-Path $projectRoot 'scripts\run_forecast_recorder.py') --study $studyPath --duration-minutes $sessionMinutes --refresh-history
exit $LASTEXITCODE
