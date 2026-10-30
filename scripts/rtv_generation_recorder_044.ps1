param([Parameter(Mandatory=$true)][ValidateSet('pause','resume')][string]$Phase)
$ErrorActionPreference = 'Stop'
$rtvRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$rtvInstall = Join-Path $rtvRoot 'data\audit\rtv_repair_features_044\installation'
$rtvPlan = Get-Content -LiteralPath (Join-Path $rtvRoot 'data\audit\rtv_repair_features_044\refit\plan.json') -Raw | ConvertFrom-Json
$rtvPointer = Join-Path $rtvRoot 'data\forecasts\recorder.json'
$rtvSettings = Get-Content -LiteralPath $rtvPointer -Raw | ConvertFrom-Json
$rtvTaskName = 'HorseForecastRecorder_038'
$rtvTask = Get-ScheduledTask -TaskName $rtvTaskName
$rtvPausePath = Join-Path $rtvInstall 'recorder_paused.json'

function Get-RtvTriggers($Task) {
    return ($Task.Triggers | Select-Object Id,StartBoundary,Enabled,@{Name='Type';Expression={$_.CimClass.CimClassName}} | ConvertTo-Json -Depth 4 -Compress)
}

if ($Phase -eq 'pause') {
    if (Test-Path -LiteralPath $rtvPausePath) { throw 'The pause receipt already exists.' }
    if (-not (Test-Path -LiteralPath (Join-Path $rtvInstall 'prepared.json'))) { throw 'Prepare the coherent installation first.' }
    if ($rtvSettings.model_id -ne $rtvPlan.parent_run) { throw 'The recorder belongs to another model.' }
    if ($rtvTask.Actions.Count -ne 1 -or $rtvTask.Actions[0].Arguments.IndexOf([string]$rtvSettings.study, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        throw 'The scheduled action differs from the current pinned study.'
    }
    Export-ScheduledTask -TaskName $rtvTaskName | Set-Content -LiteralPath (Join-Path $rtvInstall 'previous_task.xml') -Encoding utf8
    Copy-Item -LiteralPath $rtvPointer -Destination (Join-Path $rtvInstall 'previous_recorder.json')
    $rtvWasEnabled = [bool]$rtvTask.Settings.Enabled
    $rtvTriggers = Get-RtvTriggers $rtvTask
    Disable-ScheduledTask -TaskName $rtvTaskName | Out-Null
    Stop-ScheduledTask -TaskName $rtvTaskName
    $env:PYTHONHOME = 'C:\Users\Leonp\AppData\Local\Programs\Python\Python314'
    $env:PYTHONPATH = Join-Path $rtvRoot '.venv\Lib\site-packages'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & (Join-Path $rtvRoot '.venv\Scripts\python.exe') -B (Join-Path $PSScriptRoot 'install_rtv_generation_044.py') stop-recorder
    if ($LASTEXITCODE -ne 0) { throw 'The exact old recorder child could not be stopped.' }
    $rtvReceipt = [ordered]@{paused=$true; model_id=$rtvSettings.model_id; study=$rtvSettings.study; was_enabled=$rtvWasEnabled; triggers=$rtvTriggers; paused_at=(Get-Date).ToUniversalTime().ToString('o')}
    [IO.File]::WriteAllText($rtvPausePath, ($rtvReceipt | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
    $rtvReceipt | ConvertTo-Json -Depth 8
    exit 0
}

$rtvPaused = Get-Content -LiteralPath $rtvPausePath -Raw | ConvertFrom-Json
$rtvPublished = Get-Content -LiteralPath (Join-Path $rtvInstall 'published.json') -Raw | ConvertFrom-Json
$rtvRegistration = Get-Content -LiteralPath (Join-Path $rtvInstall 'recorder_registration.json') -Raw | ConvertFrom-Json
if ($rtvSettings.model_id -ne $rtvPublished.run_id -or $rtvSettings.study -ne $rtvRegistration.study) { throw 'Recorder pointer differs from the qualified publication.' }
if ((Get-RtvTriggers $rtvTask) -ne $rtvPaused.triggers) { throw 'Scheduled triggers changed during maintenance.' }
$rtvStart = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'start_forecast_recorder.ps1')).Path
$rtvExecutable = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$rtvArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $rtvStart + '" -Study "' + $rtvSettings.study + '"'
$rtvAction = New-ScheduledTaskAction -Execute $rtvExecutable -Argument $rtvArguments -WorkingDirectory $rtvRoot
Set-ScheduledTask -TaskName $rtvTaskName -Action $rtvAction | Out-Null
if ($rtvPaused.was_enabled) { Enable-ScheduledTask -TaskName $rtvTaskName | Out-Null }
$rtvInstalled = Get-ScheduledTask -TaskName $rtvTaskName
if ($rtvInstalled.Actions[0].Execute -ne $rtvExecutable -or $rtvInstalled.Actions[0].Arguments -ne $rtvArguments -or (Get-RtvTriggers $rtvInstalled) -ne $rtvPaused.triggers) { throw 'Installed recorder action or retained triggers differ.' }
$rtvSettings.scheduled_task_verified = $true
[IO.File]::WriteAllText($rtvPointer, ($rtvSettings | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
if ($rtvPaused.was_enabled) { Start-ScheduledTask -TaskName $rtvTaskName }
$rtvReceipt = [ordered]@{verified=$true; task_name=$rtvTaskName; study=$rtvSettings.study; model_id=$rtvSettings.model_id; triggers_preserved=$true; restarted=[bool]$rtvPaused.was_enabled; verified_at=(Get-Date).ToUniversalTime().ToString('o')}
[IO.File]::WriteAllText((Join-Path $rtvInstall 'scheduler.json'), ($rtvReceipt | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
$rtvReceipt | ConvertTo-Json -Depth 8
