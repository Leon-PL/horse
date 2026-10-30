param([Parameter(Mandatory=$true)][ValidateSet('pause','resume')][string]$Phase)
$ErrorActionPreference='Stop'
$fxRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$fxOutput=Join-Path $fxRoot 'data\audit\result_amendments_042\consolidated\fx_refresh'
$fxSettings=Get-Content -LiteralPath (Join-Path $fxRoot 'data\forecasts\recorder.json') -Raw | ConvertFrom-Json
$fxPlan=Get-Content -LiteralPath (Join-Path $fxOutput 'plan.json') -Raw | ConvertFrom-Json
$fxTaskName='HorseForecastRecorder_038'
$fxTask=Get-ScheduledTask -TaskName $fxTaskName
if ($fxSettings.model_id -ne $fxPlan.model_id -or $fxTask.Actions.Count -ne 1 -or
    $fxTask.Actions[0].Arguments.IndexOf([string]$fxSettings.study,[StringComparison]::OrdinalIgnoreCase) -lt 0) {
    throw 'The recorder action or model differs from the qualified daily update.'
}
$fxPausePath=Join-Path $fxOutput 'recorder_paused.json'
if ($Phase -eq 'pause') {
    if (Test-Path -LiteralPath $fxPausePath) { throw 'Preserve the existing pause receipt.' }
    Export-ScheduledTask -TaskName $fxTaskName | Set-Content -LiteralPath (Join-Path $fxOutput 'previous_task.xml') -Encoding utf8
    $fxWasEnabled=[bool]$fxTask.Settings.Enabled
    Disable-ScheduledTask -TaskName $fxTaskName | Out-Null
    Stop-ScheduledTask -TaskName $fxTaskName
    $fxProcesses=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
        $_.CommandLine -like '*run_forecast_recorder.py*' -and
        $_.CommandLine.IndexOf([string]$fxSettings.study,[StringComparison]::OrdinalIgnoreCase) -ge 0
    })
    foreach ($fxProcess in $fxProcesses) { Stop-Process -Id $fxProcess.ProcessId -Force -ErrorAction SilentlyContinue }
    $fxRemaining=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
        $_.CommandLine -like '*run_forecast_recorder.py*' -and
        $_.CommandLine.IndexOf([string]$fxSettings.study,[StringComparison]::OrdinalIgnoreCase) -ge 0
    })
    if ($fxRemaining.Count) { throw 'The current recorder has not stopped.' }
    $fxReceipt=[ordered]@{paused=$true; model_id=$fxSettings.model_id; study=$fxSettings.study;
        was_enabled=$fxWasEnabled; arguments=$fxTask.Actions[0].Arguments;
        paused_at_utc=(Get-Date).ToUniversalTime().ToString('o')}
    [IO.File]::WriteAllText($fxPausePath,($fxReceipt | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
    $fxReceipt | ConvertTo-Json -Depth 8
    exit 0
}
$fxPaused=Get-Content -LiteralPath $fxPausePath -Raw | ConvertFrom-Json
$fxInstalled=Get-Content -LiteralPath (Join-Path $fxOutput 'installed.json') -Raw | ConvertFrom-Json
if ($fxInstalled.model_id -ne $fxSettings.model_id -or $fxPaused.arguments -ne $fxTask.Actions[0].Arguments) {
    throw 'The installed daily update or scheduled action changed.'
}
if ($fxPaused.was_enabled) {
    Enable-ScheduledTask -TaskName $fxTaskName | Out-Null
    Start-ScheduledTask -TaskName $fxTaskName
}
$fxFinal=Get-ScheduledTask -TaskName $fxTaskName
$fxReceipt=[ordered]@{verified=$true; restarted=[bool]$fxPaused.was_enabled;
    enabled=[bool]$fxFinal.Settings.Enabled; task_name=$fxTaskName; state=[string]$fxFinal.State;
    model_id=$fxSettings.model_id; study=$fxSettings.study;
    verified_at_utc=(Get-Date).ToUniversalTime().ToString('o')}
[IO.File]::WriteAllText((Join-Path $fxOutput 'scheduler.json'),($fxReceipt | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
$fxReceipt | ConvertTo-Json -Depth 8
