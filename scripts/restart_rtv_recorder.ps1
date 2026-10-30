<# Retarget the existing recorder after the source-only RTV rollout qualifies. #>
$ErrorActionPreference = 'Stop'
$rtvProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$rtvAudit = Join-Path $rtvProjectRoot 'data\audit\rtv_pipeline_043\installation'
$rtvActivation = Get-Content -LiteralPath (Join-Path $rtvAudit 'activation.json') -Raw | ConvertFrom-Json
$rtvStudy = (Resolve-Path -LiteralPath $rtvActivation.study).Path
$rtvPointerPath = Join-Path $rtvProjectRoot 'data\forecasts\recorder.json'
$rtvSettings = Get-Content -LiteralPath $rtvPointerPath -Raw | ConvertFrom-Json
if ($rtvSettings.study -ne $rtvStudy -or -not $rtvActivation.canonical_model_and_history_unchanged) {
    throw 'Recorder pointer or source-only qualification differs.'
}
$rtvTaskName = 'HorseForecastRecorder_038'
$rtvExisting = Get-ScheduledTask -TaskName $rtvTaskName
$rtvTriggersBefore = $rtvExisting.Triggers | Select-Object Id,StartBoundary,Enabled,@{Name='Type';Expression={$_.CimClass.CimClassName}} | ConvertTo-Json -Depth 4 -Compress
$rtvExecutable = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$rtvLauncher = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'start_forecast_recorder.ps1')).Path
$rtvArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $rtvLauncher + '" -Study "' + $rtvStudy + '"'
$rtvAction = New-ScheduledTaskAction -Execute $rtvExecutable -Argument $rtvArguments -WorkingDirectory $rtvProjectRoot
Stop-ScheduledTask -TaskName $rtvTaskName
Set-ScheduledTask -TaskName $rtvTaskName -Action $rtvAction | Out-Null
$rtvInstalled = Get-ScheduledTask -TaskName $rtvTaskName
$rtvTriggersAfter = $rtvInstalled.Triggers | Select-Object Id,StartBoundary,Enabled,@{Name='Type';Expression={$_.CimClass.CimClassName}} | ConvertTo-Json -Depth 4 -Compress
if ($rtvInstalled.Actions.Execute -ne $rtvExecutable -or $rtvInstalled.Actions.Arguments -ne $rtvArguments -or $rtvTriggersBefore -ne $rtvTriggersAfter) {
    throw 'Recorder action or retained triggers differ.'
}
$rtvSettings.scheduled_task_verified = $true
[IO.File]::WriteAllText($rtvPointerPath, ($rtvSettings | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
Start-ScheduledTask -TaskName $rtvTaskName
$rtvReceipt = [ordered]@{ verified = $true; task_name = $rtvTaskName; study = $rtvStudy; executable = $rtvExecutable; arguments = $rtvArguments; triggers_preserved = $true; restarted_at = (Get-Date).ToUniversalTime().ToString('o') }
[IO.File]::WriteAllText((Join-Path $rtvAudit 'scheduler.json'), ($rtvReceipt | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
$rtvReceipt | ConvertTo-Json -Depth 8
