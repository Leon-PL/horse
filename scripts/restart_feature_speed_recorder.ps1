<# Retarget the existing recorder task after the exact speed migration qualifies. #>
$ErrorActionPreference = 'Stop'
$speedProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$speedAudit = Join-Path $speedProjectRoot 'data\audit\live_feature_speed_20261029\installation'
$speedActivation = Get-Content -LiteralPath (Join-Path $speedAudit 'activation.json') -Raw | ConvertFrom-Json
$speedStudy = (Resolve-Path -LiteralPath $speedActivation.study).Path
$speedPointerPath = Join-Path $speedProjectRoot 'data\forecasts\recorder.json'
$speedSettings = Get-Content -LiteralPath $speedPointerPath -Raw | ConvertFrom-Json
if ($speedSettings.study -ne $speedStudy) { throw 'Recorder pointer and qualified study differ.' }
$speedTaskName = 'HorseForecastRecorder_038'
$speedExisting = Get-ScheduledTask -TaskName $speedTaskName
$speedTriggersBefore = $speedExisting.Triggers | Select-Object Id,StartBoundary,Enabled,@{Name='Type';Expression={$_.CimClass.CimClassName}} | ConvertTo-Json -Depth 4 -Compress
$speedExecutable = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$speedLauncher = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'start_forecast_recorder.ps1')).Path
$speedArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $speedLauncher + '" -Study "' + $speedStudy + '"'
$speedAction = New-ScheduledTaskAction -Execute $speedExecutable -Argument $speedArguments -WorkingDirectory $speedProjectRoot
Set-ScheduledTask -TaskName $speedTaskName -Action $speedAction | Out-Null
Enable-ScheduledTask -TaskName $speedTaskName | Out-Null
$speedInstalled = Get-ScheduledTask -TaskName $speedTaskName
$speedTriggersAfter = $speedInstalled.Triggers | Select-Object Id,StartBoundary,Enabled,@{Name='Type';Expression={$_.CimClass.CimClassName}} | ConvertTo-Json -Depth 4 -Compress
if ($speedInstalled.Actions.Execute -ne $speedExecutable -or $speedInstalled.Actions.Arguments -ne $speedArguments -or $speedInstalled.Actions.WorkingDirectory -ne $speedProjectRoot -or $speedTriggersBefore -ne $speedTriggersAfter) {
    throw 'Recorder action or retained triggers differ from the reviewed configuration.'
}
$speedSettings.scheduled_task = $speedTaskName
$speedSettings.scheduled_task_verified = $true
$speedTemporaryPointer = $speedPointerPath + '.speed.tmp'
[IO.File]::WriteAllText($speedTemporaryPointer, ($speedSettings | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
$speedPointerBackup = Join-Path $speedAudit ('recorder_pointer_before_task_' + [Guid]::NewGuid().ToString('N') + '.json')
[IO.File]::Replace($speedTemporaryPointer, $speedPointerPath, $speedPointerBackup)
Start-ScheduledTask -TaskName $speedTaskName
$speedInfo = Get-ScheduledTaskInfo -TaskName $speedTaskName
$speedReceipt = [ordered]@{ verified = $true; task_name = $speedTaskName; study = $speedStudy; executable = $speedExecutable; arguments = $speedArguments; working_directory = $speedProjectRoot; triggers_preserved = $true; next_run = $speedInfo.NextRunTime.ToString('o'); restarted_at = (Get-Date).ToUniversalTime().ToString('o') }
[IO.File]::WriteAllText((Join-Path $speedAudit 'scheduler.json'), ($speedReceipt | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
$speedReceipt | ConvertTo-Json -Depth 8
