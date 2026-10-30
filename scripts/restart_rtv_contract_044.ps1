$ErrorActionPreference = 'Stop'
$rtvRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$rtvContract = Join-Path $rtvRoot 'data\audit\rtv_repair_features_044\source_contract'
$rtvSettingsPath = Join-Path $rtvRoot 'data\forecasts\recorder.json'
$rtvSettings = Get-Content -LiteralPath $rtvSettingsPath -Raw | ConvertFrom-Json
$rtvRegistration = Get-Content -LiteralPath (Join-Path $rtvContract 'recorder_registration.json') -Raw | ConvertFrom-Json
$rtvPaused = Get-Content -LiteralPath (Join-Path $rtvRoot 'data\audit\rtv_repair_features_044\installation\recorder_paused.json') -Raw | ConvertFrom-Json
if ($rtvSettings.study -ne $rtvRegistration.study -or $rtvSettings.model_id -ne '20261030_135650_092286') { throw 'Contract study or model differs.' }
$rtvTask = Get-ScheduledTask -TaskName 'HorseForecastRecorder_038'
$rtvTriggers = $rtvTask.Triggers | Select-Object Id,StartBoundary,Enabled,@{Name='Type';Expression={$_.CimClass.CimClassName}} | ConvertTo-Json -Depth 4 -Compress
if ($rtvTriggers -ne $rtvPaused.triggers) { throw 'Original task triggers changed.' }
$rtvStart = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'start_forecast_recorder.ps1')).Path
$rtvExecutable = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$rtvArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $rtvStart + '" -Study "' + $rtvSettings.study + '"'
$rtvAction = New-ScheduledTaskAction -Execute $rtvExecutable -Argument $rtvArguments -WorkingDirectory $rtvRoot
Set-ScheduledTask -TaskName 'HorseForecastRecorder_038' -Action $rtvAction | Out-Null
Enable-ScheduledTask -TaskName 'HorseForecastRecorder_038' | Out-Null
$rtvInstalled = Get-ScheduledTask -TaskName 'HorseForecastRecorder_038'
if ($rtvInstalled.Actions[0].Execute -ne $rtvExecutable -or $rtvInstalled.Actions[0].Arguments -ne $rtvArguments) { throw 'Installed task action differs.' }
$rtvSettings.scheduled_task_verified = $true
[IO.File]::WriteAllText($rtvSettingsPath, ($rtvSettings | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
Start-ScheduledTask -TaskName 'HorseForecastRecorder_038'
$rtvReceipt = [ordered]@{verified=$true; task_name='HorseForecastRecorder_038'; study=$rtvSettings.study; model_id=$rtvSettings.model_id; triggers_preserved=$true; restarted=$true; verified_at=(Get-Date).ToUniversalTime().ToString('o')}
[IO.File]::WriteAllText((Join-Path $rtvContract 'scheduler.json'), ($rtvReceipt | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
$rtvReceipt | ConvertTo-Json -Depth 8
