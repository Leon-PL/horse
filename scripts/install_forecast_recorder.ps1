param([Parameter(Mandatory=$true)][string]$Study)
$ErrorActionPreference = 'Stop'
$studyPath = (Resolve-Path -LiteralPath $Study).Path
if (-not (Test-Path -LiteralPath (Join-Path $studyPath 'protocol.sha256.json'))) {
    throw 'Create and verify a pinned study before installing its recorder.'
}
$runnerPath = (Resolve-Path (Join-Path $PSScriptRoot 'start_forecast_recorder.ps1')).Path
$taskName = 'HorseForecastRecorder_038'
$executable = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
if (-not (Test-Path -LiteralPath $executable)) { throw 'The stable Windows PowerShell launcher is unavailable.' }
$arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $runnerPath + '" -Study "' + $studyPath + '"'
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and ($existing.Actions.Execute -ne $executable -or $existing.Actions.Arguments -ne $arguments)) {
    throw 'This task name already has a different action; inspect it before replacing it.'
}
$action = New-ScheduledTaskAction -Execute $executable -Argument $arguments -WorkingDirectory (Split-Path $PSScriptRoot -Parent)
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$triggers = @((New-ScheduledTaskTrigger -Daily -At '09:00'), (New-ScheduledTaskTrigger -AtLogOn -User $identity))
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 24) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $triggers -Settings $settings -Principal $principal -Description 'Capture pinned horse forecasts, actual cards and market references; refresh settled history and assess immutable records.' -Force | Out-Null
$installed = Get-ScheduledTask -TaskName $taskName
$taskInfo = Get-ScheduledTaskInfo -TaskName $taskName
if ($installed.Actions.Execute -ne $executable -or $installed.Actions.Arguments -ne $arguments) {
    throw 'Registered recorder action differs from the reviewed command.'
}
$projectRoot = Split-Path $PSScriptRoot -Parent
$pointerPath = Join-Path $projectRoot 'data\forecasts\recorder.json'
$pointer = Get-Content -LiteralPath $pointerPath -Raw | ConvertFrom-Json
if ($pointer.study -ne $studyPath) { throw 'Recorder pointer and installed task name different studies.' }
$pointer.scheduled_task = $taskName
$pointer.scheduled_task_verified = $true
[System.IO.File]::WriteAllText($pointerPath, ($pointer | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
$receipt = [ordered]@{ task_name = $taskName; state = [string]$installed.State; study = $studyPath; executable = $executable; arguments = $arguments; next_run = $taskInfo.NextRunTime.ToString('o'); registered_at = (Get-Date).ToUniversalTime().ToString('o'); verified = $true }
[System.IO.File]::WriteAllText((Join-Path $projectRoot 'data\audit\predictive_review_038\scheduler.json'), ($receipt | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
$receipt | ConvertTo-Json -Depth 8
