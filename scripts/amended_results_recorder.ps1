param([Parameter(Mandatory=$true)][ValidateSet('pause','resume')][string]$Phase)
$ErrorActionPreference = 'Stop'
$repairRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$repairInstall = Join-Path $repairRoot 'data\audit\result_amendments_042\consolidated\installation'
$repairPlan = Get-Content -LiteralPath (Join-Path $repairInstall 'plan.json') -Raw | ConvertFrom-Json
$repairPointer = Join-Path $repairRoot 'data\forecasts\recorder.json'
$repairSettings = Get-Content -LiteralPath $repairPointer -Raw | ConvertFrom-Json
$repairTaskName = 'HorseForecastRecorder_038'
$repairTask = Get-ScheduledTask -TaskName $repairTaskName
$repairPausePath = Join-Path $repairInstall 'recorder_paused.json'

if ($Phase -eq 'pause') {
    if (Test-Path -LiteralPath $repairPausePath) { throw 'The pause receipt already exists; inspect it before repeating maintenance.' }
    if ($repairSettings.model_id -ne $repairPlan.parent_run) { throw 'The recorder belongs to another model.' }
    if ($repairTask.Actions.Count -ne 1 -or $repairTask.Actions[0].Arguments.IndexOf([string]$repairSettings.study, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        throw 'The scheduled action differs from the current pinned study.'
    }
    $repairWasEnabled = [bool]$repairTask.Settings.Enabled
    Export-ScheduledTask -TaskName $repairTaskName | Set-Content -LiteralPath (Join-Path $repairInstall 'previous_task.xml') -Encoding utf8
    Copy-Item -LiteralPath $repairPointer -Destination (Join-Path $repairInstall 'previous_recorder.json')
    Disable-ScheduledTask -TaskName $repairTaskName | Out-Null
    Stop-ScheduledTask -TaskName $repairTaskName
    $repairProcesses = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
        $_.CommandLine -like '*run_forecast_recorder.py*' -and
        $_.CommandLine.IndexOf([string]$repairSettings.study, [StringComparison]::OrdinalIgnoreCase) -ge 0
    })
    foreach ($repairProcess in $repairProcesses) { Stop-Process -Id $repairProcess.ProcessId -Force -ErrorAction SilentlyContinue }
    $repairRemaining = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
        $_.CommandLine -like '*run_forecast_recorder.py*' -and
        $_.CommandLine.IndexOf([string]$repairSettings.study, [StringComparison]::OrdinalIgnoreCase) -ge 0
    })
    if ($repairRemaining.Count -gt 0) { throw 'The old recorder has not stopped.' }
    $repairReceipt = [ordered]@{
        paused = $true; model_id = $repairSettings.model_id; study = $repairSettings.study
        was_enabled = $repairWasEnabled; stopped_pids = @($repairProcesses | ForEach-Object { $_.ProcessId })
        paused_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    [IO.File]::WriteAllText($repairPausePath, ($repairReceipt | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
    $repairReceipt | ConvertTo-Json -Depth 8
    exit 0
}

$repairPaused = Get-Content -LiteralPath $repairPausePath -Raw | ConvertFrom-Json
$repairPublished = Get-Content -LiteralPath (Join-Path $repairInstall 'published.json') -Raw | ConvertFrom-Json
$repairRegistration = Get-Content -LiteralPath (Join-Path $repairInstall 'recorder_registration.json') -Raw | ConvertFrom-Json
if ($repairSettings.model_id -ne $repairPublished.run_id -or $repairSettings.study -ne $repairRegistration.study) {
    throw 'The new recorder pointer differs from the qualified publication.'
}
$repairStartScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'start_forecast_recorder.ps1')).Path
$repairExecutable = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$repairArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $repairStartScript + '" -Study "' + $repairSettings.study + '"'
$repairAction = New-ScheduledTaskAction -Execute $repairExecutable -Argument $repairArguments -WorkingDirectory $repairRoot
Set-ScheduledTask -TaskName $repairTaskName -Action $repairAction | Out-Null
if ($repairPaused.was_enabled) {
    Enable-ScheduledTask -TaskName $repairTaskName | Out-Null
    Start-ScheduledTask -TaskName $repairTaskName
}
$repairInstalled = Get-ScheduledTask -TaskName $repairTaskName
if ($repairInstalled.Actions[0].Execute -ne $repairExecutable -or $repairInstalled.Actions[0].Arguments -ne $repairArguments) {
    throw 'The registered action differs from the new pinned study.'
}
$repairSettings.scheduled_task = $repairTaskName
$repairSettings.scheduled_task_verified = $true
$repairPointerTemporary = Join-Path (Split-Path $repairPointer -Parent) ([IO.Path]::GetRandomFileName() + '.tmp')
[IO.File]::WriteAllText($repairPointerTemporary, ($repairSettings | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
Move-Item -LiteralPath $repairPointerTemporary -Destination $repairPointer -Force
$repairTaskInfo = Get-ScheduledTaskInfo -TaskName $repairTaskName
$repairReceipt = [ordered]@{
    verified = $true; task_name = $repairTaskName; state = [string]$repairInstalled.State
    enabled = [bool]$repairInstalled.Settings.Enabled; restarted = [bool]$repairPaused.was_enabled
    model_id = $repairSettings.model_id; study = $repairSettings.study
    executable = $repairExecutable; arguments = $repairArguments
    next_run = $repairTaskInfo.NextRunTime.ToString('o'); verified_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
}
[IO.File]::WriteAllText((Join-Path $repairInstall 'scheduler.json'), ($repairReceipt | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
$repairReceipt | ConvertTo-Json -Depth 8
