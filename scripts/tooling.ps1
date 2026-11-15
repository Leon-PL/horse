<#
Run the isolated development tools, or rebuild the project .venv from uv.lock.
Examples: .\scripts\tooling.ps1 setup; .\scripts\tooling.ps1 test; .\scripts\tooling.ps1 venv
#>
[CmdletBinding()]
param(
    [ValidateSet('setup', 'venv', 'lint', 'test')]
    [string]$Action = 'setup',
    [string[]]$TestPath = @('tests/test_generated_racing_invariants.py',
                          'tests/test_matchbook.py',
                          'tests/test_layout.py')
)

$ErrorActionPreference = 'Stop'
$toolRoot = Split-Path -Parent $PSScriptRoot
$toolBase = Join-Path $toolRoot 'data/tooling'
$toolUv = Join-Path $toolBase 'bootstrap/Scripts/uv.exe'
$toolPython = Join-Path $toolBase 'project/Scripts/python.exe'
$toolVariables = @('PYTHONHOME', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
    'UV_PROJECT_ENVIRONMENT', 'UV_PYTHON_INSTALL_DIR', 'UV_PYTHON_DOWNLOADS',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')
$toolPrevious = @{}
foreach ($toolName in $toolVariables) {
    $toolPrevious[$toolName] = [Environment]::GetEnvironmentVariable($toolName, 'Process')
}

Push-Location -LiteralPath $toolRoot
try {
    Remove-Item Env:PYTHONHOME, Env:PYTHONPATH -ErrorAction SilentlyContinue
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $toolBase 'project'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $toolBase 'python'
    $env:UV_PYTHON_DOWNLOADS = 'automatic'
    $env:OMP_NUM_THREADS = '2'
    $env:OPENBLAS_NUM_THREADS = '2'
    $env:NUMBA_NUM_THREADS = '2'

    if ($Action -eq 'setup') {
        if (-not (Test-Path -LiteralPath $toolUv)) {
            & python -m venv (Join-Path $toolBase 'bootstrap')
            if ($LASTEXITCODE -ne 0) { throw 'Could not create the bootstrap environment' }
            & (Join-Path $toolBase 'bootstrap/Scripts/python.exe') -m pip install 'uv==0.13.6'
            if ($LASTEXITCODE -ne 0) { throw 'Could not install uv' }
        }
        & $toolUv python install 3.14.0 --no-bin --no-registry
        if ($LASTEXITCODE -ne 0) { throw 'Could not install the isolated Python runtime' }
        & $toolUv sync --locked --group tooling --python 3.14.0 --managed-python
    } elseif ($Action -eq 'venv') {
        # The project .venv: runtime + dev groups on the same project-local
        # Python. AppData is not used because the Claude app virtualizes it.
        if (-not (Test-Path -LiteralPath $toolUv)) {
            throw 'Run .\scripts\tooling.ps1 setup first'
        }
        $env:UV_PROJECT_ENVIRONMENT = Join-Path $toolRoot '.venv'
        & $toolUv python install 3.14.0 --no-bin --no-registry
        if ($LASTEXITCODE -ne 0) { throw 'Could not install the isolated Python runtime' }
        & $toolUv sync --locked --no-group tooling --python 3.14.0 --managed-python
    } else {
        if (-not (Test-Path -LiteralPath $toolPython)) {
            throw 'Run .\scripts\tooling.ps1 setup first'
        }
        switch ($Action) {
            'lint' { & $toolPython -B -m ruff check . }
            'test' { & $toolPython -B -m pytest @TestPath -q -p no:cacheprovider }
        }
    }
    $toolExitCode = $LASTEXITCODE
} finally {
    Pop-Location
    foreach ($toolName in $toolVariables) {
        [Environment]::SetEnvironmentVariable($toolName, $toolPrevious[$toolName], 'Process')
    }
}
exit $toolExitCode
