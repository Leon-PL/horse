"""Keep the running017 first-fold checkpoint and stop its later queued fits.

The user requested faster feature screening. This changes the amount of
evaluation, never fitted parameters, predictions, or the original manifest.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"data/audit/model_improvement/017_surface_family"


def main():
    amendment=OUTPUT/"screening_amendment.json"
    if not amendment.exists():
        registration=OUTPUT/"registration.json"
        value={"changed_at_utc":datetime.now(timezone.utc).isoformat(),
            "reason":"User requested less repeated quarter training and faster feature screening",
            "original_registration_sha256":hashlib.sha256(registration.read_bytes()).hexdigest(),
            "retained_period":"2025Q3","deferred_periods":["2025Q4","2026Q1"],
            "first_period_result_existed_at_amendment":(OUTPUT/"2025Q3_report.json").exists(),
            "scope":"One-period screen only; original three-period advancement gate is not satisfied or redefined"}
        amendment.write_text(json.dumps(value,indent=2))
    report=OUTPUT/"2025Q3_report.json"
    deadline=time.monotonic()+3600
    while not report.exists():
        if time.monotonic()>deadline:raise TimeoutError("First-fold result not ready; inspect the native job")
        time.sleep(.25)
    # The report publishes after the model and predictions are complete. Stop
    # only this frozen017 runner; the selected production refit stays running.
    command=("$jobs=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' "
             "-and $_.CommandLine -like '*_research_snapshots/017_surface_family/scripts/benchmark_surface_repair.py*' }; "
             "$jobs | Sort-Object ParentProcessId -Descending | ForEach-Object { "
             "Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
    subprocess.run(["powershell","-NoProfile","-Command",command],check=True,capture_output=True)
    r=json.loads(report.read_text())
    result={"stage":"one_period_screen_complete","period":"2025Q3","races":r["races"],
        "scores":{k:r[k] for k in ("baseline","candidate","paired_delta_candidate_minus_baseline")},
        "scope":"Screening evidence; no multi-period advancement or deployment claim",
        "later_fits_stopped":True,"production_refit_untouched":True,
        "recorded_at_utc":datetime.now(timezone.utc).isoformat()}
    (OUTPUT/"screening_result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
