"""One small corrected154 versus159 screen; preserve all deployed definitions."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import quick_win_feature_screen as screen
from src.deployment_validation import file_sha256 as sha
from src.featured_dataset import fe_signature
from src.run_store import get_production_run_id

CASE = ROOT / "data/audit/market_transfer_034"


def main():
    plan = json.loads((CASE / "preparation.json").read_text())
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Fixed research source changed after input audit")
    for name, digest in plan["input_sha256"].items():
        if sha(CASE / name) != digest:
            raise ValueError("Audited research inputs changed")
    candidate = ROOT / "feature_sets/market_transfer_034.txt"
    if sha(candidate) != plan["candidate_file_sha256"]:
        raise ValueError("The preset feature family changed")
    before = {"production": get_production_run_id(), "fe_signature": fe_signature()}
    screen.REFERENCE = CASE / "baseline"
    # A corrected154 cheap baseline has not existed before this case. Train
    # it once, then require exact cache reuse in the actual proposal comparison.
    screen.OUTPUT = CASE / "baseline_cache"
    screen.run(CASE / "baseline/features.txt")
    screen.OUTPUT = CASE / "quick_screen"
    screen.run(candidate, CASE / "projection.parquet")
    if before != {"production": get_production_run_id(), "fe_signature": fe_signature()}:
        raise ValueError("The research screen changed the installed generation")


if __name__ == "__main__":
    main()
