"""One157-input candidate fit; reuse the corrected154 baseline from034."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import quick_win_feature_screen as screen
from src.deployment_validation import file_sha256 as sha
from src.featured_dataset import fe_signature
from src.run_store import get_production_run_id

CASE = ROOT / "data/audit/market_finish_surprise_035"


def main():
    plan = json.loads((CASE / "preparation.json").read_text())
    for name, digest in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError("The audited source/projection changed")
    candidate = ROOT / "feature_sets/market_finish_surprise_035.txt"
    if sha(candidate) != plan["candidate_file_sha256"]:
        raise ValueError("The fixed feature family changed")
    before = {"production": get_production_run_id(), "fe_signature": fe_signature()}
    screen.REFERENCE = ROOT / "data/audit/market_transfer_034/baseline"
    screen.OUTPUT = CASE / "quick_screen"
    screen.run(candidate, CASE / "projection.parquet")
    if before != {"production": get_production_run_id(), "fe_signature": fe_signature()}:
        raise ValueError("Research changed the installed generation")


if __name__ == "__main__":
    main()
