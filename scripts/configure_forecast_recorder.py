"""Create/verify the final pinned study and register its descriptive readout."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.forward_store import ForwardStudy, create_study, utcnow
from src.forward_readout import register_plan, load_plan, readout_readiness
from src.forward_recorder import replace_json, run_recorder
from src.matchbook_client import MatchbookClient
from src.run_store import get_production_run_id


def run(path,discover=False):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT/"data/forecasts/forward").resolve()):
        raise ValueError("Recorder studies belong inside data/forecasts/forward")
    study = ForwardStudy(path) if (path/"protocol.json").exists() else create_study(path)
    study.verify()
    if study.protocol["model_id"] != get_production_run_id():
        raise ValueError("The recorder must pin the selected installed model")
    plan_path = path/"readout_plan.json"
    if not plan_path.exists():
        register_plan(study,plan_path)
    readiness = readout_readiness(study,load_plan(plan_path))
    settings = {"study":str(path),"model_id":study.protocol["model_id"],
        "protocol_sha256":study.protocol_sha256,"first_eligible_date":study.protocol["first_eligible_date"],
        "configured_at":utcnow().isoformat(),"scheduled_task":None,"scheduled_task_verified":False}
    replace_json(ROOT/"data/forecasts/recorder.json",settings)
    result = {"settings":settings,"readout":readiness}
    if discover:
        result["feed_probe"] = run_recorder(study,client=MatchbookClient(),duration_minutes=1,discover_only=True)
        if not result["feed_probe"].get("last_discovery"):
            raise RuntimeError("Recorder feed discovery did not succeed; inspect its heartbeat errors")
    replace_json(ROOT/"data/audit/predictive_review_038/recorder_setup.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True,type=Path)
    parser.add_argument("--discover",action="store_true")
    args = parser.parse_args()
    run(args.study,args.discover)
