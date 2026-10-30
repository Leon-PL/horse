"""Run exactly one registered fold, using the experiment's frozen execution code."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--registration",type=Path,required=True)
    parser.add_argument("--module",default="scripts.benchmark_surface_repair")
    parser.add_argument("--period",help="Defaults to the first registered period; never queues another")
    args=parser.parse_args()
    spec=json.loads(args.registration.read_text())
    period=args.period or spec["candidate_periods"][0]
    if period not in spec["candidate_periods"]:parser.error("Period must already be registered")
    code=("import sys,importlib; sys.path.insert(0,sys.argv[1]); "
          "module=importlib.import_module(sys.argv[2]); engine=getattr(module,'native',module); "
          "engine.fit_fold(sys.argv[4],registration=sys.argv[3])")
    subprocess.run([sys.executable,"-u","-c",code,str(args.bundle.resolve()),args.module,
                    str(args.registration.resolve()),period],check=True)


if __name__=="__main__":main()
