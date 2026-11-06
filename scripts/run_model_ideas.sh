#!/usr/bin/env bash
# Run experiments 057-060 end to end (docs/research/MODEL_IDEAS_BRIEF.md), one stage at a
# time so each fit has the whole machine. Every stage is resumable: rerunning
# this script skips folds whose predictions are already saved.
#
#   bash scripts/run_model_ideas.sh            # everything after the cache
#   bash scripts/run_model_ideas.sh --prepare  # rebuild the research cache first
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=data/logs/model_ideas_057_060.log
mkdir -p data/logs

step() {
  echo "=== $(date '+%F %T') $*" | tee -a "$LOG"
  "$PY" -m "$@" 2>&1 | tee -a "$LOG"
}

if [[ "${1:-}" == "--prepare" ]]; then
  step scripts.select_features prepare
fi
step scripts.luck_weights_059 build
step scripts.connection_profiles_060 build
step scripts.exploded_softmax_057 dev
step scripts.luck_weights_059 dev
step scripts.model_ideas baseline
step scripts.exploded_softmax_057 confirm
step scripts.luck_weights_059 confirm
step scripts.connection_profiles_060 confirm
step scripts.thurstone_058 confirm
echo "=== $(date '+%F %T') all stages done" | tee -a "$LOG"
