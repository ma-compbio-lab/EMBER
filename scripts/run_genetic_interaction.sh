#!/bin/bash
# Run the genetic-interaction (HAP1 signed-qGI) task for one or more embeddings.
#
# Usage:  scripts/run_genetic_interaction.sh <embed_name> [<embed_name> ...]
#   embed_name = stem of embeddings/<embed_name>.pkl (nested {'HAP1': {gene: vec}} or flat)
# Env overrides:  EXTRA="max_folds=2"   # extra Hydra overrides passed to every run
# Resumable: an embedding is skipped when its summary already contains split_10.
# Residual GI metrics (AUPRC/AUROC per arm) are written next to the summary as
# <embed_name>_mlp_residual_metrics.csv.
set -uo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "Usage: $0 <embed_name> [<embed_name> ...]"; exit 1; }
EXTRA=${EXTRA:-}
mkdir -p logs

for embed in "$@"; do
  summary="tasks/genetic-interaction/results/${embed}_mlp_summary.csv"
  if [ -f "$summary" ] && grep -q split_10 "$summary"; then
    echo "SKIP (done): $embed"; continue
  fi
  echo "[$(date '+%H:%M:%S')] START genetic_interaction embed_name=$embed"
  python evaluate.py --config-name genetic_interaction embed_name="$embed" $EXTRA \
    2>&1 | tee "logs/genetic_interaction__${embed}.log" | grep -E "^(Fold|  \[|  Best|  split|  [a-z_]+ +:|Summary saved|Residual)"
done
echo "ALL DONE"
