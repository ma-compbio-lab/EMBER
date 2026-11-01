#!/bin/bash
# Run the three perturbation-response tasks for one or more embeddings.
#
# Usage:  scripts/run_pert_response.sh <embed_name> [<embed_name> ...]
#   embed_name = stem of embeddings/<embed_name>.pkl (nested {cell_line: {gene: vec}},
#                keys HepG2 / Jurkat / K562 / RPE1)
# Env overrides:
#   TASKS="pert_response_unseen_pert pert_response_cross_cellline pert_response_cross_cellline_unseen_pert"
#   CELL_LINES="hepg2 jurkat k562 rpe1"
#   EXTRA="max_folds=2"          # extra Hydra overrides passed to every run
# Resumable: a (task, cell line, embedding) run is skipped when its summary
# already contains the final fold (split_10).
set -uo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "Usage: $0 <embed_name> [<embed_name> ...]"; exit 1; }

TASKS=${TASKS:-"pert_response_unseen_pert pert_response_cross_cellline pert_response_cross_cellline_unseen_pert"}
CELL_LINES=${CELL_LINES:-"hepg2 jurkat k562 rpe1"}
EXTRA=${EXTRA:-}
mkdir -p logs

for task in $TASKS; do
  task_dir="tasks/${task//_/-}"
  for cl in $CELL_LINES; do
    for embed in "$@"; do
      summary="${task_dir}/results/${cl}/${embed}_mlp_summary.csv"
      if [ -f "$summary" ] && grep -q split_10 "$summary"; then
        echo "SKIP (done): $task $cl $embed"; continue
      fi
      echo "[$(date '+%H:%M:%S')] START $task cell_line=$cl embed_name=$embed"
      python evaluate.py --config-name "$task" cell_line="$cl" embed_name="$embed" $EXTRA \
        2>&1 | tee "logs/${task}__${cl}__${embed}.log" | grep -E "^(Fold|  \[|  Best|Summary saved)" 
    done
  done
done
echo "ALL DONE"
