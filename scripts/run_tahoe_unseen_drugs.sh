#!/bin/bash
# Run tahoe-unseen-drugs for every cell line x embedding. Resumable: a
# (embedding, cell line) pair is skipped when its summary already has split_10.
#
# Usage: ./scripts/run_tahoe_unseen_drugs.sh [embed_name ...]
#   Default embed_name: ember_base_tahoe (-> embeddings/<name>.pkl).
set -uo pipefail
cd "$(dirname "$0")/.."
EMBEDS=("$@"); [ ${#EMBEDS[@]} -eq 0 ] && EMBEDS=(ember_base_tahoe)
CELLS=(a172 sw480 ncih460 bt474)
TASK=tasks/tahoe-unseen-drugs

for e in "${EMBEDS[@]}"; do
  for c in "${CELLS[@]}"; do
    out="$TASK/results/$c/${e}_targets_summary.csv"
    if [ -f "$out" ] && grep -q split_10 "$out"; then echo "skip $e $c (done)"; continue; fi
    echo "=== $e :: $c ==="
    python evaluate.py --config-name tahoe_unseen_drugs cell_line="$c" embed_name="$e" \
      || echo "FAILED $e $c"
  done
done
