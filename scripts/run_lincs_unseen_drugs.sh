#!/bin/bash
# Run lincs-unseen-drugs for every cell line x embedding. Resumable: a
# (embedding, cell line) pair is skipped when its summary already has split_10.
#
# Usage: ./scripts/run_lincs_unseen_drugs.sh [embed_name ...]
#   embed_name resolves to embeddings/<embed_name>.pkl (override embed_path in
#   the config to use another location). Default: ember_base_lincs.
set -uo pipefail
cd "$(dirname "$0")/.."
EMBEDS=("$@"); [ ${#EMBEDS[@]} -eq 0 ] && EMBEDS=(ember_base_lincs)
CELLS=(a375 mcf7 ht29 a549)
TASK=tasks/lincs-unseen-drugs

for e in "${EMBEDS[@]}"; do
  for c in "${CELLS[@]}"; do
    out="$TASK/results/$c/${e}_targets_summary.csv"
    if [ -f "$out" ] && grep -q split_10 "$out"; then echo "skip $e $c (done)"; continue; fi
    echo "=== $e :: $c ==="
    python evaluate.py --config-name lincs_unseen_drugs cell_line="$c" embed_name="$e" \
      || echo "FAILED $e $c"
  done
done
