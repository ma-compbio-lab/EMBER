#!/bin/bash
# Download the Ember data release from the Hugging Face Hub.
#
# Usage: ./scripts/download_data.sh [component ...]
#   components: checkpoint pretrain controls tasks embeddings
#   (default: checkpoint controls tasks)
#
# Uses the `hf` CLI from huggingface_hub (installed by ./install.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="NickDST/EMBER-data"
COMPONENTS=${*:-"checkpoint controls tasks"}

dl() {  # dl <remote path pattern> <local dir>
    hf download "$REPO" --repo-type dataset --include "$1" --local-dir "$2"
}

for c in $COMPONENTS; do
    case "$c" in
        checkpoint) dl "checkpoints/ember_base/*" . ;;                     # ember_base.pt + config.yaml + gene_vocab.tsv
        pretrain)   dl "data/cxg_250K_train.h5ad" . ; dl "data/cxg_250K_val.h5ad" . ;;
        controls)   dl "data/controls/*" . ;;                             # control cells for extraction
        tasks)      dl "tasks/*" . ;;                                     # task CSVs / atlases
        embeddings) dl "gene_embeddings/*" .                              # precomputed Ember + baseline embeddings
                    mkdir -p embeddings                                   # link as embeddings/<embed_name>.pkl
                    for f in gene_embeddings/*/*.pkl; do ln -sf "../$f" "embeddings/$(basename "$f")"; done ;;
        *) echo "Unknown component: $c"; exit 1 ;;
    esac
done
echo "Done."
