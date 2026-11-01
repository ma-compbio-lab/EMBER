#!/bin/bash
# Extract the contextual gene embeddings used by the six benchmark tasks from one
# checkpoint. Control-cell inputs come from the Ember data release (see README:
# Data). Each call writes embeddings/<checkpoint stem>_<task>.pkl, keyed by the
# cell-line label the corresponding task expects.
#
# Usage: ./scripts/run_extract_all.sh [checkpoint.pt]      (default: the released checkpoint)
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:-checkpoints/ember_base/ember_base.pt}
OUT="embeddings/$(basename "$CKPT" .pt)"      # embeddings/ember_base_<task>.pkl

# pert-response-* (K562 / Jurkat / HepG2 / RPE1 non-targeting control cells; var has
# Ensembl ids with symbols in var['gene_name'])
python extract_embeddings.py --checkpoint "$CKPT" \
    --h5ad data/controls/K562_controls.h5ad data/controls/Jurkat_controls.h5ad \
           data/controls/HepG2_controls.h5ad data/controls/RPE1_controls.h5ad \
    --group-const K562 Jurkat HepG2 RPE1 --gene-col gene_name \
    --n-samples 1 --out "${OUT}_pert_response.pkl"

# genetic-interaction (HAP1 control cells, var_names are symbols)
python extract_embeddings.py --checkpoint "$CKPT" \
    --h5ad data/controls/hap1_controls.h5ad --group-const HAP1 \
    --n-samples 1 --out "${OUT}_genetic_interaction.pkl"

# lincs-unseen-drugs (A549 / MCF7 / A375 / HT-29 matched control cells, keyed by
# DepMap ACH id in obs['ach'])
python extract_embeddings.py --checkpoint "$CKPT" \
    --h5ad data/controls/lincs_cellline_controls.h5ad --group-col ach \
    --n-samples 4 --out "${OUT}_lincs.pkl"

# tahoe-unseen-drugs (Tahoe plate-12 DMSO cells, keyed by Cellosaurus id in obs['cell_line'])
python extract_embeddings.py --checkpoint "$CKPT" \
    --h5ad data/controls/tahoe_controls.h5ad --group-col cell_line \
    --n-samples 2 --out "${OUT}_tahoe.pkl"
