# data/

Tracked in git:
- `cxg_overfit.h5ad` — 10 CELLxGENE cells, input for the pretraining smoke test.

Downloaded by `scripts/download_data.sh` (Hugging Face Hub, `NickDST/EMBER-data`):

| Path | Used by | Contents |
|---|---|---|
| `cxg_250K_train.h5ad`, `cxg_250K_val.h5ad` | `pretrain.py` | 250K CELLxGENE cells (raw counts, 22,172 protein-coding genes, `var['feature_name']` symbols, `obs['cell_type']`); val = held-out donors |
| `controls/{K562,Jurkat,HepG2,RPE1}_controls.h5ad` | `pert-response-*` embeddings | non-targeting control cells of the four Perturb-seq lines; symbols in `var['gene_name']` |
| `controls/hap1_controls.h5ad` | `genetic-interaction` embeddings | HAP1 control cells; `var_names` are symbols |
| `controls/lincs_cellline_controls.h5ad` | `lincs-unseen-drugs` embeddings | A549 / MCF7 (sci-Plex vehicle), A375 (GSE164614), HT-29 (GSE167262) control cells; `obs['ach']` = DepMap id |
| `controls/tahoe_controls.h5ad` | `tahoe-unseen-drugs` embeddings | Tahoe-100M plate-12 DMSO cells; `obs['cell_line']` = Cellosaurus id |

All control files hold **raw UMI counts** in `.X`; `extract_embeddings.py` applies
the model's own normalisation and checks gene alignment before use.
