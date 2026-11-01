# pert-response-unseen-pert

Unseen perturbations within one cell line: folds hold out perturbations; each CSV holds a single cell line.

Config: `configs/eval/pert_response_unseen_pert.yaml` — see the comments there for
the full task definition.

## Expected files (download, not in git)

```
tasks/pert-response-unseen-pert/
├── hepg2.csv
├── jurkat.csv
├── k562.csv
└── rpe1.csv
```

Download from the Ember dataset on the Hugging Face Hub:
<https://huggingface.co/datasets/NickDST/EMBER-data> → `tasks/pert-response-unseen-pert/`,
or run `scripts/download_data.sh`.

Columns: `pert, gene, cell_line, lfc, padj, label, neg_type, split_1 … split_10`
— `label` is 1 if `gene` is differentially expressed under `pert`; `split_N`
is one of `train` / `val` / `test` (/ `exclude`, dropped for that fold).

## Run

```bash
python evaluate.py --config-name pert_response_unseen_pert cell_line=k562 embed_name=<pkl stem>
scripts/run_pert_response.sh <embed_name>        # all 3 tasks x 4 cell lines
```

Results land in `results/<cell_line>/<embed_name>_mlp_{summary,preds}.csv`.
