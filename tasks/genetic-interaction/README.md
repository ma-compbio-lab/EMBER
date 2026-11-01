# genetic-interaction

Signed quantitative genetic-interaction (qGI) regression in HAP1 with held-out
LIBRARY genes. Config: `configs/eval/genetic_interaction.yaml` — see the
comments there for the full task definition and the residual metrics.

## Expected files (download, not in git)

```
tasks/genetic-interaction/
└── gi_dataset.csv
```

Download from the Ember dataset on the Hugging Face Hub:
<https://huggingface.co/datasets/NickDST/EMBER-data> → `tasks/genetic-interaction/`,
or run `scripts/download_data.sh`.

Columns: `query_gene, library_gene, qGI, abs_qGI, FDR, cell_line, split_1 … split_10`
— `qGI` is the signed interaction score (regression target), `FDR` its
significance (used to build the residual-metric labels), `cell_line` is `HAP1`,
and `split_N` is `train` / `val` / `test` (library genes are held out per fold).

## Run

```bash
python evaluate.py --config-name genetic_interaction embed_name=<pkl stem>
scripts/run_genetic_interaction.sh <embed_name> [<embed_name> ...]
```

Results land in `results/<embed_name>_mlp_{summary,preds,residual_metrics}.csv`.
The residual metrics (main-effect-corrected AUPRC / AUROC per arm) can be
recomputed from a preds file with
`python -m ember.eval.gi_residual --preds results/<embed_name>_mlp_preds.csv --dataset gi_dataset.csv`.
