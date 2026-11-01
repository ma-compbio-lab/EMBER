# lincs-unseen-drugs

Within-cell-line drug-response regression on **unseen drugs** (LINCS L1000 CRISPR/compound
atlas, DESeq2 log2FC over 964 landmark genes). Four cell lines, each evaluated separately
with 10 target-aware folds: drugs that share an exact target set are kept in the same fold, so
a held-out drug's target set never appears in training.

```bash
python evaluate.py --config-name lincs_unseen_drugs cell_line=a549 embed_name=<name>
./scripts/run_lincs_unseen_drugs.sh <name>          # all 4 lines, resumable
```
Config: `configs/eval/lincs_unseen_drugs.yaml`. Results: `results/<cell_line>/<name>_targets_{summary,preds}.csv`.

## Files

| File | In repo | Description |
|---|---|---|
| `{a375,a549,ht29,mcf7}_split.json` | yes | 10-fold drug holdout per line (schema below) |
| `genes_eval.txt` | yes | the 964 landmark genes, one per line, in atlas `var` order |
| `drug_lookup.csv` | yes | `broad_id,targets,moa_classes,n_targets`; `targets`/`moa_classes` are `\|`-joined and aligned; MoA ∈ {inhibitory, activating, other} |
| `atlas.h5ad` | **download** | 1088 (cell line × drug) contrasts × 964 genes |

Download the atlas from the Ember data release:
`https://huggingface.co/datasets/NickDST/EMBER-data` → `tasks/lincs-unseen-drugs/atlas.h5ad`.

### Atlas schema
- `X` (dense float32): DESeq2 log2FC. **NaN = contrast untested/filtered — genuinely missing,
  never zero.** The evaluation only trains/scores on observed cells; it never 0-fills.
- `obs['broad_id']`: drug id (BRD-…); `obs['cell_line']`: DepMap id (the embedding key);
  `obs['cmap_name']`, `obs['cell_iname']`: readable names.
- No layers.

### Split json schema
```
ct_label            DepMap id of the held-out line (= atlas obs.cell_line = embedding key)
ct_slug, cell_name  'a549', 'A549'
core_training_cts   {name: ct_label} -- the same line (within-line task)
n_folds             10
fold_k              {val_perts: [broad_id, ...], test_perts: [broad_id, ...]}
                    train = every other drug of the line
split_strategy      exact_target_set_group
```

## Cell-line keys
| cell_line | DepMap id (`ct_label`) |
|---|---|
| a375 | ACH-000219 |
| a549 | ACH-000681 |
| ht29 | ACH-000552 |
| mcf7 | ACH-000019 |

## Embedding contract
`embeds[ct_label][gene_symbol] -> np.float32[D]`, one sub-dict per line, built by running the
model over that line's **own control cells** (see `extract_embeddings.py`; Ember's reference
pickle uses sci-Plex vehicle controls for A549/MCF7, GSE164614 for A375, GSE167262 for HT-29).
Genes absent from a line's sub-dict are allowed: their test cells are reported with the
per-gene train-mean LFC and `source='imputed'`. A drug is dropped when more than half of its
targets have no embedding. A flat `{gene_symbol: vector}` pickle is accepted too.

## Output columns
`*_summary.csv`: `fold, cell_line, per_drug_pearson, per_drug_spearman, per_drug_pearson_delta,
per_drug_spearman_delta, n_drugs_delta, val_selection_criteria, n_drugs, pearson, spearman, rmse,
n_samples, n_imputed, lr, batch_size, weight_decay, dropout` (+ a `fold='all'` mean row).
`*_preds.csv`: `fold, cell_line, broad_id, gene, y_true, y_pred, source`.
The headline metric is `per_drug_pearson_delta` (drug-specific signal above the per-gene
train-mean floor); `per_drug_pearson` is the raw version.
