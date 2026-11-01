# tahoe-unseen-drugs

Within-cell-line drug-response regression on **unseen drugs** on the Tahoe-100M atlas
(DESeq2 log2FC vs DMSO, 1024 genes, 172 DrugBank drugs). Four cell lines, each evaluated
separately with 10 target-aware folds (drugs with the same target set share a fold). Same
driver, model and sweep as `lincs-unseen-drugs`.

```bash
python evaluate.py --config-name tahoe_unseen_drugs cell_line=sw480 embed_name=<name>
./scripts/run_tahoe_unseen_drugs.sh <name>          # all 4 lines, resumable
```
Config: `configs/eval/tahoe_unseen_drugs.yaml`. Results: `results/<cell_line>/<name>_targets_{summary,preds}.csv`.

## Files

| File | In repo | Description |
|---|---|---|
| `{a172,bt474,ncih460,sw480}_split.json` | yes | 10-fold drug holdout per line |
| `genes_eval.txt` | yes | the 1024 evaluated genes, in atlas `var` order |
| `drug_lookup.csv` | yes | `broad_id,targets,moa_classes,n_targets` (`broad_id` = DrugBank id here); `\|`-joined, aligned |
| `atlas.h5ad` | **download** | 688 (cell line × drug) contrasts × 1024 genes |

Download the atlas from the Ember data release:
`https://huggingface.co/datasets/NickDST/EMBER-data` → `tasks/tahoe-unseen-drugs/atlas.h5ad`.

### Atlas schema
- `X` (dense float32): DESeq2 log2FC; ~2.6 % NaN. **NaN = untested/filtered — genuinely
  missing, never zero.** Only observed cells are trained on / scored.
- `layers['padj']`: DESeq2 adjusted p-values. **Ignored by default** (`use_padj: false`);
  set `use_padj=true` to additionally require finite padj. Opt-in so that the valid-gene set
  never changes silently.
- `obs['broad_id']`: DrugBank id; `obs['cell_line']`: Cellosaurus id (the embedding key);
  `obs['cell_name']`: readable name.

### Split json schema
Same as lincs-unseen-drugs: `ct_label` (CVCL id), `core_training_cts` = the same line,
`n_folds: 10`, `fold_k: {val_perts, test_perts}` (train = remaining drugs of the line).

## Cell-line keys
| cell_line | Cellosaurus id (`ct_label`) |
|---|---|
| a172 | CVCL_0131 |
| sw480 | CVCL_0546 |
| ncih460 | CVCL_0459 |
| bt474 | CVCL_0179 |

## Embedding contract
`embeds[ct_label][gene_symbol] -> np.float32[D]`, built on each line's own DMSO control cells
(Tahoe plate-12 controls, 2000 cells/line; see `extract_embeddings.py`). Missing genes are
tolerated (test cells imputed with the train-mean LFC, `source='imputed'`); drugs with more than
half their targets missing are dropped. Flat `{gene: vector}` pickles are accepted.

## Output columns
Identical to lincs-unseen-drugs (`*_summary.csv` with a `fold='all'` row; `*_preds.csv` with
`source` ∈ {model, imputed}). Headline metric: `per_drug_pearson_delta`.
