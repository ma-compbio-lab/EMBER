"""Evaluate gene embeddings on a gene-pair task (classification or regression).

Each task CSV holds one row per gene pair with a label and `split_N` columns
(train / val / test; any other value, e.g. `exclude`, drops the row for that
fold). For every fold an MLP over the concatenated pair embeddings is trained
with a hyperparameter sweep, selected on the validation set, and scored on
the test set. Test pairs with a gene missing from the embedding are scored at
the training base rate so every embedding is evaluated on the same test set.

Entry point: ``evaluate.py`` dispatches here for configs with ``evaluator: pairs``.

    python evaluate.py --config-name pert_response_unseen_pert cell_line=k562 \
        embed_name=ember_base_pert_response
    python evaluate.py --config-name genetic_interaction embed_path=embeddings/my.pkl embed_name=my

    # Override sweep values / restrict to the first N folds
    python evaluate.py --config-name pert_response_cross_cellline cell_line=hepg2 \
        embed_name=ember_base_pert_response sweep.lr=[1e-3] max_folds=2

Configs live in configs/eval/. Outputs:
    {output_dir}/{embed_name}{name_suffix}_mlp_summary.csv   one row per (fold, cell line)
    {output_dir}/{embed_name}{name_suffix}_mlp_preds.csv     one row per test pair
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from ember.eval.pairs import (seed_everything, is_nested_embeds, filter_missing_rows,
                              run_fold, compute_metrics, compute_regression_metrics,
                              compute_macro_metrics, compute_macro_metrics_pooled)


def load_embeddings(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def run(cfg: DictConfig):
    """Run all folds for one (config, embedding) pair and write the CSVs."""
    print("=" * 60)
    print("Ember Evaluation")
    print("=" * 60)
    seed = cfg.get('seed', 42)
    seed_everything(seed)
    print(f"seed={seed} (deterministic)")

    embed_path = cfg.get('embed_path', None) or Path(cfg.root) / 'embeddings' / f"{cfg.embed_name}.pkl"
    print(f"Embeddings: {embed_path}")
    print(f"Dataset: {cfg.dataset}")

    task_type = cfg.get('task_type', 'classification')
    label_col = cfg.get('label_col', 'label')
    gene_columns = list(cfg.gene_columns) if cfg.get('gene_columns', None) else ['pert', 'gene']
    missing_gene_strategy = cfg.get('missing_gene_strategy', 'filter')
    val_selection_criteria = cfg.get('val_selection_criteria', 'loss')
    if cfg.get('model_type', 'mlp') != 'mlp' or cfg.get('fusion_strat', 'concat') != 'concat':
        raise ValueError("Only model_type='mlp' with fusion_strat='concat' is supported")
    if missing_gene_strategy != 'filter':
        raise ValueError("Only missing_gene_strategy='filter' is supported")
    if task_type not in ('classification', 'regression'):
        raise ValueError(f"Unknown task_type: {task_type!r}")
    print(f"Task type: {task_type}, Label column: {label_col}, Gene columns: {gene_columns}")
    if val_selection_criteria != 'loss':
        print(f"Val selection criteria: {val_selection_criteria}")

    ct_specific = cfg.get('ct_specific', False)
    ct_col = cfg.get('ct_column', None)
    if ct_specific:
        print(f"Cell-line specific: {ct_col}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Embeddings: flat {gene: vec} or nested {cell_line: {gene: vec}}
    embeds = load_embeddings(str(embed_path))
    embeds_nested = is_nested_embeds(embeds)
    if embeds_nested:
        print(f"Loaded nested embeddings: { {ct: len(g) for ct, g in embeds.items()} }")
        embed_keys = set()
        for ct_embeds in embeds.values():
            embed_keys.update(ct_embeds.keys())
        per_ct_genes = {ct: set(ct_embeds.keys()) for ct, ct_embeds in embeds.items()}
    else:
        print(f"Loaded {len(embeds)} embeddings")
        embed_keys = set(embeds.keys())
        per_ct_genes = None

    # Dataset
    if str(cfg.dataset).endswith('.parquet'):
        df = pd.read_parquet(cfg.dataset)
    else:
        df = pd.read_csv(cfg.dataset)
    print(f"Loaded {len(df)} samples")
    use_ct_col = ct_col if (ct_specific and ct_col and ct_col in df.columns) else None
    if use_ct_col:
        print(f"Cell lines: {list(df[use_ct_col].unique())}")

    for col in gene_columns:
        col_genes = set(df[col].unique())
        print(f"Missing embeddings: {len(col_genes - embed_keys)}/{len(col_genes)} {col} genes")

    # Folds: split_N / foldN columns in CSV order, optionally capped to the first N
    split_cols = [c for c in df.columns if c.startswith('split_') or c.startswith('fold')]
    if not split_cols:
        raise ValueError("No split_N or foldN columns found in dataset")
    max_folds = cfg.get('max_folds', None)
    if max_folds is not None:
        split_cols = sorted(
            split_cols,
            key=lambda c: int(''.join(ch for ch in c if ch.isdigit()) or 0),
        )[:int(max_folds)]
    print(f"Running {len(split_cols)} folds: {split_cols}")

    fold_cfg = {
        'gene_columns': gene_columns,
        'label_col': label_col,
        'hidden_dim': cfg.hidden_dim,
        'n_layers': cfg.n_layers,
        'max_epochs': cfg.max_epochs,
        'patience': cfg.patience,
        'tol': cfg.tol,
        'sweep': {k: OmegaConf.to_container(v, resolve=True) for k, v in cfg.sweep.items()},
        'val_selection_criteria': val_selection_criteria,
        'optimizer': cfg.get('optimizer', 'adam'),
        # Defaults to 'FDR', the column used for the reported results.
        'residual_fdr_col': cfg.get('residual_fdr_col', 'FDR'),
    }

    all_results = []
    all_predictions = []
    for fold_col in split_cols:
        print(f"\n{'=' * 40}\nFold: {fold_col}\n{'=' * 40}")

        df_train = df[df[fold_col] == 'train']
        df_val = df[df[fold_col] == 'val']
        df_test = df[df[fold_col] == 'test'].reset_index(drop=True)
        print(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

        # Keep only pairs whose genes have embeddings (per cell line if nested).
        # Filtered-out TEST pairs are scored at the train base rate below.
        filter_kw = dict(ct_col=ct_col, per_ct_genes=per_ct_genes)
        df_train, _ = filter_missing_rows(df_train, gene_columns, embed_keys, **filter_kw)
        df_val, _ = filter_missing_rows(df_val, gene_columns, embed_keys, **filter_kw)
        df_test, df_test_missing = filter_missing_rows(df_test, gene_columns, embed_keys, **filter_kw)
        base_rate = df_train[label_col].mean()
        print(f"  After filtering — Train: {len(df_train)}, Val: {len(df_val)}, "
              f"Test: {len(df_test)} present + {len(df_test_missing)} missing")
        print(f"  Base rate (train): {base_rate:.4f}")

        metrics = run_fold(df_train, df_val, df_test, embeds, fold_cfg, device,
                           ct_col=use_ct_col, embeds_nested=embeds_nested,
                           task_type=task_type, train_desc=fold_col)
        metrics['fold'] = fold_col
        y_true = metrics.pop('y_true')
        y_prob = metrics.pop('y_prob')
        n_present = len(y_true)

        if len(df_test_missing) > 0:
            y_true_missing = df_test_missing[label_col].values.astype(np.float32)
            y_prob_missing = np.full(len(df_test_missing), base_rate, dtype=np.float32)
            y_true = np.concatenate([y_true, y_true_missing])
            y_prob = np.concatenate([y_prob, y_prob_missing])
            df_test = pd.concat([df_test, df_test_missing], ignore_index=True)
            if task_type == 'regression':
                metrics.update(compute_regression_metrics(y_true, y_prob))
            else:
                metrics.update(compute_metrics(y_true, y_prob))

        if task_type == 'regression':
            print(f"  [{cfg.embed_name}] All (n={len(y_true)}): Pearson={metrics['pearson']:.4f}, "
                  f"Spearman={metrics['spearman']:.4f}, MSE={metrics['mse']:.4f}")
        else:
            print(f"  [{cfg.embed_name}] All (n={len(y_true)}): AUROC={metrics['auroc']:.4f}, "
                  f"AUPRC={metrics['auprc']:.4f}, F1={metrics['f1']:.4f}")

        # Pooled row + one row per cell line
        ct_indices = {}
        if use_ct_col:
            metrics['cell_line'] = 'all'
            metrics['n_samples'] = len(y_true)
            all_results.append(metrics)
            all_row_idx = len(all_results) - 1

            ct_values = df_test[use_ct_col].values
            for ct in sorted(df_test[use_ct_col].unique()):
                mask = ct_values == ct
                if mask.sum() == 0:
                    continue
                if task_type == 'regression':
                    ct_metrics = compute_regression_metrics(y_true[mask], y_prob[mask])
                else:
                    ct_metrics = compute_metrics(y_true[mask], y_prob[mask])
                ct_metrics['fold'] = fold_col
                ct_metrics['cell_line'] = ct
                ct_metrics['n_samples'] = int(mask.sum())
                for hp in ('lr', 'batch_size', 'weight_decay', 'dropout',
                           'hidden_dim', 'n_layers', 'val_loss'):
                    if hp in metrics:
                        ct_metrics[hp] = metrics[hp]
                all_results.append(ct_metrics)
                ct_indices[ct] = len(all_results) - 1
                if task_type == 'regression':
                    print(f"  [{cfg.embed_name}] {ct} (n={mask.sum()}): Pearson={ct_metrics['pearson']:.4f}, "
                          f"Spearman={ct_metrics['spearman']:.4f}, MSE={ct_metrics['mse']:.4f}")
                else:
                    print(f"  [{cfg.embed_name}] {ct} (n={mask.sum()}): AUROC={ct_metrics['auroc']:.4f}, "
                          f"AUPRC={ct_metrics['auprc']:.4f}, F1={ct_metrics['f1']:.4f}")
        else:
            metrics['n_samples'] = len(y_true)
            all_results.append(metrics)
            all_row_idx = len(all_results) - 1

        # Predictions (present rows first, then base-rate-imputed rows)
        preds_df = df_test[gene_columns].copy()
        if ct_col and ct_col in df.columns:
            preds_df[ct_col] = df_test[ct_col].values
        preds_df['y_true'] = y_true
        preds_df['y_prob'] = y_prob
        preds_df['fold'] = fold_col
        preds_df['lr'] = metrics['lr']
        preds_df['batch_size'] = metrics['batch_size']
        preds_df['weight_decay'] = metrics['weight_decay']
        preds_df['source'] = ['model'] * n_present + ['imputed'] * (len(y_true) - n_present)
        all_predictions.append(preds_df)

        # Macro (per-gene) metrics — classification only
        if task_type != 'regression':
            macro = compute_macro_metrics(preds_df, gene_columns)
            for col in gene_columns:
                print(f"  Macro-{col} (n={macro[f'n_{col}_evaluated']}): "
                      f"AUROC={macro[f'macro_{col}_auroc']:.4f}, AUPRC={macro[f'macro_{col}_auprc']:.4f}")
            macro_pooled = compute_macro_metrics_pooled(preds_df, gene_columns)
            macro.update(macro_pooled)
            print(f"  Macro-pooled (n={macro_pooled['n_pooled_evaluated']}): "
                  f"AUROC={macro_pooled['macro_pooled_auroc']:.4f}, AUPRC={macro_pooled['macro_pooled_auprc']:.4f}")
            for key in macro:
                if key.startswith('macro_'):
                    all_results[all_row_idx][key] = macro[key]

            for ct, ct_idx in ct_indices.items():
                ct_preds = preds_df[preds_df[use_ct_col] == ct]
                if len(ct_preds) == 0:
                    continue
                ct_macro = compute_macro_metrics(ct_preds, gene_columns)
                ct_macro.update(compute_macro_metrics_pooled(ct_preds, gene_columns))
                for key in ct_macro:
                    if key.startswith('macro_'):
                        all_results[ct_idx][key] = ct_macro[key]

    # Summary
    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    if task_type == 'regression':
        summary_cols = ['pearson', 'spearman', 'mse']
    else:
        summary_cols = ['auroc', 'auprc', 'f1', 'precision', 'recall', 'accuracy']
    summary_cols = [c for c in summary_cols if c in results_df.columns]
    if use_ct_col:
        for cl in results_df['cell_line'].unique():
            cl_df = results_df[results_df['cell_line'] == cl]
            print(f"\n{cl}:")
            for col in summary_cols:
                print(f"  {col.upper():10s}: {cl_df[col].mean():.4f} +/- {cl_df[col].std():.4f}")
    else:
        for col in summary_cols:
            print(f"{col.upper():10s}: {results_df[col].mean():.4f} +/- {results_df[col].std():.4f}")
    macro_metric_cols = [c for c in results_df.columns if c.startswith('macro_')]
    if macro_metric_cols:
        print("\nMacro Metrics (averaged across folds):")
        for col in sorted(macro_metric_cols):
            print(f"  {col:30s}: {results_df[col].mean():.4f} +/- {results_df[col].std():.4f}")

    # Save
    results_dir = Path(cfg.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"{cfg.embed_name}{cfg.get('name_suffix', '') or ''}"
    summary_path = results_dir / f"{save_name}_mlp_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    preds_df = pd.concat(all_predictions, ignore_index=True)
    preds_path = results_dir / f"{save_name}_mlp_preds.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path}")

    # Residual GI metrics (signed-qGI regression): main-effect-corrected
    # AUPRC/AUROC per arm on the test predictions.
    if task_type == 'regression' and cfg.get('residual_metrics', False):
        from ember.eval.gi_residual import run as run_residual
        print("\n" + "=" * 60 + "\nRESIDUAL GI METRICS\n" + "=" * 60)
        run_residual(preds_df, df, results_dir / f"{save_name}_mlp_residual_metrics.csv",
                     lib_col=gene_columns[0], query_col=gene_columns[1])

    return summary_path, preds_path

