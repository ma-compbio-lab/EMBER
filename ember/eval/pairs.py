"""Gene-pair prediction over frozen gene embeddings.

Every row of a task CSV names two genes (a perturbation and a readout gene, or
a library and a query gene) plus a label. The two gene vectors are looked up
in an embedding pickle, concatenated, and fed to a small MLP that is trained
per fold with a hyperparameter sweep. Supports binary classification
(BCE-with-logits, AUROC/AUPRC/...) and regression (MSE, Pearson/Spearman).

Embedding pickles are either flat ``{gene: vector}`` or nested
``{cell_line: {gene: vector}}``; nested pickles are indexed by each row's
``ct_col`` value so every cell line uses its own contextual embedding.
"""

import random
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score,
                             mean_squared_error)
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Reproducibility / embedding helpers
# ---------------------------------------------------------------------------

def seed_everything(seed):
    """Make a run reproducible (weight init, shuffling, cuda)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_nested_embeds(embeds):
    """True if embeddings are cell-line specific: embeds[cell_line][gene]."""
    first_val = next(iter(embeds.values()))
    return isinstance(first_val, dict)


def get_embedding(embeds, gene, cell_type, embeds_nested):
    """Look up one gene vector, optionally within a cell line."""
    if embeds_nested:
        return embeds[cell_type][gene]
    return embeds[gene]


def filter_missing_rows(df, gene_columns, genes_present, ct_col=None, per_ct_genes=None):
    """Split df into (rows whose genes all have embeddings, rows that do not).

    With nested embeddings (`per_ct_genes`: cell_line -> set of genes) each row
    is checked against its own cell line's gene set; otherwise against the
    global `genes_present` set.
    """
    if per_ct_genes and ct_col and ct_col in df.columns:
        mask = pd.Series(True, index=df.index)
        for col in gene_columns:
            col_mask = pd.Series(False, index=df.index)
            for ct, ct_genes in per_ct_genes.items():
                ct_rows = df[ct_col] == ct
                col_mask |= ct_rows & df[col].isin(ct_genes)
            mask &= col_mask
    else:
        mask = pd.Series(True, index=df.index)
        for col in gene_columns:
            mask &= df[col].isin(genes_present)
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)


def prepare_data(df, embeds, gene_columns, label_col, ct_col=None, embeds_nested=False):
    """Build X (N, 2, D) float32 and y (N,) float32 in dataframe row order."""
    X, y = [], []
    for _, row in df.iterrows():
        ct = row[ct_col] if ct_col else None
        X.append(np.stack([
            get_embedding(embeds, row[col], ct, embeds_nested)
            for col in gene_columns
        ]))
        y.append(row[label_col])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_loader(X, y, batch_size, shuffle=False):
    """DataLoader over in-memory numpy arrays."""
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPConcat(nn.Module):
    """[GeneA || GeneB] -> (Linear, ReLU, [Dropout]) x n_layers -> Linear(1).

    Dropout layers are only inserted when p > 0, so the parameter layout (and
    hence the RNG stream used at init) is identical to a dropout-free MLP.
    """

    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim * 2
        for _ in range(n_layers):
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 2, dim)
        x = x.view(x.size(0), -1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Validation criterion for signed-GI regression
# ---------------------------------------------------------------------------

# Residual-AUPRC thresholds. MUST match ember/eval/gi_residual.py.
RESID_POS_QGI = 0.3
RESID_POS_FDR = 0.1
RESID_NEG_QGI = 0.05
RESID_NEG_FDR = 0.5


def residual_auprc_any(lib, query, qgi, fdr, pred):
    """ANY-arm residual AUPRC used to select regression models on the val set.

    Within the val set, remove per-library and per-query main effects with the
    two-way additive residual
        R = pred - mean_lib(pred) - mean_query(pred) + grand_mean
    then score |R| against the ANY label
        positive = any significant GI (|qGI| > 0.3 & FDR < 0.1)
        negative = confident zero     (|qGI| < 0.05 & FDR > 0.5)
    Ambiguous rows are excluded. Returns nan if either class is empty (the
    trainer treats nan as "no improvement").
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    qgi = np.asarray(qgi, dtype=np.float64).ravel()
    fdr = np.asarray(fdr, dtype=np.float64).ravel()

    def _group_mean(keys):
        _, inv = np.unique(np.asarray(keys), return_inverse=True)
        sums = np.bincount(inv, weights=pred)
        counts = np.bincount(inv)
        return (sums / counts)[inv]

    resid = pred - _group_mean(lib) - _group_mean(query) + pred.mean()

    pos = (np.abs(qgi) > RESID_POS_QGI) & (fdr < RESID_POS_FDR)
    neg = (np.abs(qgi) < RESID_NEG_QGI) & (fdr > RESID_NEG_FDR)
    mask = pos | neg
    if int(pos.sum()) < 1 or int(neg.sum()) < 1:
        return float('nan')
    labels = pos[mask].astype(int)
    scores = np.abs(resid)[mask]
    return float(average_precision_score(labels, scores))


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, *, lr, weight_decay, max_epochs,
                patience, tol, device, task_type='classification',
                val_selection_criteria='loss', optimizer_name='adam',
                val_residual_ctx=None, desc='Training'):
    """Train with early stopping on a validation criterion.

    val_selection_criteria:
        'loss'               -> validation loss (lower is better)  [default]
        'auroc' / 'auprc'    -> classification metric on val (higher is better)
        'residual_auprc_any' -> regression; needs `val_residual_ctx`, a dict
                                with 'lib', 'query', 'qgi', 'fdr' arrays aligned
                                to the (unshuffled) val loader.
    Returns (model restored to its best epoch, best val score).
    """
    opt_name = str(optimizer_name).lower()
    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name!r} (expected 'adam' or 'adamw')")

    criterion = nn.MSELoss() if task_type == 'regression' else nn.BCEWithLogitsLoss()

    vsc = str(val_selection_criteria).lower()
    use_metric = vsc in ('auroc', 'auprc') and task_type == 'classification'
    use_residual_auprc = vsc == 'residual_auprc_any' and task_type == 'regression'
    if use_residual_auprc and val_residual_ctx is None:
        raise ValueError("val_selection_criteria='residual_auprc_any' requires val_residual_ctx")
    higher_is_better = use_metric or use_residual_auprc

    best_val_score = float('-inf') if higher_is_better else float('inf')
    best_state = None
    patience_counter = 0

    epoch_pbar = tqdm(range(max_epochs), desc=desc, leave=False)
    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            optimizer.zero_grad()
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_preds_list, val_labels_list = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
                val_preds_list.append(pred.cpu())
                val_labels_list.append(y_batch.cpu())
        val_loss /= len(val_loader)

        val_preds_all = torch.cat(val_preds_list).numpy().flatten()
        val_labels_all = torch.cat(val_labels_list).numpy().flatten()
        if use_metric:
            val_probs = 1.0 / (1.0 + np.exp(-val_preds_all))  # sigmoid
            if vsc == 'auroc':
                val_score = roc_auc_score(val_labels_all, val_probs)
            else:
                val_score = average_precision_score(val_labels_all, val_probs)
        elif use_residual_auprc:
            try:
                val_score = residual_auprc_any(
                    val_residual_ctx['lib'], val_residual_ctx['query'],
                    val_residual_ctx['qgi'], val_residual_ctx['fdr'],
                    val_preds_all.astype(np.float64))
            except Exception as e:
                print(f'  [warn] residual_auprc_any computation failed: {e}')
                val_score = float('nan')
        else:
            val_score = val_loss

        postfix = {'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'}
        if higher_is_better:
            postfix[f'val_{vsc}'] = f'{val_score:.4f}'
        epoch_pbar.set_postfix(postfix)
        epoch_pbar.update(1)

        # Early stopping (a nan score never counts as an improvement)
        if higher_is_better:
            improved = val_score > best_val_score + tol
        else:
            improved = val_score < best_val_score - tol
        if improved:
            best_val_score = val_score
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if not improved and patience_counter >= patience:
            break

    epoch_pbar.close()

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        del best_state

    return model, best_val_score


def compute_metrics(y_true, y_prob):
    """Binary classification metrics (threshold 0.5 for the hard metrics)."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }


def compute_regression_metrics(y_true, y_pred):
    """Regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        'pearson': pearsonr(y_true, y_pred)[0],
        'spearman': spearmanr(y_true, y_pred)[0],
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
    }


def evaluate_model(model, loader, device, task_type='classification'):
    """Metrics dict plus 'y_true' / 'y_prob' arrays in loader order.

    Classification returns sigmoid probabilities; regression raw outputs.
    """
    model.eval()
    y_true, raw_outs = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            raw_outs.append(model(X_batch.to(device)).cpu().numpy())
            y_true.extend(y_batch.numpy())

    y_true = np.array(y_true)
    y_pred = np.concatenate(raw_outs, axis=0) if raw_outs else np.array([])
    if task_type == 'classification':
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
        metrics = compute_metrics(y_true, y_pred)
    else:
        metrics = compute_regression_metrics(y_true, y_pred)
    metrics['y_true'] = y_true
    metrics['y_prob'] = y_pred
    return metrics


def run_fold(df_train, df_val, df_test, embeds, cfg, device, ct_col=None,
             embeds_nested=False, task_type='classification', train_desc='Training'):
    """Hyperparameter sweep for one fold; returns test metrics of the best val model.

    `cfg` is a plain dict with: gene_columns, label_col, hidden_dim, n_layers,
    max_epochs, patience, tol, sweep (lr/batch_size/weight_decay[/dropout/
    hidden_dim/n_layers] lists), val_selection_criteria, optimizer,
    residual_fdr_col.
    """
    gene_columns = cfg['gene_columns']
    label_col = cfg['label_col']
    data_kwargs = dict(gene_columns=gene_columns, label_col=label_col,
                       ct_col=ct_col, embeds_nested=embeds_nested)
    X_train, y_train = prepare_data(df_train, embeds, **data_kwargs)
    X_val, y_val = prepare_data(df_val, embeds, **data_kwargs)
    X_test, y_test = prepare_data(df_test, embeds, **data_kwargs)
    input_dim = X_train.shape[-1]

    val_selection_criteria = cfg.get('val_selection_criteria', 'loss')
    vsc = str(val_selection_criteria).lower()
    use_metric = vsc in ('auroc', 'auprc') and task_type == 'classification'
    use_residual_auprc = vsc == 'residual_auprc_any' and task_type == 'regression'
    higher_is_better = use_metric or use_residual_auprc
    criteria_label = f'val_{val_selection_criteria}' if higher_is_better else 'val_loss'

    # Per-val residual context (library/query ids + qGI + FDR), aligned to
    # df_val row order == X_val == the unshuffled val loader order.
    val_residual_ctx = None
    if use_residual_auprc:
        if not gene_columns or len(gene_columns) < 2:
            raise ValueError("val_selection_criteria='residual_auprc_any' needs "
                             "gene_columns=[library_gene, query_gene].")
        fdr_col = cfg.get('residual_fdr_col', 'FDR')
        missing = [c for c in (gene_columns[0], gene_columns[1], label_col, fdr_col)
                   if c not in df_val.columns]
        if missing:
            raise ValueError(f"residual_auprc_any: df_val missing columns {missing}")
        val_residual_ctx = {
            'lib': df_val[gene_columns[0]].to_numpy(),
            'query': df_val[gene_columns[1]].to_numpy(),
            'qgi': df_val[label_col].to_numpy(dtype=np.float64),
            'fdr': df_val[fdr_col].to_numpy(dtype=np.float64),
        }

    best_val_score = float('-inf') if higher_is_better else float('inf')
    best_params = None
    best_metrics = None

    sweep = cfg['sweep']
    sweep_params = list(product(
        sweep['lr'], sweep['batch_size'], sweep['weight_decay'],
        sweep.get('dropout', [0.0]),
        sweep.get('hidden_dim', [cfg['hidden_dim']]),
        sweep.get('n_layers', [cfg['n_layers']]),
    ))
    print(f"  Sweeping {len(sweep_params)} hyperparameter combinations...")

    for lr, batch_size, weight_decay, dropout, hidden_dim, n_layers in sweep_params:
        train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = make_loader(X_val, y_val, batch_size)

        model = MLPConcat(input_dim, hidden_dim, n_layers, dropout=dropout).to(device)
        model, val_score = train_model(
            model, train_loader, val_loader,
            lr=lr, weight_decay=weight_decay,
            max_epochs=cfg['max_epochs'], patience=cfg['patience'], tol=cfg['tol'],
            device=device, task_type=task_type, desc=train_desc,
            val_selection_criteria=val_selection_criteria,
            optimizer_name=cfg.get('optimizer', 'adam'),
            val_residual_ctx=val_residual_ctx,
        )

        # Strict comparison: ties keep the earlier combination.
        is_new_best = (val_score > best_val_score) if higher_is_better else (val_score < best_val_score)
        if is_new_best:
            best_val_score = val_score
            best_params = {'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay,
                           'dropout': dropout, 'hidden_dim': hidden_dim, 'n_layers': n_layers}
            test_loader = make_loader(X_test, y_test, batch_size)
            best_metrics = evaluate_model(model, test_loader, device, task_type=task_type)
            best_metrics.update(best_params)
            best_metrics['val_loss'] = val_score

        status = "NEW BEST" if is_new_best else ""
        print(f"    lr={lr}, bs={batch_size}, wd={weight_decay}, drop={dropout}, "
              f"hidden={hidden_dim}, layers={n_layers} -> {criteria_label}={val_score:.4f} {status}")

    if best_params is None:
        raise RuntimeError("No hyperparameter combination produced a finite validation "
                           f"score ({criteria_label}); cannot select a model for this fold.")
    print(f"  Best: {criteria_label}={best_val_score:.4f}, lr={best_params['lr']}, "
          f"bs={best_params['batch_size']}, wd={best_params['weight_decay']}, "
          f"hidden={best_params['hidden_dim']}, layers={best_params['n_layers']}")

    return best_metrics


# ---------------------------------------------------------------------------
# Macro (per-gene) metrics for classification
# ---------------------------------------------------------------------------

def compute_macro_metrics(preds_df, gene_columns):
    """AUROC/AUPRC averaged over the unique values of each gene column."""
    results = {}
    for col in gene_columns:
        if col not in preds_df.columns:
            continue
        aurocs, auprcs = [], []
        for val in preds_df[col].unique():
            mask = preds_df[col] == val
            y_true = preds_df.loc[mask, 'y_true'].values
            y_prob = preds_df.loc[mask, 'y_prob'].values
            if len(np.unique(y_true)) < 2:
                continue
            aurocs.append(roc_auc_score(y_true, y_prob))
            auprcs.append(average_precision_score(y_true, y_prob))

        results[f'macro_{col}_auroc'] = np.mean(aurocs) if aurocs else np.nan
        results[f'macro_{col}_auprc'] = np.mean(auprcs) if auprcs else np.nan
        results[f'n_{col}_evaluated'] = len(aurocs)
    return results


def compute_macro_metrics_pooled(preds_df, gene_columns):
    """AUROC/AUPRC averaged over unique genes pooled across all gene columns.

    For each gene, every row where it appears in ANY column is gathered.
    """
    from collections import defaultdict
    gene_rows = defaultdict(set)
    for col in gene_columns:
        if col not in preds_df.columns:
            continue
        for idx, gene in enumerate(preds_df[col].values):
            gene_rows[gene].add(idx)

    y_true = preds_df['y_true'].values
    y_prob = preds_df['y_prob'].values

    aurocs, auprcs = [], []
    for gene, indices in gene_rows.items():
        indices = sorted(indices)
        yt = y_true[indices]
        yp = y_prob[indices]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, yp))
        auprcs.append(average_precision_score(yt, yp))

    return {
        'macro_pooled_auroc': np.mean(aurocs) if aurocs else np.nan,
        'macro_pooled_auprc': np.mean(auprcs) if auprcs else np.nan,
        'n_pooled_evaluated': len(aurocs),
    }
