"""Within-cell-line drug-response regression on UNSEEN drugs (LINCS / Tahoe).

For one cell line, predicts the DESeq2 log2FC of every (drug, gene) pair for
drugs held out by target-aware 10-fold splits. A drug is a DeepSet over the
contextual embeddings of its annotated target genes (+ mechanism of action);
the readout gene is its own contextual embedding; both come from embeddings
built on that cell line's control cells. Loss = MSE on observed cells.

Reported per fold (and averaged in a ``fold='all'`` row):
  * ``per_drug_pearson``        -- mean over held-out drugs of Pearson(truth, pred)
                                   across the drug's observed genes
  * ``per_drug_pearson_delta``  -- same after subtracting the per-gene TRAIN-mean
                                   LFC from both vectors. Raw per-drug Pearson is
                                   inflated by the average drug response (genes
                                   that move under any perturbation); the delta
                                   metric measures drug-specific signal above
                                   that train-mean floor.
  * pooled ``pearson`` / ``spearman`` / ``rmse`` over all test cells.

Entry point: ``evaluate.py`` dispatches here for configs with ``evaluator: drug``.

    python evaluate.py --config-name lincs_unseen_drugs cell_line=a549 \
        embed_name=ember_base_lincs
    python evaluate.py --config-name tahoe_unseen_drugs cell_line=sw480 \
        embed_path=/abs/path/to/embeds.pkl +folds=[1]

Embedding pickle contract: ``embeds[cell_line_key][gene_symbol] -> float32[D]``
where ``cell_line_key`` is the split's ``ct_label`` (DepMap ACH id for LINCS,
Cellosaurus CVCL id for Tahoe). A flat ``{gene_symbol: vector}`` pickle is also
accepted (same embedding for every line).
"""

import pickle
import random
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from ember.eval.drug_lfc import (
    DrugGeneDeepSet, LFCAtlas, LFCDrugGeneDataset, compute_regression_metrics,
    load_drug_targets, load_split_spec, per_group_corr, split_obs_for_fold,
)


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _is_nested(embeds):
    return isinstance(next(iter(embeds.values())), dict)


def _forward_batch(model, batch, device):
    drug_x, mask, gene, moa, y = batch
    out = model(drug_x.to(device), mask.to(device), gene.to(device), moa_ids=moa.to(device))
    return out, y


def _infer(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            out, _ = _forward_batch(model, batch, device)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)


def _delta_mask(base):
    """Cells whose gene has a finite train mean; the rest are dropped from the
    delta metric, never imputed."""
    return np.isfinite(base) if base is not None else None


def _score(sel, obs, y_true, y_pred, base=None, mask=None):
    """Validation score (higher is better) for early stopping / HP selection."""
    if sel == 'pearson_per_group_delta':
        score = per_group_corr(obs[mask], y_true[mask] - base[mask], y_pred[mask] - base[mask], min_n=5)[0]
    elif sel == 'pearson_per_group':
        score = per_group_corr(obs, y_true, y_pred, min_n=5)[0]
    else:  # 'mse' -> negative MSE
        return -float(np.mean((y_true - y_pred) ** 2))
    return score if np.isfinite(score) else -np.inf


def _train(model, train_loader, val_loader, val_obs, val_y, lr, wd, max_epochs, patience,
           tol, device, sel, val_base=None, optimizer='adam'):
    if optimizer == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    best_score, best_state, bad = -np.inf, None, 0
    dmask = _delta_mask(val_base)
    pbar = tqdm(range(max_epochs), desc=f'lr={lr} wd={wd}', leave=False, unit='ep')
    for _ in pbar:
        model.train()
        for batch in train_loader:
            out, y = _forward_batch(model, batch, device)
            loss = loss_fn(out, y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        yp = _infer(model, val_loader, device)
        score = _score(sel, val_obs, val_y, yp, base=val_base, mask=dmask)
        if score > best_score + tol:
            best_score, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        pbar.set_postfix({'val': f'{score:.4f}', 'best': f'{best_score:.4f}', 'patience': f'{bad}/{patience}'})
        if bad >= patience:
            break
    pbar.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def _filter_cells(atlas, obs_arr, gene_arr, mode, drug_targets, morgan_vecs, embeds, embeds_nested):
    """Keep cells whose drug is usable and (targets mode) whose gene has an embedding
    in that cell's line."""
    if len(obs_arr) == 0:
        return np.zeros(0, dtype=bool)
    drugs = atlas.pert_per_obs[obs_arr.astype(np.int64)]
    if mode == 'morgan':
        return np.array([d in morgan_vecs for d in drugs], dtype=bool)
    keep = np.array([d in drug_targets for d in drugs], dtype=bool)
    cts = atlas.ct_per_obs[obs_arr.astype(np.int64)]
    genes = atlas.gene_names[gene_arr.astype(np.int64)]
    if embeds_nested:
        gene_ok = np.array([g in embeds.get(ct, {}) for ct, g in zip(cts, genes)], dtype=bool)
    else:
        gene_ok = np.array([g in embeds for g in genes], dtype=bool)
    return keep & gene_ok


def run(cfg: DictConfig):
    """Run all folds for one (config, embedding) pair and write the CSVs."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = cfg.get('drug_repr_mode', 'targets')
    base_seed = int(cfg.get('seed', 42))
    seed_everything(base_seed)
    print(f'drug-lfc | mode={mode} | device={device} | seed={base_seed}')

    hvg = Path(cfg.hvg_path).read_text().strip().split('\n')
    atlas = LFCAtlas(cfg.atlas_path, gene_allowlist=hvg, use_padj=cfg.get('use_padj', False),
                     obs_ct_column='cell_line', obs_pert_column='broad_id')

    embeds = pickle.load(open(cfg.embed_path, 'rb'))
    embeds_nested = _is_nested(embeds)
    if mode == 'morgan':
        morgan_vecs = embeds
        embed_dim = len(next(iter(morgan_vecs.values())))
        drug_targets = drug_moa = {}
    else:
        morgan_vecs = None
        if embeds_nested:
            embed_genes = set().union(*(set(d.keys()) for d in embeds.values()))
            embed_dim = len(next(iter(next(iter(embeds.values())).values())))
        else:
            embed_genes = set(embeds.keys())
            embed_dim = len(next(iter(embeds.values())))
        drug_targets, drug_moa, drug_drop = load_drug_targets(cfg.drug_lookup, embed_genes)
        print(f'  drugs usable: {len(drug_targets)}, dropped (>1/2 targets missing): {len(drug_drop)}')

    spec = load_split_spec(cfg.split_spec)
    n_folds = spec['n_folds']
    core_cts = list(spec['core_training_cts'].values())
    print(f'  holdout: {spec["ct_label"]} ({spec.get("cell_name")}); core: {core_cts}; folds={n_folds}')

    max_targets = cfg.get('max_targets', 25)
    sweep = list(product(cfg.sweep.lr, cfg.sweep.batch_size, cfg.sweep.weight_decay,
                         list(cfg.sweep.get('dropout', [0.0]))))
    sel = cfg.get('val_selection_criteria', 'pearson_per_group')
    nw = int(cfg.get('num_workers', 0))

    def make_ds(obs_arr, gene_arr):
        return LFCDrugGeneDataset(atlas, obs_arr, gene_arr, embeds, embeds_nested,
                                  drug_targets=drug_targets, drug_moa=drug_moa,
                                  max_targets=max_targets, mode=mode,
                                  morgan_vecs=morgan_vecs, embed_dim=embed_dim)

    folds_override = cfg.get('folds', None)
    folds_to_run = [int(f) for f in folds_override] if folds_override else list(range(1, n_folds + 1))

    summary, preds_rows = [], []
    for fold in folds_to_run:
        print(f'\n=== Fold {fold}/{n_folds} (running {folds_to_run}) ===')
        obs = split_obs_for_fold(atlas, spec, fold, core_cts)
        arrs = {r: atlas.build_split_arrays(obs[r]) for r in ('train', 'val', 'test')}

        # Filter to usable cells; keep dropped TEST cells for imputation.
        kept, dropped = {}, {}
        for r in ('train', 'val', 'test'):
            oa, ga = arrs[r]
            keep = _filter_cells(atlas, oa, ga, mode, drug_targets, morgan_vecs, embeds, embeds_nested)
            kept[r] = (oa[keep], ga[keep])
            if r == 'test':
                dropped[r] = (oa[~keep], ga[~keep])

        # Per-gene TRAIN-mean LFC: the imputation value for dropped test cells and
        # the baseline for the delta metric. Train-only, so val/test never inform it.
        tr_oa, tr_ga = kept['train']
        gmean = np.full(atlas.n_genes, np.nan, dtype=np.float64)
        if len(tr_oa):
            yv = atlas.lfc[tr_oa.astype(np.int64), tr_ga.astype(np.int64)].astype(np.float64)
            order = np.argsort(tr_ga, kind='stable')
            g_s, y_s = tr_ga[order], yv[order]
            uniq, start = np.unique(g_s, return_index=True)
            sums = np.add.reduceat(y_s, start)
            cnts = np.diff(np.append(start, len(y_s)))
            gmean[uniq] = sums / cnts
        global_mean = float(np.nanmean(gmean)) if np.isfinite(gmean).any() else 0.0

        n_drop = len(dropped['test'][0])
        print(f'  train={len(tr_oa):,}  val={len(kept["val"][0]):,}  '
              f'test={len(kept["test"][0]):,} (+{n_drop:,} imputed)')
        if len(tr_oa) == 0 or len(kept['val'][0]) == 0:
            print('  empty split, skipping'); continue

        ds_tr, ds_val, ds_test = make_ds(*kept['train']), make_ds(*kept['val']), make_ds(*kept['test'])
        val_obs = kept['val'][0].astype(np.int64)
        val_y = atlas.lfc[val_obs, kept['val'][1].astype(np.int64)].astype(np.float64)
        test_obs = kept['test'][0].astype(np.int64)
        test_y = atlas.lfc[test_obs, kept['test'][1].astype(np.int64)].astype(np.float64)
        val_base = gmean[kept['val'][1].astype(np.int64)]

        best = {'score': -np.inf}
        for ci, (lr, bs, wd, dr) in enumerate(sweep):
            # Deterministic per (fold, HP combo): identical results on re-run.
            seed_everything(base_seed + fold * 100 + ci)
            g = torch.Generator().manual_seed(base_seed + fold * 100 + ci)
            tr_loader = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=nw, generator=g)
            val_loader = DataLoader(ds_val, batch_size=bs, num_workers=nw)
            model = DrugGeneDeepSet(
                embed_dim, cfg.hidden_dim, cfg.n_layers, dropout=dr, mode=mode,
                n_moa_classes=3 if cfg.get('use_moa', True) else 0,
                moa_embed_dim=cfg.get('moa_embed_dim', 8),
                drug_input_dim=embed_dim if mode == 'morgan' else None,
                n_genes=atlas.n_genes, gene_embed_dim=cfg.get('gene_embed_dim', 128),
                drug_proj_layers=cfg.get('drug_proj_layers', 1)).to(device)
            model, vscore = _train(model, tr_loader, val_loader, val_obs, val_y, lr, wd,
                                   cfg.max_epochs, cfg.patience, cfg.tol, device, sel,
                                   val_base=val_base, optimizer=cfg.get('optimizer', 'adam'))
            test_loader = DataLoader(ds_test, batch_size=bs, num_workers=nw)
            yp = _infer(model, test_loader, device)
            if vscore > best['score']:
                best = {'score': vscore, 'lr': lr, 'bs': bs, 'wd': wd, 'dr': dr, 'yp': yp,
                        'val_yp': _infer(model, val_loader, device)}
            print(f'    lr={lr} bs={bs} wd={wd} dr={dr} -> val_{sel}={vscore:.4f}')

        # Assemble the full test set: model predictions + imputed dropped cells
        # (per-gene train mean, global train mean if the gene has none; never 0).
        yp = best['yp']
        obs_full, y_full, p_full, src = test_obs.copy(), test_y.copy(), yp.copy(), ['model'] * len(yp)
        if n_drop:
            do, dg = dropped['test'][0].astype(np.int64), dropped['test'][1].astype(np.int64)
            y_imp = atlas.lfc[do, dg].astype(np.float64)
            p_imp = np.where(np.isfinite(gmean[dg]), gmean[dg], global_mean)
            obs_full = np.concatenate([obs_full, do])
            y_full = np.concatenate([y_full, y_imp])
            p_full = np.concatenate([p_full, p_imp])
            src += ['imputed'] * len(do)
        gene_full = np.concatenate([kept['test'][1], dropped['test'][1]]) if n_drop else kept['test'][1]
        # Every observed test cell must appear exactly once (model or imputed).
        assert len(gene_full) == len(arrs['test'][1]), (
            f"fold {fold}: assembled {len(gene_full)} test cells but atlas has "
            f"{len(arrs['test'][1])} valid -- prediction cells were lost")
        if n_drop:
            imp_drugs = np.unique(atlas.pert_per_obs[dropped['test'][0].astype(np.int64)])
            print(f'  [coverage] {n_drop:,}/{len(gene_full):,} test cells IMPUTED '
                  f'(embedding lacks them); drugs affected: {len(imp_drugs)}')

        per_drug = per_group_corr(obs_full, y_full, p_full, min_n=5)
        reg = compute_regression_metrics(y_full, p_full)
        # Delta metric: cells with no finite train-mean baseline are dropped. Imputed
        # cells predict gmean exactly (delta pred 0), so a fully-imputed drug becomes
        # constant-pred and is skipped, as it should be.
        test_base = gmean[gene_full.astype(np.int64)]
        dm = _delta_mask(test_base)
        per_drug_d = per_group_corr(obs_full[dm], y_full[dm] - test_base[dm], p_full[dm] - test_base[dm], min_n=5)
        print(f'  Best HP lr={best["lr"]} bs={best["bs"]} wd={best["wd"]} dr={best["dr"]} '
              f'-> per-drug Pearson={per_drug[0]:.4f} (n={per_drug[2]})  '
              f'delta={per_drug_d[0]:.4f} (n={per_drug_d[2]})  '
              f'pooled Pearson={reg["pearson"]:.4f} RMSE={reg["rmse"]:.4f}')
        summary.append({'fold': f'split_{fold}', 'cell_line': spec['ct_label'],
                        'per_drug_pearson': per_drug[0], 'per_drug_spearman': per_drug[1],
                        'per_drug_pearson_delta': per_drug_d[0], 'per_drug_spearman_delta': per_drug_d[1],
                        'n_drugs_delta': per_drug_d[2], 'val_selection_criteria': sel,
                        'n_drugs': per_drug[2], 'pearson': reg['pearson'], 'spearman': reg['spearman'],
                        'rmse': reg['rmse'], 'n_samples': len(y_full), 'n_imputed': n_drop,
                        'lr': best['lr'], 'batch_size': best['bs'], 'weight_decay': best['wd'],
                        'dropout': best['dr']})
        for k in range(len(y_full)):
            oi = int(obs_full[k])
            preds_rows.append({'fold': f'split_{fold}', 'cell_line': atlas.ct_per_obs[oi],
                               'broad_id': atlas.pert_per_obs[oi],
                               'gene': atlas.gene_names[int(gene_full[k])],
                               'y_true': float(y_full[k]), 'y_pred': float(p_full[k]), 'source': src[k]})

    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    name = f'{cfg.embed_name}{cfg.get("name_suffix", "") or ""}_{mode}'
    sdf = pd.DataFrame(summary)
    if len(sdf):
        allrow = {'fold': 'all', 'cell_line': spec['ct_label'],
                  'per_drug_pearson': sdf['per_drug_pearson'].mean(),
                  'per_drug_spearman': sdf['per_drug_spearman'].mean(),
                  'per_drug_pearson_delta': sdf['per_drug_pearson_delta'].mean(),
                  'per_drug_spearman_delta': sdf['per_drug_spearman_delta'].mean(),
                  'val_selection_criteria': sel,
                  'pearson': sdf['pearson'].mean(), 'spearman': sdf['spearman'].mean(),
                  'rmse': sdf['rmse'].mean()}
        sdf = pd.concat([sdf, pd.DataFrame([allrow])], ignore_index=True)
    sdf.to_csv(out_dir / f'{name}_summary.csv', index=False)
    pd.DataFrame(preds_rows).to_csv(out_dir / f'{name}_preds.csv', index=False)
    print(f'\nSaved: {out_dir}/{name}_summary.csv')

