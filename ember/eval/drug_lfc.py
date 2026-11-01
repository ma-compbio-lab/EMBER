"""Within-cell-line drug-response (log2FC) regression on unseen drugs.

Components used by ``ember.eval.run_drug`` (``evaluate.py`` with ``evaluator: drug``):

* :class:`LFCAtlas` -- a DESeq2 log2FC atlas (obs = (cell_line, drug) contrast,
  var = gene) loaded into a dense float32 matrix plus a CSR-style index of the
  *observed* cells.
* :func:`load_split_spec` / :func:`split_obs_for_fold` -- the target-aware
  10-fold drug holdout defined by ``<cell_line>_split.json``.
* :class:`LFCDrugGeneDataset` -- streams ``(drug, gene) -> lfc`` samples. A drug
  is represented by the set of contextual embeddings of its annotated target
  genes (+ a mechanism-of-action id per target); the candidate gene by its own
  contextual embedding, both looked up in the held-out line's embedding layer.
* :class:`DrugGeneDeepSet` -- DeepSet over the target set, fused with the gene
  vector in a small MLP head.
* :func:`per_group_corr` / :func:`compute_regression_metrics` -- metrics.

MISSING-VALUE POLICY
--------------------
A NaN in the atlas means the contrast was untested or filtered by DESeq2 -- it
is genuinely missing, NOT zero. Nothing in this module ever 0-fills (or
mean-fills) a missing LFC: the validity index excludes NaN cells from train,
val and test; per-drug metrics are computed on each drug's own observed cells;
and test cells the embedding cannot cover are reported with the per-gene
*train-mean* LFC and flagged ``source='imputed'`` so they can be excluded
downstream. An imputed constant block shared by the truth and prediction
vectors would manufacture correlation, so the delta metric drops cells whose
gene has no finite train mean rather than imputing them.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset

MOA_CLASS_TO_ID = {"inhibitory": 0, "activating": 1, "other": 2}
_MOA_PAD = 2  # 'other'


# ---------------------------------------------------------------------------
# Atlas + splits
# ---------------------------------------------------------------------------

class LFCAtlas:
    """DESeq2 log2FC atlas: dense ``lfc`` (n_obs x n_genes, float32) + a
    CSR-style index of the observed (non-NaN) cells per obs.

    Args:
        atlas_path: h5ad with ``X`` = log2FC, ``obs[obs_ct_column]`` = cell-line
            key, ``obs[obs_pert_column]`` = drug id.
        gene_allowlist: restrict the gene axis to these genes (var order kept).
        use_padj: if True a cell must ALSO have finite ``layers['padj']`` to be
            valid. OPT-IN (default False) so that merely adding a padj layer to
            an atlas never silently changes the valid-gene set.
    """

    def __init__(self, atlas_path: str, gene_allowlist: list[str] | None = None,
                 use_padj: bool = False, obs_ct_column: str = 'cell_line',
                 obs_pert_column: str = 'broad_id'):
        print(f'  Loading atlas: {atlas_path}')
        adata = ad.read_h5ad(atlas_path)
        full_n_genes = adata.shape[1]
        if gene_allowlist is not None:
            allow_set = set(gene_allowlist)
            keep_mask = np.array([g in allow_set for g in adata.var_names])
            print(f'  Restricting to gene allowlist: {keep_mask.sum()} / {full_n_genes} kept')
            adata = adata[:, keep_mask].copy()

        self.n_obs, self.n_genes = adata.shape
        self.gene_names = np.asarray(adata.var_names)
        self.pert_per_obs = adata.obs[obs_pert_column].astype(str).values
        self.ct_per_obs = adata.obs[obs_ct_column].astype(str).values
        self.obs_idx_by_ct_pert = {(ct, pert): i for i, (ct, pert)
                                   in enumerate(zip(self.ct_per_obs, self.pert_per_obs))}
        self.lfc = np.asarray(adata.X, dtype=np.float32)

        # Validity = observed cells. padj is opt-in (see class docstring).
        valid = ~np.isnan(self.lfc)
        if 'padj' in adata.layers and use_padj:
            valid &= ~np.isnan(np.asarray(adata.layers['padj']))
            print('  padj filter ON (use_padj=True): genes with NaN padj dropped')
        elif 'padj' in adata.layers:
            print('  padj layer present but IGNORED (use_padj=False, default)')

        n_valid_per_obs = valid.sum(axis=1)
        self.valid_offsets = np.concatenate([[0], np.cumsum(n_valid_per_obs)]).astype(np.int64)
        _, gene_idxs_flat = np.where(valid)
        self.valid_gene_flat = gene_idxs_flat.astype(np.int32)
        print(f'  Atlas ready: {self.n_obs} obs, {self.n_genes} genes, '
              f'{len(self.valid_gene_flat):,} valid (obs, gene) pairs')

    def valid_genes(self, obs_idx: int) -> np.ndarray:
        return self.valid_gene_flat[self.valid_offsets[obs_idx]:self.valid_offsets[obs_idx + 1]]

    def build_split_arrays(self, obs_idx_list: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Flat int32 ``(obs_arr, gene_arr)`` listing every valid cell of the given obs."""
        obs_arrs, gene_arrs = [], []
        for obs_idx in obs_idx_list:
            valid = self.valid_genes(obs_idx)
            if len(valid) == 0:
                continue
            obs_arrs.append(np.full(len(valid), obs_idx, dtype=np.int32))
            gene_arrs.append(valid)
        if not obs_arrs:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        return np.concatenate(obs_arrs), np.concatenate(gene_arrs)


def load_split_spec(spec_path: str) -> dict:
    return json.loads(Path(spec_path).read_text())


def split_obs_for_fold(atlas: LFCAtlas, spec: dict, fold: int,
                       core_cts: list[str]) -> dict[str, list[int]]:
    """``{train, val, test} -> [obs_idx]`` for one fold.

    ``val_perts`` / ``test_perts`` are drugs of the held-out line. Train = every
    other drug of the core lines; when the held-out line is itself a core line
    (the within-line setup used here) its remaining drugs are the training pool.
    """
    holdout_ct = spec['ct_label']
    val_perts = set(spec[f'fold_{fold}']['val_perts'])
    test_perts = set(spec[f'fold_{fold}']['test_perts'])
    unseen = val_perts | test_perts
    core_set = set(core_cts)
    holdout_in_core = holdout_ct in core_set
    exclude_unseen = bool(spec.get('exclude_unseen_from_train', True))

    out: dict[str, list[int]] = {'train': [], 'val': [], 'test': []}
    for (ct, pert), obs_idx in atlas.obs_idx_by_ct_pert.items():
        if ct == holdout_ct:
            if pert in val_perts:
                out['val'].append(obs_idx)
            elif pert in test_perts:
                out['test'].append(obs_idx)
            elif holdout_in_core:
                out['train'].append(obs_idx)
        elif ct in core_set:
            if exclude_unseen and pert in unseen:
                continue
            out['train'].append(obs_idx)
    return out


# ---------------------------------------------------------------------------
# Drug representation + dataset
# ---------------------------------------------------------------------------

def load_drug_targets(lookup_path: str, embed_genes: set) -> tuple[dict, dict, set]:
    """Parse ``drug_lookup.csv`` (``broad_id,targets,moa_classes,...``; ``|``-joined).

    Returns ``(drug_targets, drug_moa, drug_drop)``. A drug is dropped when it has
    no targets, no target with an embedding, or more than half of its targets
    lack an embedding.
    """
    df = pd.read_csv(lookup_path)
    drug_targets, drug_moa, drug_drop = {}, {}, set()
    for _, r in df.iterrows():
        targets = str(r['targets']).split('|') if pd.notna(r['targets']) else []
        moas = str(r['moa_classes']).split('|') if pd.notna(r['moa_classes']) else []
        if len(moas) != len(targets):
            moas = ['other'] * len(targets)
        valid = [(g, MOA_CLASS_TO_ID.get(m, 2)) for g, m in zip(targets, moas) if g in embed_genes]
        if not targets or len(valid) < len(targets) / 2 or not valid:
            drug_drop.add(r['broad_id'])
            continue
        drug_targets[r['broad_id']] = [g for g, _ in valid]
        drug_moa[r['broad_id']] = [m for _, m in valid]
    return drug_targets, drug_moa, drug_drop


class LFCDrugGeneDataset(Dataset):
    """``(drug_x, mask, gene, moa_ids, y)`` samples over observed atlas cells.

    targets mode: ``drug_x`` = (max_targets, D) stack of target-gene embeddings
    (zero-padded; ``mask`` True = padding), ``moa_ids`` = per-target MoA class,
    ``gene`` = (D,) candidate-gene embedding. Embeddings come from
    ``embeds[cell_line]`` when the pickle is nested, else from the flat dict.

    morgan mode: ``drug_x`` = a fixed per-drug fingerprint vector, ``gene`` = the
    gene's atlas index (long) for a learnable embedding; mask/moa are dummies.
    """

    def __init__(self, atlas: LFCAtlas, obs_arr, gene_arr, embeds, embeds_nested: bool,
                 drug_targets=None, drug_moa=None, max_targets: int = 25,
                 mode: str = 'targets', morgan_vecs=None, embed_dim: int | None = None):
        self.atlas = atlas
        self.obs_arr = np.asarray(obs_arr)
        self.gene_arr = np.asarray(gene_arr)
        self.embeds = embeds
        self.nested = embeds_nested
        self.drug_targets = drug_targets or {}
        self.drug_moa = drug_moa or {}
        self.max_targets = int(max_targets)
        self.mode = mode
        if mode == 'morgan':
            self.morgan = morgan_vecs
            self.embed_dim = len(next(iter(morgan_vecs.values())))
            return
        self.embed_dim = embed_dim
        # Precompute per-line gene tables + per-(drug, line) target tensors so
        # __getitem__ is pure indexing.
        obs64 = self.obs_arr.astype(np.int64)
        cts = set(atlas.ct_per_obs[obs64]) if len(self.obs_arr) else set()
        self._gene_tbl = {ct: self._build_gene_table(ct) for ct in cts}
        seen = set(zip(atlas.pert_per_obs[obs64], atlas.ct_per_obs[obs64])) if len(self.obs_arr) else set()
        self._drug_cache = {(drug, ct): self._build_drug(drug, ct) for drug, ct in seen}

    def _layer(self, ct):
        return self.embeds.get(ct, {}) if self.nested else self.embeds

    def _build_gene_table(self, ct):
        layer = self._layer(ct)
        tbl = np.zeros((self.atlas.n_genes, self.embed_dim), dtype=np.float32)
        for gi, g in enumerate(self.atlas.gene_names):
            v = layer.get(g)
            if v is not None:
                tbl[gi] = v
        return tbl

    def _build_drug(self, drug, ct):
        layer = self._layer(ct)
        targets = self.drug_targets.get(drug, [])
        moa = self.drug_moa.get(drug, [_MOA_PAD] * len(targets))
        x = torch.zeros(self.max_targets, self.embed_dim)
        mask = torch.ones(self.max_targets, dtype=torch.bool)
        moa_ids = torch.full((self.max_targets,), _MOA_PAD, dtype=torch.long)
        j = 0
        for g, m in zip(targets, moa):
            v = layer.get(g)
            if v is not None:
                x[j] = torch.as_tensor(v, dtype=torch.float32)
                mask[j] = False
                moa_ids[j] = int(m)
                j += 1
                if j >= self.max_targets:
                    break
        return x, mask, moa_ids

    def __len__(self):
        return len(self.obs_arr)

    def __getitem__(self, i):
        obs_idx, gene_idx = int(self.obs_arr[i]), int(self.gene_arr[i])
        y = torch.tensor(float(self.atlas.lfc[obs_idx, gene_idx]), dtype=torch.float32)
        drug = self.atlas.pert_per_obs[obs_idx]
        if self.mode == 'morgan':
            return (torch.as_tensor(self.morgan[drug], dtype=torch.float32),
                    torch.zeros(1, dtype=torch.bool), torch.tensor(gene_idx, dtype=torch.long),
                    torch.zeros(1, dtype=torch.long), y)
        ct = self.atlas.ct_per_obs[obs_idx]
        x, mask, moa_ids = self._drug_cache[(drug, ct)]
        return x, mask, torch.from_numpy(self._gene_tbl[ct][gene_idx]), moa_ids, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DrugGeneDeepSet(nn.Module):
    """DeepSet over a drug's target-gene embeddings (+ MoA) fused with the
    candidate gene embedding -> scalar log2FC.

    targets mode: each target token ``[target_emb || moa_emb]`` goes through a
    shared ``n_layers`` MLP encoder, padding is masked out, tokens are sum-pooled.
    morgan mode: the drug vector is a fixed fingerprint projected by
    ``drug_proj`` (``drug_proj_layers`` 1 = Linear, >=2 = MLP) and the gene is a
    learnable ``nn.Embedding``. Head: ``concat([drug, gene]) -> MLP -> 1``.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2,
                 dropout: float = 0.0, mode: str = 'targets', n_moa_classes: int = 3,
                 moa_embed_dim: int = 8, drug_input_dim: int | None = None,
                 n_genes: int | None = None, gene_embed_dim: int = 128,
                 drug_proj_layers: int = 1):
        super().__init__()
        self.mode = mode
        self.use_moa = mode == 'targets' and n_moa_classes > 0
        self.learnable_gene = mode == 'morgan'

        # Module creation order is fixed: it determines the RNG stream at init.
        if mode == 'targets':
            self.moa_embedding = nn.Embedding(n_moa_classes, moa_embed_dim) if self.use_moa else None
            enc_layers, prev = [], input_dim + (moa_embed_dim if self.use_moa else 0)
            for _ in range(n_layers):
                enc_layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
                if dropout > 0:
                    enc_layers.append(nn.Dropout(dropout))
                prev = hidden_dim
            self.target_encoder = nn.Sequential(*enc_layers)
            self.drug_proj = None
        elif mode == 'morgan':
            self.target_encoder = None
            if drug_proj_layers <= 1:
                self.drug_proj = nn.Linear(drug_input_dim, hidden_dim)
            else:
                proj_layers, prev = [], drug_input_dim
                for _ in range(drug_proj_layers - 1):
                    proj_layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
                    if dropout > 0:
                        proj_layers.append(nn.Dropout(dropout))
                    prev = hidden_dim
                proj_layers.append(nn.Linear(prev, hidden_dim))
                self.drug_proj = nn.Sequential(*proj_layers)
        else:
            raise ValueError(f"mode must be 'targets' or 'morgan', got {mode!r}")

        if self.learnable_gene:
            self.gene_embedding = nn.Embedding(n_genes, gene_embed_dim)
            self.gene_proj = nn.Linear(gene_embed_dim, hidden_dim)
        else:
            self.gene_embedding = None
            self.gene_proj = nn.Linear(input_dim, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _drug_vec(self, drug_x, mask, moa_ids):
        if self.mode == 'morgan':
            return self.drug_proj(drug_x)
        x = drug_x  # (B, T, D); mask (B, T) True = pad
        if self.use_moa and moa_ids is not None:
            x = torch.cat([x, self.moa_embedding(moa_ids)], dim=-1)
        B, T, _ = x.shape
        enc = self.target_encoder(x.view(B * T, -1)).view(B, T, -1)
        enc = enc * (~mask).unsqueeze(-1).float()
        return enc.sum(dim=1)

    def forward(self, drug_x, mask, gene, moa_ids=None):
        drug_vec = self._drug_vec(drug_x, mask, moa_ids)
        gene_vec = self.gene_proj(self.gene_embedding(gene)) if self.learnable_gene else self.gene_proj(gene)
        return self.head(torch.cat([drug_vec, gene_vec], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_group_corr(group_ids, y_true, y_pred, min_n: int = 5):
    """Mean per-group Pearson / Spearman (groups = obs = (cell_line, drug)).

    Each group uses only its own observed cells. Groups with fewer than ``min_n``
    cells or zero-variance truth/prediction are skipped.
    Returns ``(mean_pearson, mean_spearman, n_groups_used, drop_stats)``.
    """
    g = np.asarray(group_ids)
    yt = np.asarray(y_true).astype(np.float64)
    yp = np.asarray(y_pred).astype(np.float64)
    order = np.argsort(g, kind='stable')
    g_s, yt_s, yp_s = g[order], yt[order], yp[order]
    boundaries = np.concatenate([[0], np.where(np.diff(g_s) != 0)[0] + 1, [len(g_s)]])
    rs, rhos = [], []
    n_total = len(boundaries) - 1
    n_too_few = n_const_truth = n_const_pred = n_exception = 0
    for i in range(n_total):
        a, b = boundaries[i], boundaries[i + 1]
        if b - a < min_n:
            n_too_few += 1
            continue
        ti, pi = yt_s[a:b], yp_s[a:b]
        t_const, p_const = ti.std() < 1e-12, pi.std() < 1e-12
        if t_const or p_const:
            if t_const:
                n_const_truth += 1
            if p_const and not t_const:
                n_const_pred += 1
            continue
        try:
            rs.append(float(pearsonr(ti, pi)[0]))
            rhos.append(float(spearmanr(ti, pi)[0]))
        except Exception:
            n_exception += 1
    drop_stats = {'n_total': int(n_total), 'n_used': int(len(rs)), 'n_too_few': int(n_too_few),
                  'n_constant_truth': int(n_const_truth), 'n_constant_pred': int(n_const_pred),
                  'n_exception': int(n_exception)}
    if not rs:
        return float('nan'), float('nan'), 0, drop_stats
    return float(np.mean(rs)), float(np.mean(rhos)), len(rs), drop_stats


def compute_regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {'pearson': pearsonr(y_true, y_pred)[0], 'spearman': spearmanr(y_true, y_pred)[0],
            'mse': mse, 'rmse': float(np.sqrt(mse))}
