"""Dataset and dataloader for masked-expression pretraining and embedding extraction.

Each cell is turned into a fixed-length context of genes:

1. ``context_length - zero_genes`` slots are filled with nonzero-expression
   genes (sampled uniformly at random by default, or the top-expressed genes if
   ``prioritize_high_expr``).
2. The remaining slots are filled with randomly sampled zero-expression genes.
3. Expression is library-size normalised (CP10K) and log1p-transformed.
4. A random ``mask_ratio`` fraction of positions is selected for masking.
5. The PPI graph is subset to the edges whose endpoints are both in the context
   (optionally restricted to nonzero-expression genes).

The same dataset is used for embedding extraction with ``mask_ratio=0.0``, so a
checkpoint sees exactly the context statistics it was trained on.
"""

from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SCDataset(Dataset):
    """Single-cell RNA-seq dataset for masked expression prediction."""

    def __init__(
        self,
        adata: ad.AnnData,
        graph_df: Optional[pd.DataFrame] = None,
        gene_to_idx: Optional[dict] = None,
        context_length: int = 1280,
        zero_genes: int = 256,
        prioritize_high_expr: bool = False,
        mask_ratio: float = 0.5,
        log1p: bool = True,
        seed: int = 42,
        gene_name_col: Optional[str] = None,
        cell_indices: Optional[np.ndarray] = None,
        graph_on_nonzero: bool = True,
    ):
        """
        Args:
            adata: AnnData object with cells x genes (raw counts in ``.X``)
            graph_df: DataFrame with columns ['gene1', 'gene2'] for PPI edges
            gene_to_idx: Mapping from gene names to model vocabulary indices
            context_length: Number of genes to include per cell
            zero_genes: Number of context slots reserved for zero-expression genes
            prioritize_high_expr: Take the top-expressed genes instead of a random sample
            mask_ratio: Fraction of context positions to mask
            log1p: Apply CP10K normalisation + log1p to the expression values
            seed: Random seed for the per-dataset sampler
            gene_name_col: Column in ``adata.var`` with gene names (None: ``var_names``)
            cell_indices: Optional subset of cell indices to use (avoids copying data)
            graph_on_nonzero: Keep only graph edges between nonzero-expression genes
        """
        self.adata = adata
        self.context_length = context_length
        self.zero_genes = zero_genes
        self.prioritize_high_expr = prioritize_high_expr
        self.mask_ratio = mask_ratio
        self.log1p_flag = log1p
        self.rng = np.random.RandomState(seed)
        self.graph_on_nonzero = graph_on_nonzero
        self.cell_indices = cell_indices if cell_indices is not None else np.arange(adata.n_obs)

        if gene_name_col is not None:
            self.gene_names = list(adata.var[gene_name_col])
        else:
            self.gene_names = list(adata.var_names)
        if gene_to_idx is None:
            self.gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        else:
            self.gene_to_idx = gene_to_idx

        # Graph operations index by adata column (tokenize returns column indices).
        self.num_genes = adata.n_vars
        gene_name_to_col = {name: i for i, name in enumerate(self.gene_names)}

        self.edge_index = torch.zeros((2, 0), dtype=torch.long)
        if graph_df is not None:
            graph_genes = set(graph_df["gene1"].unique()) | set(graph_df["gene2"].unique())
            adata_genes = set(self.gene_names)
            genes_in_graph = adata_genes & graph_genes
            print("  Gene coverage check:")
            print(f"    Genes in anndata: {len(adata_genes)}")
            print(f"    Genes in graph: {len(graph_genes)}")
            print(f"    Anndata genes found in graph: {len(genes_in_graph)} "
                  f"({100 * len(genes_in_graph) / len(adata_genes):.1f}%)")

            g1_in = graph_df["gene1"].isin(gene_name_to_col)
            g2_in = graph_df["gene2"].isin(gene_name_to_col)
            valid = graph_df[g1_in & g2_in]
            if len(valid) > 0:
                i1 = valid["gene1"].map(gene_name_to_col).values
                i2 = valid["gene2"].map(gene_name_to_col).values
                self.edge_index = torch.tensor(np.stack([i1, i2]), dtype=torch.long)  # (2, E)
            print(f"    Edges in subsetted graph: {self.edge_index.shape[1]}")

    def __len__(self) -> int:
        return len(self.cell_indices)

    @staticmethod
    def transform_expr(cell_expr: np.ndarray) -> np.ndarray:
        """Library-size normalise to counts per 10,000 and apply log1p."""
        total = cell_expr.sum()
        if total > 0:
            normalized = cell_expr / total * 1e4
            return np.log1p(normalized)
        return cell_expr

    def tokenize(self, cell_expr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Select the genes that form this cell's context.

        Returns:
            gene_indices: (context_length,) adata column indices
            expression: (context_length,) expression values for those genes
        """
        nonzero_mask = cell_expr > 0
        nonzero_indices = np.where(nonzero_mask)[0]
        zero_indices = np.where(~nonzero_mask)[0]

        n_nonzero_slots = self.context_length - self.zero_genes
        n_other = min(n_nonzero_slots, len(nonzero_indices))

        if self.prioritize_high_expr:
            nonzero_pool = nonzero_indices
            if len(nonzero_pool) > 0:
                sorted_idx = np.argsort(-cell_expr[nonzero_pool])
                nonzero_pool = nonzero_pool[sorted_idx]
            selected_nonzero = nonzero_pool[:n_other]
        else:
            if n_other > 0:
                selected_nonzero = self.rng.choice(nonzero_indices, size=n_other, replace=False)
            else:
                selected_nonzero = np.array([], dtype=np.int64)

        # Remaining slots (zero_genes + any nonzero shortfall) come from zero genes.
        n_zeros_needed = self.context_length - len(selected_nonzero)
        if n_zeros_needed > 0 and len(zero_indices) > 0:
            n_zeros = min(n_zeros_needed, len(zero_indices))
            selected_zeros = self.rng.choice(zero_indices, size=n_zeros, replace=False)
        else:
            selected_zeros = np.array([], dtype=np.int64)

        gene_indices = np.concatenate([selected_nonzero, selected_zeros])
        expression = cell_expr[gene_indices]

        if len(gene_indices) < self.context_length:
            pad_len = self.context_length - len(gene_indices)
            gene_indices = np.pad(gene_indices, (0, pad_len), constant_values=0)
            expression = np.pad(expression, (0, pad_len), constant_values=0.0)

        return gene_indices, expression

    def subset_graph(self, gene_indices: np.ndarray) -> torch.Tensor:
        """Edges between the selected genes, remapped to local positions [0, N)."""
        if self.edge_index.shape[1] == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        selected_mask = torch.zeros(self.num_genes, dtype=torch.bool)
        selected_mask[gene_indices] = True
        edge_mask = selected_mask[self.edge_index[0]] & selected_mask[self.edge_index[1]]
        if not edge_mask.any():
            return torch.zeros((2, 0), dtype=torch.long)

        selected_edges = self.edge_index[:, edge_mask]
        vocab_to_local = torch.zeros(self.num_genes, dtype=torch.long)
        vocab_to_local[gene_indices] = torch.arange(len(gene_indices))
        return vocab_to_local[selected_edges]

    def __getitem__(self, idx: int) -> dict:
        actual_idx = self.cell_indices[idx]

        cell_expr = self.adata[actual_idx].X
        if hasattr(cell_expr, "toarray"):
            cell_expr = cell_expr.toarray().flatten()
        else:
            cell_expr = np.asarray(cell_expr).flatten()

        if self.log1p_flag:
            cell_expr = self.transform_expr(cell_expr)

        gene_indices, expression = self.tokenize(cell_expr)
        nonzero_expr_mask = expression > 0

        edge_index = self.subset_graph(gene_indices)
        if self.graph_on_nonzero and edge_index.shape[1] > 0:
            nz = torch.from_numpy(nonzero_expr_mask)
            keep = nz[edge_index[0]] & nz[edge_index[1]]
            edge_index = edge_index[:, keep]

        # At least one position is always drawn (also with mask_ratio=0, where the
        # caller discards it) so the sampler state is identical across settings.
        n_genes = len(gene_indices)
        n_mask = max(1, int(n_genes * self.mask_ratio))
        mask_positions = self.rng.choice(n_genes, size=n_mask, replace=False)

        return {
            "gene_ids": torch.tensor(gene_indices, dtype=torch.long),
            "expression": torch.tensor(expression, dtype=torch.float32),
            "mask_positions": torch.tensor(mask_positions, dtype=torch.long),
            "edge_index": edge_index,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Stack a batch; edge_index stays a list because edge counts differ per cell."""
    return {
        "gene_ids": torch.stack([b["gene_ids"] for b in batch]),
        "expression": torch.stack([b["expression"] for b in batch]),
        "mask_positions": torch.stack([b["mask_positions"] for b in batch]),
        "edge_index": [b["edge_index"] for b in batch],
    }


def create_dataloader(
    adata: ad.AnnData,
    graph_df: Optional[pd.DataFrame],
    gene_to_idx: dict,
    data_cfg,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    cell_indices: Optional[np.ndarray] = None,
) -> DataLoader:
    """Build an ``SCDataset`` from the ``data`` section of a pretraining config."""
    dataset = SCDataset(
        adata=adata,
        graph_df=graph_df,
        gene_to_idx=gene_to_idx,
        context_length=data_cfg.context_length,
        zero_genes=data_cfg.zero_genes,
        prioritize_high_expr=data_cfg.prioritize_high_expr,
        mask_ratio=data_cfg.mask_ratio,
        log1p=data_cfg.log1p,
        seed=seed,
        gene_name_col=data_cfg.gene_name_col,
        cell_indices=cell_indices,
        graph_on_nonzero=data_cfg.graph_on_nonzero,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
