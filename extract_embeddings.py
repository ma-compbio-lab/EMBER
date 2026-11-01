"""Extract contextual gene embeddings from an Ember checkpoint.

Each control cell is tokenised exactly as during pretraining (same context
length, zero-gene slots and log1p normalisation; no masking), run through the
model, and the per-gene output embeddings are averaged over all cells of a
group (typically a cell line / cell type) and over ``--n-samples`` independent
context samplings. Genes absent from the model vocabulary are dropped before
tokenisation and never embedded.

Output: a pickle ``embeds[group][gene_symbol] -> np.float32[embed_dim]``, the
format consumed by ``evaluate.py``.

Examples
--------
# one file, one group per obs value
python extract_embeddings.py --checkpoint checkpoints/ember_base/ember_base.pt \\
    --h5ad data/controls/tahoe_controls.h5ad --group-col cell_line --n-samples 2 \\
    --out embeddings/ember_base_tahoe.pkl

# several files, one constant group label each
python extract_embeddings.py --checkpoint ... \\
    --h5ad data/controls/K562_controls.h5ad data/controls/Jurkat_controls.h5ad \\
    --group-const K562 Jurkat --gene-col gene_name \\
    --out embeddings/ember_base_pert_response.pkl
"""

import argparse
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ember.alignment import check_gene_alignment
from ember.checkpoint import load_checkpoint
from ember.data import SCDataset, collate_fn


@torch.no_grad()
def accumulate_embeddings(model, dataloader, col_to_model_idx, gene_names, row_groups,
                          sums, counts, device, pbar):
    """Run the model over a dataloader and accumulate per-(group, gene) sums."""
    row_offset = 0
    for batch in dataloader:
        col_ids = batch["gene_ids"].numpy()  # (B, N) adata column indices
        B, N = col_ids.shape
        outputs = model(
            gene_ids=torch.from_numpy(col_to_model_idx[col_ids]).long().to(device),
            expression=batch["expression"].to(device),
            mask_positions=torch.zeros((B, 0), dtype=torch.long, device=device),
        )
        emb = outputs["embeddings"].cpu().numpy().astype(np.float64)  # (B, N, D)

        for b in range(B):
            group = row_groups[row_offset + b]
            if group not in sums:
                sums[group] = np.zeros((len(gene_names), emb.shape[-1]), dtype=np.float64)
                counts[group] = np.zeros(len(gene_names), dtype=np.int64)
            np.add.at(sums[group], col_ids[b], emb[b])
            np.add.at(counts[group], col_ids[b], 1)
        pbar.update(B)
        row_offset += B


def extract(model, gene_to_idx, data_cfg, adata, row_groups, n_samples, batch_size, device,
            sums, counts):
    """Tokenise ``adata`` ``n_samples`` times and accumulate embeddings per group."""
    model_genes = set(gene_to_idx)
    in_vocab = np.array([g in model_genes for g in adata.var_names])
    print(f"  Genes: {adata.n_vars} in data, {int(in_vocab.sum())} in model vocabulary "
          f"(dropping {int((~in_vocab).sum())} out-of-vocabulary genes)")
    adata = adata[:, in_vocab].copy()
    if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()

    gene_names = list(adata.var_names)
    col_to_model_idx = np.array([gene_to_idx[g] for g in gene_names])

    # Accumulators are keyed by adata column; keep a per-call view then fold into
    # the global per-gene dicts at the end.
    local_sums, local_counts = {}, {}
    for sample_idx in range(n_samples):
        dataset = SCDataset(
            adata=adata,
            graph_df=None,
            gene_to_idx=gene_to_idx,
            context_length=data_cfg["context_length"],
            zero_genes=data_cfg["zero_genes"],
            prioritize_high_expr=data_cfg["prioritize_high_expr"],
            mask_ratio=0.0,
            log1p=data_cfg["log1p"],
            seed=42 + sample_idx,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)
        with tqdm(total=adata.n_obs, desc=f"  sample {sample_idx + 1}/{n_samples}") as pbar:
            accumulate_embeddings(model, loader, col_to_model_idx, gene_names, row_groups,
                                  local_sums, local_counts, device, pbar)

    for group in local_sums:
        g_sums = sums.setdefault(group, {})
        g_counts = counts.setdefault(group, {})
        for col, gene in enumerate(gene_names):
            if local_counts[group][col] > 0:
                g_sums[gene] = g_sums.get(gene, 0.0) + local_sums[group][col]
                g_counts[gene] = g_counts.get(gene, 0) + int(local_counts[group][col])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_epoch_N.pt")
    parser.add_argument("--h5ad", nargs="+", required=True, help="Control-cell h5ad file(s), raw counts in .X")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--group-col", help="obs column whose values become the embedding groups")
    group.add_argument("--group-const", nargs="+", help="One constant group label per --h5ad file")
    parser.add_argument("--gene-col", default=None,
                        help="var column holding gene symbols (default: var_names)")
    parser.add_argument("--filter-col", default=None, help="Keep only cells where obs[filter-col] == filter-value")
    parser.add_argument("--filter-value", default=None)
    parser.add_argument("--n-samples", type=int, default=1, help="Independent context samplings per cell")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random-init", action="store_true", help="Use a randomly initialised model (baseline)")
    parser.add_argument("--skip-alignment-check", action="store_true")
    parser.add_argument("--out", required=True, help="Output pickle path")
    args = parser.parse_args()

    if args.group_const is not None and len(args.group_const) != len(args.h5ad):
        parser.error("--group-const needs exactly one label per --h5ad file")
    if (args.filter_col is None) != (args.filter_value is None):
        parser.error("--filter-col and --filter-value must be given together")

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Loading model from {args.checkpoint}...")
    model, gene_to_idx, data_cfg, _ = load_checkpoint(args.checkpoint, args.device, random_init=args.random_init)
    print(f"  vocabulary: {len(gene_to_idx)} genes; context_length={data_cfg['context_length']}, "
          f"zero_genes={data_cfg['zero_genes']}, prioritize_high_expr={data_cfg['prioritize_high_expr']}")

    sums, counts = {}, {}
    for file_idx, path in enumerate(args.h5ad):
        print(f"\nLoading {path}...")
        adata = ad.read_h5ad(path)
        if args.filter_col is not None:
            keep = adata.obs[args.filter_col].astype(str).values == args.filter_value
            adata = adata[keep].copy()
            print(f"  kept {adata.n_obs} cells with {args.filter_col} == {args.filter_value!r}")
        if args.gene_col is not None:
            adata.var_names = adata.var[args.gene_col].astype(str).values
        adata.var_names_make_unique()
        print(f"  {adata.n_obs} cells x {adata.n_vars} genes")
        if not args.skip_alignment_check:
            check_gene_alignment(adata)

        if args.group_col is not None:
            row_groups = list(adata.obs[args.group_col].astype(str).values)
        else:
            row_groups = [args.group_const[file_idx]] * adata.n_obs
        for g in sorted(set(row_groups)):
            print(f"  group {g}: {row_groups.count(g)} cells")

        extract(model, gene_to_idx, data_cfg, adata, row_groups, args.n_samples, args.batch_size,
                args.device, sums, counts)

    embeds = {
        group: {gene: (sums[group][gene] / counts[group][gene]).astype(np.float32) for gene in sums[group]}
        for group in sums
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(embeds, f)

    print(f"\nSaved embeddings to {out}")
    for group in sorted(embeds):
        print(f"  {group}: {len(embeds[group])} genes")
    any_group = next(iter(embeds))
    print(f"Embedding dim: {len(next(iter(embeds[any_group].values())))}   access: embeds[group][gene]")


if __name__ == "__main__":
    main()
