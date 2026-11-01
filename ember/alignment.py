#!/usr/bin/env python3
"""Guard against gene-label misalignment in cells fed to contextualization.

WHY THIS EXISTS: a control-cell h5ad with SCRAMBLED gene labels (var_names shifted by
one relative to X after a malformed export) once produced meaningless, collapsed
contextual embeddings (all cell lines ~0.99 cosine) that went unnoticed for a long
time. Contextualising on a misaligned matrix silently attaches every gene's
expression to the wrong symbol.

THE TELL is dead simple and universal: in ANY real single-cell matrix, the top-expressed
genes are always mitochondrial (MT-*) and ribosomal (RPL/RPS) plus a few housekeeping
genes (ACTB, GAPDH, EEF1A1, TMSB4X). If those read ~0 — or olfactory receptors /
pseudogenes / lncRNAs top the list — the gene axis is misaligned. This module asserts the
housekeeping tell, and is called by ``extract_embeddings.py`` before contextualizing.

Usage:
    from ember.alignment import check_gene_alignment
    check_gene_alignment(adata)            # raises AssertionError if misaligned

    python -m ember.alignment <cells.h5ad>   # CLI, exit 1 if misaligned
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

# Genes that are top-expressed in essentially every human single cell.
HOUSEKEEPING = ['ACTB', 'GAPDH', 'EEF1A1', 'TMSB4X', 'B2M', 'RPL13', 'RPS18', 'RPLP0',
                'ACTG1', 'TPT1', 'FTL', 'RPL10', 'RPS27', 'RPL41']
MITO = ['MT-CO1', 'MT-CO2', 'MT-CO3', 'MT-ATP6', 'MT-ND4', 'MT-ND1', 'MT-CYB']


def check_gene_alignment(adata, *, n_cells=3000, top_frac=0.05, min_hk_in_top=0.5,
                         min_top20_housekeeping=5, raise_on_fail=True, verbose=True):
    """Assert the housekeeping/mito 'top-expressed' tell holds (i.e. genes are aligned).

    Two independent checks, both of which a correctly-aligned matrix passes easily and a
    scrambled one fails badly:
      1. >= `min_hk_in_top` fraction of the present housekeeping+mito genes rank in the
         top `top_frac` of genes by mean expression.
      2. >= `min_top20_housekeeping` of the 20 highest-expressed genes are MT-*/RPL/RPS/
         housekeeping (real cells are dominated by these; scrambled ones are not).

    Args:
        adata: AnnData with RAW counts (or any monotonic expression) and gene-symbol
               var_names. Only a subsample of `n_cells` rows is used.
    Returns: dict of diagnostics. Raises AssertionError on failure if raise_on_fail.
    """
    var = np.asarray(adata.var_names, dtype=str)
    n = min(n_cells, adata.n_obs)
    X = adata[:n].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    mean = np.asarray(X).mean(0).ravel()
    order = np.argsort(-mean)
    rank = np.empty(len(mean), int)
    rank[order] = np.arange(len(mean))            # 0 = highest expressed
    top_cut = max(1, int(top_frac * len(mean)))

    present = [g for g in HOUSEKEEPING + MITO if g in set(var)]
    gi = {g: i for i, g in enumerate(var)}
    hk_in_top = [g for g in present if rank[gi[g]] < top_cut]
    frac_hk = len(hk_in_top) / max(1, len(present))

    top20 = set(var[order[:20]])
    hk_like = sum(1 for g in top20 if g.startswith(('MT-', 'RPL', 'RPS', 'RP', 'MRPL', 'MRPS'))
                  or g in set(HOUSEKEEPING))
    diag = dict(n_genes=len(mean), n_present_housekeeping=len(present),
                frac_housekeeping_in_top=round(frac_hk, 3),
                top20_housekeeping_like=hk_like,
                top5_genes=list(var[order[:5]]),
                worst_housekeeping=[(g, int(rank[gi[g]])) for g in present
                                    if rank[gi[g]] >= top_cut][:6])
    if verbose:
        print(f"[check_gene_alignment] {diag['frac_housekeeping_in_top']:.0%} of "
              f"{len(present)} housekeeping/mito genes in top {top_frac:.0%}; "
              f"{hk_like}/20 top genes are MT-/RP-/housekeeping; top5={diag['top5_genes']}")

    ok = (frac_hk >= min_hk_in_top) and (hk_like >= min_top20_housekeeping)
    if not ok and raise_on_fail:
        raise AssertionError(
            "GENE MISALIGNMENT DETECTED — housekeeping/mito genes are NOT top-expressed "
            f"(only {frac_hk:.0%} in top {top_frac:.0%}; {hk_like}/20 top genes are "
            f"housekeeping-like; top-5 expressed = {diag['top5_genes']}). The gene axis is "
            "almost certainly scrambled. DO NOT contextualize on "
            "this matrix. Verify var_names are aligned to X before proceeding. "
            f"Diagnostics: {diag}")
    diag['aligned'] = bool(ok)
    return diag


if __name__ == '__main__':
    import sys
    import anndata as ad
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    a = ad.read_h5ad(sys.argv[1], backed='r')
    try:
        check_gene_alignment(a)
        print(f"OK: {sys.argv[1]} — genes appear correctly aligned.")
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
