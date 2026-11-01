"""Smoke tests for the pretraining stack on the 10 bundled cells (CPU, seconds)."""

from pathlib import Path

import anndata as ad
import pandas as pd
import torch

from ember.data import SCDataset, collate_fn
from ember.model import EmberModel
from ember.checkpoint import model_kwargs_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def _dataset(mask_ratio):
    adata = ad.read_h5ad(ROOT / "data/cxg_overfit.h5ad")
    graph = pd.read_csv(ROOT / "graphs/9606.filtered_800_adaptive_gene_symbols.csv")
    return SCDataset(adata, graph, context_length=256, zero_genes=64, mask_ratio=mask_ratio,
                     gene_name_col="feature_name", seed=0)


def test_dataset_shapes_and_graph():
    ds = _dataset(0.5)
    item = ds[0]
    assert item["gene_ids"].shape == (256,) and item["expression"].shape == (256,)
    assert item["mask_positions"].shape == (128,)
    assert item["edge_index"].shape[0] == 2
    # edges are local positions, and graph_on_nonzero keeps only expressed endpoints
    e = item["edge_index"]
    assert e.max() < 256 and bool((item["expression"][e] > 0).all())


def test_model_train_step_and_checkpoint_roundtrip(tmp_path):
    ds = _dataset(0.5)
    batch = collate_fn([ds[i] for i in range(4)])
    torch.manual_seed(0)
    model = EmberModel(num_genes=ds.num_genes, embed_dim=32, num_layers=2, num_heads=4, graph_proj_dim=16)
    out = model(**batch)
    assert out["embeddings"].shape == (4, 256, 32) and out["predictions"].shape == (4, 256)
    assert torch.isfinite(out["loss"]) and "graph_loss" in out
    out["loss"].backward()

    ckpt = {"model_state_dict": model.state_dict(), "model_config": model.model_config()}
    torch.save(ckpt, tmp_path / "ck.pt")
    loaded = torch.load(tmp_path / "ck.pt", weights_only=False)
    model2 = EmberModel(**model_kwargs_from_checkpoint(loaded))
    model2.load_state_dict(loaded["model_state_dict"], strict=True)

    model.eval(); model2.eval()
    with torch.no_grad():
        a = model(batch["gene_ids"], batch["expression"], torch.zeros((4, 0), dtype=torch.long))["embeddings"]
        b = model2(batch["gene_ids"], batch["expression"], torch.zeros((4, 0), dtype=torch.long))["embeddings"]
    assert torch.equal(a, b)
