"""Entry point for Ember pretraining.

    python pretrain.py                                  # configs/pretrain/ember_base.yaml
    python pretrain.py --config-name overfit_smoke      # 10-cell smoke test
    python pretrain.py name=my-run data.batch_size=128  # CLI overrides
"""

from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from ember.data import create_dataloader
from ember.model import EmberModel
from ember.trainer import PretrainTrainer
from ember.utils import count_parameters, set_seed


def gene_names_of(adata: ad.AnnData, gene_name_col) -> list[str]:
    return list(adata.var[gene_name_col]) if gene_name_col is not None else list(adata.var_names)


@hydra.main(version_base=None, config_path="configs/pretrain", config_name="ember_base")
def main(cfg: DictConfig):
    print("=" * 60)
    print("Ember Pretraining")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    # ------------------------------------------------------------------ data
    print("\nLoading data...")
    adata = ad.read_h5ad(cfg.paths.anndata_path)
    print(f"  Loaded training data: {adata.n_obs} cells, {adata.n_vars} genes")

    adata_val = None
    val_path = cfg.paths.get("val_anndata_path", None)
    if val_path is not None and Path(val_path).exists():
        adata_val = ad.read_h5ad(val_path)
        print(f"  Loaded validation data: {adata_val.n_obs} cells, {adata_val.n_vars} genes")

    graph_df = None
    if cfg.model.use_graph_loss:
        graph_path = Path(cfg.paths.graph_path)
        if not graph_path.exists():
            raise ValueError(f"use_graph_loss=True but graph_path does not exist: {graph_path}")
        graph_df = pd.read_csv(graph_path)
        print(f"  Loaded graph: {len(graph_df)} edges from {graph_path}")
        if cfg.data.graph_on_nonzero:
            print("  Graph loss restricted to nonzero-expression genes")

    # ---------------------------------------------------------- vocabulary
    gene_name_col = cfg.data.gene_name_col
    gene_names = gene_names_of(adata, gene_name_col)
    if len(gene_names) != len(set(gene_names)):
        raise ValueError(
            f"Found {len(gene_names) - len(set(gene_names))} duplicate gene names. "
            "De-duplicate genes in your anndata before pretraining."
        )
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    num_genes = adata.n_vars
    print(f"  Gene vocabulary: {num_genes} genes")

    if adata_val is not None:
        val_gene_names = gene_names_of(adata_val, gene_name_col)
        if val_gene_names != gene_names:
            raise ValueError(
                "Validation data must have the same genes in the same order as the training data."
            )

    # ------------------------------------------------------------- split
    overfit_mode = cfg.training.overfit
    all_indices = np.arange(adata.n_obs)
    if overfit_mode:
        train_indices, val_indices = all_indices, None
        print(f"\nOverfit mode: using all {len(train_indices)} cells for training")
    elif adata_val is not None:
        train_indices, val_indices = all_indices, np.arange(adata_val.n_obs)
        print(f"\nUsing separate validation data: train={len(train_indices)} val={len(val_indices)} cells")
    else:
        train_indices, val_indices = train_test_split(
            all_indices, train_size=cfg.data.train_ratio, random_state=cfg.seed
        )
        print(f"\nRandom split: train={len(train_indices)} val={len(val_indices)} cells")

    print("\nCreating dataloaders...")
    train_loader = create_dataloader(
        adata, graph_df, gene_to_idx, cfg.data, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, seed=cfg.seed, cell_indices=train_indices,
    )
    val_loader = None
    if not overfit_mode:
        val_loader = create_dataloader(
            adata_val if adata_val is not None else adata, graph_df, gene_to_idx, cfg.data,
            batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers,
            seed=cfg.seed + 1, cell_indices=val_indices,
        )
    print(f"  Train: {len(train_loader)} batches")
    if val_loader:
        print(f"  Val: {len(val_loader)} batches")

    # ------------------------------------------------------------- model
    print("\nInitializing model...")
    model = EmberModel(
        num_genes=num_genes,
        embed_dim=cfg.model.embed_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        encoder_dropout=cfg.model.encoder_dropout,
        decoder_dropout=cfg.model.decoder_dropout,
        use_graph_loss=cfg.model.use_graph_loss,
        graph_loss_weight=cfg.model.graph_loss_weight,
        graph_proj_dim=cfg.model.graph_proj_dim,
        graph_proj_dropout=cfg.model.graph_proj_dropout,
        graph_loss_batch_size=cfg.model.graph_loss_batch_size,
        graph_neg_ratio=cfg.model.graph_neg_ratio,
        neg_samp_strat=cfg.model.neg_samp_strat,
        graph_logit_scale=cfg.model.graph_logit_scale,
    )
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,} (trainable: {trainable_params:,})")

    # --------------------------------------------------- optimizer / schedule
    optimizer_class = getattr(optim, cfg.optimizer.name)
    optimizer = optimizer_class(
        model.parameters(), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay
    )

    total_steps = len(train_loader) * cfg.training.epochs
    warmup_steps = cfg.scheduler.warmup_steps
    print(f"\nLearning rate schedule: {total_steps} total steps, {warmup_steps} warmup steps")
    if warmup_steps >= total_steps:
        raise ValueError(
            f"scheduler.warmup_steps ({warmup_steps}) must be smaller than the total number of "
            f"optimizer steps ({total_steps} = {len(train_loader)} batches x {cfg.training.epochs} epochs)."
        )
    if warmup_steps > 0:
        # Linear warmup from ~0 to the base LR, then cosine decay to eta_min.
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8 / cfg.optimizer.learning_rate, end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=cfg.scheduler.eta_min
        )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    else:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=cfg.scheduler.eta_min
        )

    wandb_config = None
    if cfg.wandb.enabled:
        wandb_config = {"project": cfg.wandb.project, "entity": cfg.wandb.entity, "name": cfg.wandb.name}

    trainer = PretrainTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        results_dir=cfg.paths.results_dir,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        grad_norm_clip=cfg.training.grad_norm_clip,
        use_wandb=cfg.wandb.enabled,
        wandb_config=wandb_config,
        config=cfg,
        load_from_checkpoint=cfg.training.load_from_checkpoint,
        gene_to_idx=gene_to_idx,
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    results = trainer.fit(
        epochs=cfg.training.epochs,
        val_epoch_freq=cfg.training.val_epoch_freq,
        save_epoch_freq=cfg.training.save_epoch_freq,
        early_stopping_patience=cfg.training.early_stopping_patience,
        early_stopping_tol=cfg.training.early_stopping_tol,
    )
    print("\nTraining complete!")
    if results["best_val_loss"] is not None:
        print(f"Best val loss: {results['best_val_loss']:.4f} at epoch {results['best_epoch']}")


if __name__ == "__main__":
    main()
