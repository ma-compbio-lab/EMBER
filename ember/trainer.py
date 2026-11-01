"""Pretraining loop: gradient accumulation (via accelerate), LR scheduling,
checkpointing with full RNG state, early stopping and CSV / W&B logging."""

import csv
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from tqdm import tqdm

from ember.model import EmberModel


class PretrainTrainer:
    """Self-supervised pretraining on gene expression."""

    def __init__(
        self,
        model: EmberModel,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler=None,
        results_dir: str = "checkpoints/run",
        accumulate_grad_batches: int = 1,
        grad_norm_clip: Optional[float] = None,
        use_wandb: bool = False,
        wandb_config: Optional[dict] = None,
        config=None,
        load_from_checkpoint: bool = False,
        gene_to_idx: Optional[dict] = None,
    ):
        """
        Args:
            model: EmberModel instance
            train_loader / val_loader: dataloaders (val may be None)
            optimizer: torch optimizer
            lr_scheduler: per-step LR scheduler (stepped after every optimizer step)
            results_dir: where checkpoints, vocab and logs are written
            accumulate_grad_batches: gradient accumulation steps
            grad_norm_clip: max gradient norm (None = no clipping)
            use_wandb / wandb_config: optional Weights & Biases logging
            config: full Hydra config (saved next to the checkpoints)
            load_from_checkpoint: resume from ``results_dir/milestone.txt`` if present
            gene_to_idx: gene vocabulary (stored in every checkpoint)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_norm_clip = grad_norm_clip
        self.config = config
        self.gene_to_idx = gene_to_idx

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.global_step = 0
        checkpoint = None
        if load_from_checkpoint:
            self.start_epoch, checkpoint = self._load_checkpoint()

        self.accelerator = Accelerator(gradient_accumulation_steps=accumulate_grad_batches)
        if self.val_loader is not None:
            self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader
            )
        else:
            self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader
            )
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
            if checkpoint is not None and "lr_scheduler_state_dict" in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # CSV logs (always on): one row per optimizer step / per validation.
        self.train_log = self._open_log("log_train.csv",
                                        ["step", "epoch", "loss", "recon_loss", "graph_loss", "lr", "grad_norm"])
        self.val_log = self._open_log("log_val.csv",
                                      ["epoch", "step", "train_loss", "val_loss", "val_recon_loss", "val_graph_loss"])

        self.use_wandb = use_wandb
        if use_wandb and self.accelerator.is_main_process:
            import wandb

            wandb_cfg = dict(wandb_config) if wandb_config else {}
            if self.config is not None:
                wandb_cfg["config"] = OmegaConf.to_container(self.config, resolve=True)
            wandb.init(**wandb_cfg)

    # ------------------------------------------------------------------ logging

    def _open_log(self, name: str, header: list[str]):
        if not self.accelerator.is_main_process:
            return None
        path = self.results_dir / name
        new_file = not path.exists() or self.start_epoch == 0
        f = open(path, "w" if new_file else "a", newline="")
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        return (f, writer)

    def _log_row(self, log, row: list):
        if log is None:
            return
        f, writer = log
        writer.writerow(row)
        f.flush()

    # -------------------------------------------------------------- checkpoints

    def _load_checkpoint(self):
        """Resume from the checkpoint named in ``milestone.txt``."""
        milestone_path = self.results_dir / "milestone.txt"
        if not milestone_path.exists():
            print(f"No checkpoint found in {self.results_dir}")
            return 0, None

        checkpoint_path = self.results_dir / milestone_path.read_text().strip()
        if not checkpoint_path.exists():
            print(f"Checkpoint {checkpoint_path} not found")
            return 0, None

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)

        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.random.set_rng_state(checkpoint["torch_rng_state"].cpu())

        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch, checkpoint

    def save_checkpoint(self, epoch: int):
        if not self.accelerator.is_main_process:
            return

        config_path = self.results_dir / "config.yaml"
        if not config_path.exists() and self.config is not None:
            OmegaConf.save(OmegaConf.create(OmegaConf.to_container(self.config, resolve=True)), config_path)

        gene_vocab_path = self.results_dir / "gene_vocab.tsv"
        if not gene_vocab_path.exists() and self.gene_to_idx is not None:
            with open(gene_vocab_path, "w") as f:
                f.write("index\tgene_name\n")
                for gene_name, idx in sorted(self.gene_to_idx.items(), key=lambda x: x[1]):
                    f.write(f"{idx}\t{gene_name}\n")

        unwrapped = self.accelerator.unwrap_model(self.model)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": unwrapped.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
            # Metadata for standalone loading (see ember/checkpoint.py)
            "gene_to_idx": self.gene_to_idx,
            "num_genes": unwrapped.num_genes,
            "model_config": unwrapped.model_config(),
            "data_config": OmegaConf.to_container(self.config.data, resolve=True) if self.config else None,
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, self.results_dir / checkpoint_name)
        print(f"Saved checkpoint: {self.results_dir / checkpoint_name}")
        (self.results_dir / "milestone.txt").write_text(checkpoint_name)

    # ----------------------------------------------------------------- training

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.accelerator.is_main_process)
        for batch in pbar:
            iter_start = time.time()
            with self.accelerator.accumulate(self.model):
                outputs = self.model(
                    gene_ids=batch["gene_ids"],
                    expression=batch["expression"],
                    mask_positions=batch["mask_positions"],
                    edge_index=batch["edge_index"],
                )
                loss = outputs["loss"]
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = None
                    if self.grad_norm_clip:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.accelerator.is_main_process:
                        lr = self.optimizer.param_groups[0]["lr"]
                        graph_loss = outputs["graph_loss"].item() if "graph_loss" in outputs else float("nan")
                        gn = grad_norm.item() if grad_norm is not None else float("nan")
                        self._log_row(self.train_log, [
                            self.global_step, epoch, loss.item(), outputs["reconstruction_loss"].item(),
                            graph_loss, lr, gn,
                        ])
                        if self.use_wandb:
                            import wandb

                            wandb.log({
                                "train/epoch": epoch,
                                "train/loss": loss.item(),
                                "train/recon_loss": outputs["reconstruction_loss"].item(),
                                "train/graph_loss": graph_loss,
                                "train/lr": lr,
                                "train/grad_norm": gn,
                                "train/iter_time": time.time() - iter_start,
                            }, step=self.global_step)

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": loss.item()})

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        self.model.eval()
        total_loss = total_recon = total_graph = 0.0
        num_batches = num_graph_batches = 0

        for batch in tqdm(dataloader, desc="Evaluating val", disable=not self.accelerator.is_main_process):
            outputs = self.model(
                gene_ids=batch["gene_ids"],
                expression=batch["expression"],
                mask_positions=batch["mask_positions"],
                edge_index=batch["edge_index"],
            )
            total_loss += outputs["loss"].item()
            total_recon += outputs["reconstruction_loss"].item()
            num_batches += 1
            if "graph_loss" in outputs:
                total_graph += outputs["graph_loss"].item()
                num_graph_batches += 1

        metrics = {
            "val_loss": total_loss / num_batches,
            "val_recon_loss": total_recon / num_batches,
            "val_graph_loss": total_graph / num_graph_batches if num_graph_batches > 0 else float("nan"),
        }
        if self.use_wandb and self.accelerator.is_main_process:
            import wandb

            wandb.log({f"val/{k[4:]}": v for k, v in metrics.items()}, step=self.global_step)
        return metrics

    def fit(
        self,
        epochs: int,
        val_epoch_freq: int = 1,
        save_epoch_freq: int = 1,
        early_stopping_patience: int = 25,
        early_stopping_tol: float = 1e-4,
    ) -> dict:
        """Main training loop. Returns the best validation loss and its epoch."""
        print(f"Training for {epochs} epochs...")
        best_val_loss = float("inf")
        best_epoch = -1
        no_improvement = 0
        skip_validation = self.val_loader is None

        if self.start_epoch == 0 and not skip_validation:
            val_metrics = self.evaluate(self.val_loader)
            if self.accelerator.is_main_process:
                print(f"Initial val loss: {val_metrics['val_loss']:.4f}")

        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_epoch(epoch)

            if skip_validation:
                if self.accelerator.is_main_process:
                    print(f"Epoch {epoch}: train={train_loss:.4f}")
            elif epoch % val_epoch_freq == 0:
                val_metrics = self.evaluate(self.val_loader)
                val_loss = val_metrics["val_loss"]
                if val_loss < best_val_loss - early_stopping_tol:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    no_improvement = 0
                    if self.accelerator.is_main_process:
                        print(f"  New best: {best_val_loss:.4f}")
                else:
                    no_improvement += 1
                if self.accelerator.is_main_process:
                    print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
                    self._log_row(self.val_log, [
                        epoch, self.global_step, train_loss, val_loss,
                        val_metrics["val_recon_loss"], val_metrics["val_graph_loss"],
                    ])

            if not skip_validation and no_improvement >= early_stopping_patience:
                if self.accelerator.is_main_process:
                    print(f"Early stopping at epoch {epoch}")
                break

            if epoch % save_epoch_freq == 0:
                self.save_checkpoint(epoch)

        if skip_validation:
            return {"best_val_loss": None, "best_epoch": None}
        print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
        return {"best_val_loss": best_val_loss, "best_epoch": best_epoch}
