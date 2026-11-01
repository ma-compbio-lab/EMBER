"""Load an Ember checkpoint for inference."""

from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from ember.model import EmberModel

# Checkpoints from the original research codebase store its full feature matrix.
# Ember only implements the released recipe; refuse anything else loudly.
_LEGACY_REQUIRED = {
    "loss_type": "mse",
    "architecture": "default",
    "use_celltype_loss": False,
    "use_tissue_loss": False,
    "use_cls_celltype_loss": False,
    "unitsphere_embeds": False,
    "mask_gene_ids": False,
    "use_recon_pred_loss": False,
}
# Only enforced when the checkpoint was trained with the graph loss.
_LEGACY_REQUIRED_GRAPH = {
    "graph_loss_type": "infonce",
    "graph_lin_proj": True,
    "graph_mlp": True,
}


def model_kwargs_from_checkpoint(checkpoint: dict) -> dict:
    """Translate a checkpoint's ``model_config`` into ``EmberModel`` kwargs."""
    cfg = dict(checkpoint["model_config"])
    sd = checkpoint["model_state_dict"]

    if "loss_type" in cfg:  # research-codebase checkpoint
        required = dict(_LEGACY_REQUIRED)
        if cfg.get("use_graph_loss", False):
            required.update(_LEGACY_REQUIRED_GRAPH)
        for key, expected in required.items():
            if key in cfg and cfg[key] != expected:
                raise ValueError(
                    f"Checkpoint uses {key}={cfg[key]!r}; Ember only supports {expected!r}."
                )
        out_dim = cfg.get("output_embed_dim")
        if out_dim is not None and out_dim != cfg["embed_dim"]:
            raise ValueError("Checkpoint uses an output projection; Ember does not support it.")

    kwargs = {
        "num_genes": cfg["num_genes"],
        "embed_dim": cfg["embed_dim"],
        "num_layers": cfg["num_layers"],
        "num_heads": cfg["num_heads"],
        "mlp_ratio": cfg.get("mlp_ratio", 4.0),
        "dropout": cfg.get("dropout", 0.1),
        "encoder_dropout": cfg.get("encoder_dropout", 0.0),
        "decoder_dropout": cfg.get("decoder_dropout", 0.0),
        "use_graph_loss": cfg.get("use_graph_loss", "graph_proj.0.weight" in sd),
        "graph_loss_weight": cfg.get("graph_loss_weight", 1.0),
        "graph_proj_dim": cfg.get("graph_proj_dim"),
        "graph_proj_dropout": cfg.get("graph_proj_dropout", 0.1),
        "graph_loss_batch_size": cfg.get("graph_loss_batch_size", 1024),
        "graph_neg_ratio": cfg.get("graph_neg_ratio", 5),
        "neg_samp_strat": cfg.get("neg_samp_strat", "per_pos_head"),
        "graph_logit_scale": cfg.get("graph_logit_scale", 5.0),
    }
    if kwargs["use_graph_loss"] and kwargs["graph_proj_dim"] is None and "graph_proj.0.weight" in sd:
        kwargs["graph_proj_dim"] = sd["graph_proj.0.weight"].shape[0]
    return kwargs


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    random_init: bool = False,
) -> tuple[EmberModel, dict, dict, Optional[str]]:
    """Load a checkpoint file for inference.

    Returns:
        model: ``EmberModel`` in eval mode on ``device``
        gene_to_idx: vocabulary used by the model
        data_cfg: the ``data`` section the model was trained with (context_length,
            zero_genes, prioritize_high_expr, log1p, ...), needed to tokenise new
            cells the same way. Taken from the checkpoint, or from ``config.yaml``
            next to it for research-codebase checkpoints.
        run_name: name of the pretraining run (from ``config.yaml``), if known
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = EmberModel(**model_kwargs_from_checkpoint(checkpoint))
    if not random_init:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    data_cfg = checkpoint.get("data_config")
    run_name = None
    config_path = checkpoint_path.parent / "config.yaml"
    if config_path.exists():
        run_cfg = OmegaConf.load(config_path)
        run_name = run_cfg.get("name")
        if data_cfg is None:
            data_cfg = OmegaConf.to_container(run_cfg.data, resolve=True)
    if data_cfg is None:
        raise FileNotFoundError(
            f"{checkpoint_path} has no 'data_config' and no config.yaml beside it; "
            "cannot determine how to tokenise cells."
        )
    data_cfg.setdefault("graph_on_nonzero", True)
    return model, checkpoint["gene_to_idx"], data_cfg, run_name
