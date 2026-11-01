"""Ember model: a transformer over gene tokens trained with masked-expression
reconstruction (MSE) and an InfoNCE link-prediction loss on a PPI graph.

Module names match the original research implementation so released checkpoints
load with ``strict=True``.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    """Feed-forward block."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Expression encoder / decoder
# ---------------------------------------------------------------------------

class ExpressionEncoder(nn.Module):
    """Continuous expression value -> embedding (2-layer MLP)."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.mlp(expression.unsqueeze(-1))


class ExpressionDecoder(nn.Module):
    """Embedding -> scalar expression prediction (2-layer MLP)."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Ember
# ---------------------------------------------------------------------------

class EmberModel(nn.Module):
    """Transformer over gene tokens with masked-expression and graph losses.

    Input tokens are ``gene_embedding[gene] + expression_encoder(expr)``; masked
    positions have their expression embedding replaced by a learned mask token.
    The per-gene outputs of the final LayerNorm are the *contextual gene
    embeddings*. Two losses are applied on top:

    - reconstruction: MSE between the decoder output and the (log1p) expression
      at the masked positions;
    - graph: InfoNCE over PPI edges in the context, scored in a projected,
      L2-normalised space, with ``graph_neg_ratio`` sampled negatives per edge.
    """

    def __init__(
        self,
        num_genes: int,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        encoder_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
        use_graph_loss: bool = True,
        graph_loss_weight: float = 3.0,
        graph_proj_dim: Optional[int] = None,
        graph_proj_dropout: float = 0.1,
        graph_loss_batch_size: int = 1024,
        graph_neg_ratio: int = 5,
        neg_samp_strat: str = "per_pos_head",
        graph_logit_scale: float = 5.0,
    ):
        """
        Args:
            num_genes: Vocabulary size
            embed_dim: Token / output embedding dimension
            num_layers, num_heads, mlp_ratio, dropout: Transformer hyperparameters
            encoder_dropout / decoder_dropout: Dropout inside the expression MLPs
            use_graph_loss: Enable the PPI InfoNCE loss
            graph_loss_weight: Weight of the graph loss in the total loss
            graph_proj_dim: Output dim of the 2-layer graph projection MLP (default: embed_dim)
            graph_proj_dropout: Dropout inside the graph projection MLP
            graph_loss_batch_size: Max positive edges sampled per cell
            graph_neg_ratio: Negatives sampled per positive edge
            neg_samp_strat: "per_pos_head" (negatives per source node, excluding
                known edges and self-loops) or "random"
            graph_logit_scale: Temperature^-1 applied to cosine scores
        """
        super().__init__()
        if neg_samp_strat not in ("per_pos_head", "random"):
            raise ValueError(f"Unknown neg_samp_strat: {neg_samp_strat}")

        self.num_genes = num_genes
        self.embed_dim = embed_dim
        self.use_graph_loss = use_graph_loss
        self.graph_loss_weight = graph_loss_weight
        self.graph_loss_batch_size = graph_loss_batch_size
        self.graph_neg_ratio = graph_neg_ratio
        self.neg_samp_strat = neg_samp_strat
        self.graph_logit_scale = graph_logit_scale

        # Module creation order is fixed: it determines the parameter-init RNG stream.
        self.gene_embedding = nn.Embedding(num_genes, embed_dim)
        self.expression_encoder = ExpressionEncoder(embed_dim, embed_dim // 2, dropout=encoder_dropout)
        self.mask_token = nn.Parameter(torch.randn(embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=dropout, attn_drop=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = ExpressionDecoder(embed_dim, embed_dim // 2, dropout=decoder_dropout)

        if use_graph_loss:
            proj_dim = graph_proj_dim if graph_proj_dim is not None else embed_dim
            self.graph_proj = nn.Sequential(
                nn.Linear(embed_dim, proj_dim),
                nn.GELU(),
                nn.Dropout(graph_proj_dropout),
                nn.Linear(proj_dim, proj_dim),
            )
        else:
            self.graph_proj = None

    def forward(
        self,
        gene_ids: torch.Tensor,
        expression: torch.Tensor,
        mask_positions: torch.Tensor,
        edge_index: Optional[list] = None,
    ) -> dict:
        """
        Args:
            gene_ids: (B, N) vocabulary indices
            expression: (B, N) log1p-normalised expression
            mask_positions: (B, num_masked) positions to mask (num_masked may be 0)
            edge_index: list of B (2, E_i) local edge tensors (optional)

        Returns:
            dict with 'loss', 'reconstruction_loss', 'graph_loss' (if computed),
            'predictions' (B, N) and 'embeddings' (B, N, embed_dim)
        """
        B, N = gene_ids.shape

        gene_emb = self.gene_embedding(gene_ids)
        expr_emb = self.expression_encoder(expression)

        expr_emb_masked = expr_emb.clone()
        if mask_positions.numel() > 0:
            for i in range(B):
                expr_emb_masked[i, mask_positions[i]] = self.mask_token
        x = gene_emb + expr_emb_masked

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        predictions = self.decoder(x)
        recon_loss = self._compute_recon_loss(predictions, expression, mask_positions)

        graph_loss = None
        if self.use_graph_loss and edge_index is not None:
            graph_losses = []
            for i in range(B):
                edge_idx_i = edge_index[i].to(x.device)
                if edge_idx_i.shape[1] > 0:
                    graph_losses.append(self._compute_graph_loss_infonce(x[i], edge_idx_i))
            if graph_losses:
                graph_loss = torch.stack(graph_losses).mean()

        total_loss = recon_loss
        if graph_loss is not None:
            total_loss = total_loss + self.graph_loss_weight * graph_loss

        result = {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "predictions": predictions,
            "embeddings": x,
        }
        if graph_loss is not None:
            result["graph_loss"] = graph_loss
        return result

    @staticmethod
    def _compute_recon_loss(predictions, targets, mask_positions) -> torch.Tensor:
        """MSE restricted to the masked positions of each cell."""
        if mask_positions.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)
        B = predictions.shape[0]
        num_masked = mask_positions.shape[1]
        batch_idx = torch.arange(B, device=predictions.device).unsqueeze(1).expand(-1, num_masked)
        pred_masked = predictions[batch_idx, mask_positions]
        target_masked = targets[batch_idx, mask_positions]
        return F.mse_loss(pred_masked, target_masked)

    def _compute_graph_loss_infonce(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """InfoNCE link prediction for one cell.

        For each positive edge (u, v) we sample ``graph_neg_ratio`` negative
        destinations and apply a (1 + R)-way cross-entropy with the positive at
        index 0. Being a ranking loss, it tolerates unrecorded positive edges.
        """
        z = self.graph_proj(z)
        z = F.normalize(z, dim=-1)

        num_nodes = z.size(0)
        num_total_pos = edge_index.size(1)

        if num_total_pos > self.graph_loss_batch_size:
            perm = torch.randperm(num_total_pos, device=edge_index.device)[:self.graph_loss_batch_size]
            pos_edge_index = edge_index[:, perm]
            num_pos = self.graph_loss_batch_size
        else:
            pos_edge_index = edge_index
            num_pos = num_total_pos

        src_nodes = pos_edge_index[0]  # (K,)
        src_emb = z[src_nodes]  # (K, D)
        pos_dst_emb = z[pos_edge_index[1]]  # (K, D)
        R = self.graph_neg_ratio

        if self.neg_samp_strat == "random":
            neg_dst_idx = torch.randint(num_nodes, (num_pos, R), device=z.device)
        else:  # per_pos_head: resample negatives that hit known edges or self-loops
            exclude = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=z.device)
            exclude[edge_index[0], edge_index[1]] = True
            exclude.fill_diagonal_(True)
            src_exclude = exclude[src_nodes]  # (K, num_nodes)
            neg_dst_idx = torch.randint(num_nodes, (num_pos, R), device=z.device)
            for _ in range(3):
                invalid = torch.gather(src_exclude, 1, neg_dst_idx)
                if not invalid.any():
                    break
                neg_dst_idx[invalid] = torch.randint(
                    num_nodes, (invalid.sum().item(),), device=z.device
                )

        neg_dst_emb = z[neg_dst_idx]  # (K, R, D)
        pos_scores = (src_emb * pos_dst_emb).sum(dim=-1, keepdim=True) * self.graph_logit_scale  # (K, 1)
        neg_scores = torch.bmm(neg_dst_emb, src_emb.unsqueeze(-1)).squeeze(-1) * self.graph_logit_scale  # (K, R)

        logits = torch.cat([pos_scores, neg_scores], dim=-1)  # (K, 1+R)
        labels = torch.zeros(num_pos, dtype=torch.long, device=z.device)
        return F.cross_entropy(logits, labels)

    def model_config(self) -> dict:
        """Constructor kwargs, stored in checkpoints for standalone loading."""
        return {
            "num_genes": self.num_genes,
            "embed_dim": self.embed_dim,
            "num_layers": len(self.blocks),
            "num_heads": self.blocks[0].attn.num_heads,
            "mlp_ratio": self.blocks[0].mlp.fc1.out_features / self.embed_dim,
            "dropout": self.blocks[0].mlp.drop1.p,
            "encoder_dropout": self.expression_encoder.mlp[2].p,
            "decoder_dropout": self.decoder.mlp[2].p,
            "use_graph_loss": self.use_graph_loss,
            "graph_loss_weight": self.graph_loss_weight,
            "graph_proj_dim": self.graph_proj[0].out_features if self.graph_proj is not None else None,
            "graph_proj_dropout": self.graph_proj[2].p if self.graph_proj is not None else 0.1,
            "graph_loss_batch_size": self.graph_loss_batch_size,
            "graph_neg_ratio": self.graph_neg_ratio,
            "neg_samp_strat": self.neg_samp_strat,
            "graph_logit_scale": self.graph_logit_scale,
        }
