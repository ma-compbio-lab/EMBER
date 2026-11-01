"""Synthetic smoke test for the gene-pair evaluation (evaluate.py / ember.eval.pairs).

Builds a tiny nested embedding pickle and task CSVs with two folds in a temp
dir, runs classification and regression end to end (max_epochs=2) and checks
the output CSVs. Run with:  pytest tests/test_eval_pairs.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember.eval.run_pairs import run as run_evaluation  # noqa: E402

GENES = [f"G{i:03d}" for i in range(40)]
CELL_LINES = ["K562", "HepG2"]
DIM = 8


def _make_embeds(path, rng):
    # drop two genes from HepG2 so the per-cell-line filter + base-rate path runs
    embeds = {}
    for ct in CELL_LINES:
        genes = GENES if ct == "K562" else GENES[:-2]
        embeds[ct] = {g: rng.normal(size=DIM).astype(np.float32) for g in genes}
    with open(path, "wb") as f:
        pickle.dump(embeds, f)


def _splits(n, rng):
    df = {}
    for k in (1, 2):
        s = np.array(["train"] * n, dtype=object)
        idx = rng.permutation(n)
        s[idx[: n // 5]] = "test"
        s[idx[n // 5: 2 * n // 5]] = "val"
        df[f"split_{k}"] = s
    return df


def _make_classification_csv(path, rng, n=400):
    pert = rng.choice(GENES, n)
    gene = rng.choice(GENES, n)
    ct = rng.choice(CELL_LINES, n)
    label = (rng.random(n) < 0.4).astype(int)
    df = pd.DataFrame({"pert": pert, "gene": gene, "cell_line": ct, "label": label, **_splits(n, rng)})
    df.to_csv(path, index=False)


def _make_regression_csv(path, rng, n=600):
    lib = rng.choice(GENES[:20], n)
    query = rng.choice(GENES[20:], n)
    # mix of strong interactions (|qGI|>0.3, low FDR) and confident zeros
    strong = rng.random(n) < 0.3
    qgi = np.where(strong, rng.choice([-1, 1], n) * rng.uniform(0.35, 1.0, n), rng.uniform(-0.04, 0.04, n))
    fdr = np.where(strong, rng.uniform(0.0, 0.05, n), rng.uniform(0.6, 1.0, n))
    df = pd.DataFrame({"query_gene": query, "library_gene": lib, "qGI": qgi, "FDR": fdr,
                       "cell_line": "K562", **_splits(n, rng)})
    df.to_csv(path, index=False)


def _base_cfg(tmp_path, dataset, embed_path, out):
    return {
        "embed_name": "synthetic", "embed_path": str(embed_path),
        "dataset": str(dataset), "output_dir": str(out), "name_suffix": "",
        "ct_specific": True, "ct_column": "cell_line",
        "model_type": "mlp", "fusion_strat": "concat",
        "hidden_dim": 16, "n_layers": 2, "max_epochs": 2, "patience": 2, "tol": 1e-4, "seed": 0,
        "sweep": {"lr": [1e-3], "batch_size": [64], "weight_decay": [0.0], "dropout": [0.0, 0.1]},
    }


def test_classification(tmp_path):
    rng = np.random.RandomState(0)
    emb = tmp_path / "emb.pkl"
    csv = tmp_path / "task.csv"
    _make_embeds(emb, rng)
    _make_classification_csv(csv, rng)
    cfg = OmegaConf.create({**_base_cfg(tmp_path, csv, emb, tmp_path / "out"),
                            "task_type": "classification", "label_col": "label",
                            "gene_columns": ["pert", "gene"], "val_selection_criteria": "auroc",
                            "optimizer": "adamw"})
    summary_path, preds_path = run_evaluation(cfg)

    summary = pd.read_csv(summary_path)
    assert list(summary.columns) == [
        "auroc", "auprc", "f1", "precision", "recall", "accuracy", "lr", "batch_size",
        "weight_decay", "dropout", "hidden_dim", "n_layers", "val_loss", "fold", "cell_line",
        "n_samples", "macro_pert_auroc", "macro_pert_auprc", "macro_gene_auroc",
        "macro_gene_auprc", "macro_pooled_auroc", "macro_pooled_auprc"]
    assert sorted(summary["cell_line"].unique()) == ["HepG2", "K562", "all"]
    assert set(summary["fold"]) == {"split_1", "split_2"}

    preds = pd.read_csv(preds_path)
    assert list(preds.columns) == ["pert", "gene", "cell_line", "y_true", "y_prob", "fold",
                                   "lr", "batch_size", "weight_decay", "source"]
    df = pd.read_csv(csv)
    n_test = sum((df[f"split_{k}"] == "test").sum() for k in (1, 2))
    assert len(preds) == n_test                      # missing-gene rows re-added
    assert (preds["source"] == "imputed").any()
    assert preds["y_prob"].between(0, 1).all()


def test_regression_with_residual_metrics(tmp_path):
    rng = np.random.RandomState(1)
    emb = tmp_path / "emb.pkl"
    csv = tmp_path / "gi.csv"
    _make_embeds(emb, rng)
    _make_regression_csv(csv, rng)
    cfg = OmegaConf.create({**_base_cfg(tmp_path, csv, emb, tmp_path / "out"),
                            "task_type": "regression", "label_col": "qGI",
                            "gene_columns": ["library_gene", "query_gene"],
                            "val_selection_criteria": "residual_auprc_any",
                            "residual_fdr_col": "FDR", "residual_metrics": True})
    summary_path, preds_path = run_evaluation(cfg)

    summary = pd.read_csv(summary_path)
    assert list(summary.columns) == [
        "pearson", "spearman", "mse", "rmse", "lr", "batch_size", "weight_decay", "dropout",
        "hidden_dim", "n_layers", "val_loss", "fold", "cell_line", "n_samples"]
    preds = pd.read_csv(preds_path)
    assert list(preds.columns)[:2] == ["library_gene", "query_gene"]

    resid = pd.read_csv(tmp_path / "out" / "synthetic_mlp_residual_metrics.csv")
    assert list(resid["fold"]) == ["split_1", "split_2", "mean", "std"]
    for c in ("any_auprc", "any_auroc", "neg_auprc", "pos_auprc", "any_bal_auroc", "any_raw_auprc"):
        assert c in resid.columns
    assert resid.loc[resid["fold"] == "mean", "any_auprc"].notna().all()


if __name__ == "__main__":
    # Also runnable without pytest: python tests/test_eval_pairs.py
    import tempfile
    for fn in (test_classification, test_regression_with_residual_metrics):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
        print(f"PASS {fn.__name__}")
