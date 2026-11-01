#!/usr/bin/env python3
"""Evaluate gene embeddings on a benchmark task.

One entry point for all six benchmarks; the config's ``evaluator`` key selects
the driver:

    evaluator: pairs   gene-pair prediction over frozen embeddings
                       (pert-response-*, genetic-interaction)   -> ember.eval.run_pairs
    evaluator: drug    unseen-drug log2FC regression
                       (lincs-unseen-drugs, tahoe-unseen-drugs)  -> ember.eval.run_drug

Usage:
    python evaluate.py --config-name pert_response_unseen_pert cell_line=k562 embed_name=<pkl stem>
    python evaluate.py --config-name genetic_interaction embed_name=<pkl stem>
    python evaluate.py --config-name lincs_unseen_drugs cell_line=a549 embed_name=<pkl stem>
    python evaluate.py --config-name tahoe_unseen_drugs cell_line=sw480 embed_path=/abs/path.pkl +folds=[1]

Configs live in configs/eval/; outputs go to the config's ``output_dir``.
"""

import hydra
from omegaconf import DictConfig

EVALUATORS = {
    "pairs": "ember.eval.run_pairs",
    "drug": "ember.eval.run_drug",
}


@hydra.main(version_base=None, config_path="configs/eval", config_name="pert_response_unseen_pert")
def main(cfg: DictConfig):
    evaluator = cfg.get("evaluator", None)
    if evaluator not in EVALUATORS:
        raise ValueError(
            f"Config must set evaluator to one of {sorted(EVALUATORS)} (got {evaluator!r})."
        )
    import importlib

    importlib.import_module(EVALUATORS[evaluator]).run(cfg)


if __name__ == "__main__":
    main()
