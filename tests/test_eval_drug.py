"""Synthetic smoke test for the drug-LFC evaluation (ember/eval/drug_lfc.py +
evaluate.py, evaluator: drug). Builds a tiny atlas with NaNs, a 1-fold split, a drug
lookup and a nested embedding pickle in a temp dir and runs the driver with
max_epochs=2. Checks the outputs exist, that no NaN LFC leaks into y_true and
that uncovered test cells are reported as ``source='imputed'``.

Run: pytest tests/test_eval_drug.py
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _build(tmp: Path):
    rng = np.random.default_rng(0)
    genes = [f'G{i}' for i in range(40)]
    drugs = [f'D{i}' for i in range(30)]
    # targets: 2 per drug; drug D29 targets a gene the embedding lacks -> dropped
    targets = {d: [genes[(2 * i) % 40], genes[(2 * i + 1) % 40]] for i, d in enumerate(drugs)}
    targets['D29'] = ['MISSING1', 'MISSING2']
    X = rng.normal(size=(len(drugs), len(genes))).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan  # untested cells
    atlas = ad.AnnData(X=X, obs=pd.DataFrame({'broad_id': drugs, 'cell_line': 'CL1'},
                                             index=[f'o{i}' for i in range(len(drugs))]),
                       var=pd.DataFrame(index=genes))
    atlas.write_h5ad(tmp / 'atlas.h5ad')
    (tmp / 'genes_eval.txt').write_text('\n'.join(genes))
    pd.DataFrame({'broad_id': drugs,
                  'targets': ['|'.join(targets[d]) for d in drugs],
                  'moa_classes': ['inhibitory|other'] * len(drugs),
                  'n_targets': 2}).to_csv(tmp / 'drug_lookup.csv', index=False)
    spec = {'ct_label': 'CL1', 'ct_slug': 'cl1', 'cell_name': 'CL1', 'task_variant': 'within',
            'core_training_cts': {'CL1': 'CL1'}, 'exclude_unseen_from_train': False,
            'n_folds': 1, 'seed': 0, 'fold_1': {'val_perts': drugs[20:25], 'test_perts': drugs[25:30]}}
    (tmp / 'cl1_split.json').write_text(json.dumps(spec))
    # nested pickle keyed by the line; gene G39 has no embedding -> its test cells are imputed
    emb = {g: rng.normal(size=16).astype(np.float32) for g in genes[:39]}
    pickle.dump({'CL1': emb}, open(tmp / 'emb.pkl', 'wb'))


def test_drug_lfc_smoke(tmp_path):
    _build(tmp_path)
    cmd = [sys.executable, 'evaluate.py', '--config-name', 'lincs_unseen_drugs',
           f'task_dir={tmp_path}', 'cell_line=cl1', 'embed_name=emb', f'embed_path={tmp_path}/emb.pkl',
           f'output_dir={tmp_path}/out', 'max_epochs=2', 'patience=2',
           'sweep.lr=[1e-3]', 'sweep.weight_decay=[0.0]', 'sweep.dropout=[0.0]', 'sweep.batch_size=[64]',
           f'hydra.run.dir={tmp_path}/hydra']
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr

    summ = pd.read_csv(tmp_path / 'out' / 'emb_targets_summary.csv')
    preds = pd.read_csv(tmp_path / 'out' / 'emb_targets_preds.csv')
    assert list(summ['fold']) == ['split_1', 'all']
    assert list(summ.columns[:5]) == ['fold', 'cell_line', 'per_drug_pearson', 'per_drug_spearman',
                                      'per_drug_pearson_delta']
    assert not preds['y_true'].isna().any() and not preds['y_pred'].isna().any()
    assert set(preds['source']) == {'model', 'imputed'}
    # imputed rows = cells of the dropped drug D29 + cells of gene G39 (no embedding)
    imp = preds[preds['source'] == 'imputed']
    assert set(imp['broad_id']) >= {'D29'} and (imp['gene'] == 'G39').any()
    assert int(summ.loc[0, 'n_imputed']) == len(imp)
    # every observed test cell is present exactly once
    atlas = ad.read_h5ad(tmp_path / 'atlas.h5ad')
    test_rows = atlas.obs['broad_id'].isin([f'D{i}' for i in range(25, 30)]).values
    assert len(preds) == int((~np.isnan(atlas.X[test_rows])).sum())


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_drug_lfc_smoke(Path(d))
    print('ok')
