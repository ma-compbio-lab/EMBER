"""Residual genetic-interaction metrics for the signed-qGI regression task.

The genetic-interaction model is trained on SIGNED qGI with MSE over
[library || query] embeddings, so ``y_prob`` in ``*_mlp_preds.csv`` is the
predicted signed qGI. Raw predictions are dominated by per-gene main effects
(a "hub" library gene looks interacting with everything), so we remove them
with a two-way additive residual computed WITHIN each fold's test set:

    r_lib[i]   = mean_j y_prob(i, j)
    r_query[j] = mean_i y_prob(i, j)
    R_ij       = y_prob(i, j) - r_lib[i] - r_query[j] + grand_mean

and score three labeled subsets of the test pairs:

    ANY: positives = any significant GI (|qGI| > 0.3 & FDR < 0.1)   score |R|
    NEG: positives = synthetic lethal  (qGI < -0.3 & FDR < 0.1)     score -R
    POS: positives = buffering         (qGI > +0.3 & FDR < 0.1)     score +R
    negatives (all arms) = confident zero (|qGI| < 0.05 & FDR > 0.5)

Per arm we report AUPRC and AUROC at natural prevalence, the same metrics on
the RAW signed prediction (``*_raw_*``, no residual) to isolate the value of
main-effect removal, and a balanced, query-anchored, library-degree-matched
AUROC (``*_bal_auroc``) in which every positive is paired with a confident-zero
partner of the SAME query gene with the closest library-gene degree.

Base-rate-imputed test rows (pairs with a gene missing from the embedding)
are dropped first: a constant must never enter r_lib / r_query.

CLI:
    python -m ember.eval.gi_residual --preds <..._mlp_preds.csv> --dataset <gi_dataset.csv>
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr

# MUST match ember/eval/pairs.py RESID_* (val selection uses the same labels).
POS_QGI = 0.3
POS_FDR = 0.1
NEG_QGI = 0.05
NEG_FDR = 0.5

# balanced degree-matched AUROC sampling
N_BAL_DRAWS = 20          # negative draws averaged per fold/arm (de-noise ties)
BAL_SEED = 12345

ARMS = ('any', 'neg', 'pos')
METRIC_COLS = (['pearson', 'spearman', 'mse']
               + [f'{a}_{m}' for a in ARMS
                  for m in ('auprc', 'auroc', 'raw_auprc', 'raw_auroc', 'bal_auroc', 'bal_raw_auroc')])


# ---------------------------------------------------------------------------
# labels / scores
# ---------------------------------------------------------------------------

def build_labeled_subsets(qgi, fdr):
    """{arm: (row mask, int labels)} with the shared confident-zero negatives."""
    conf_neg_gi = (qgi < -POS_QGI) & (fdr < POS_FDR)   # synthetic lethal
    conf_pos_gi = (qgi > POS_QGI) & (fdr < POS_FDR)    # buffering
    conf_any_gi = conf_neg_gi | conf_pos_gi
    conf_zero = (qgi.abs() < NEG_QGI) & (fdr > NEG_FDR)
    return {
        'any': (conf_any_gi | conf_zero, conf_any_gi.astype(int)),
        'neg': (conf_neg_gi | conf_zero, conf_neg_gi.astype(int)),
        'pos': (conf_pos_gi | conf_zero, conf_pos_gi.astype(int)),
    }


def two_way_residual(g, lib_col, query_col):
    """R = y_prob - r_lib - r_query + grand_mean, all computed within g."""
    r_lib = g.groupby(lib_col)['y_prob'].transform('mean')
    r_query = g.groupby(query_col)['y_prob'].transform('mean')
    gm = float(g['y_prob'].mean())
    return (g['y_prob'] - r_lib - r_query + gm).to_numpy()


def regression_metrics(y_true, y_pred):
    if np.std(y_pred) < 1e-12:
        return np.nan, np.nan, float(np.mean((y_true - y_pred) ** 2))
    return (pearsonr(y_true, y_pred)[0],
            spearmanr(y_true, y_pred)[0],
            float(np.mean((y_true - y_pred) ** 2)))


def binary_metrics(labels, score):
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    return roc_auc_score(labels, score), average_precision_score(labels, score)


# ---------------------------------------------------------------------------
# balanced, degree-matched AUROC
# ---------------------------------------------------------------------------

def arm_library_degree(df_full, lib_col):
    """Atlas-wide, arm-specific library-gene degree = #significant partners.

    A design quantity from the ground-truth atlas used only to build fair
    evaluation negatives; it is never fed to the model.
    """
    sig_neg = (df_full['qGI'] < -POS_QGI) & (df_full['FDR'] < POS_FDR)
    sig_pos = (df_full['qGI'] > POS_QGI) & (df_full['FDR'] < POS_FDR)
    sig_any = sig_neg | sig_pos
    return {
        'neg': df_full.loc[sig_neg].groupby(lib_col).size().to_dict(),
        'pos': df_full.loc[sig_pos].groupby(lib_col).size().to_dict(),
        'any': df_full.loc[sig_any].groupby(lib_col).size().to_dict(),
    }


def _safe_auc(labels, score):
    if len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, score)


def _match_negatives_one_draw(pos_deg, neg_deg, rng):
    """Greedy nearest-degree assignment without replacement, random tie-break.

    Returns, for each positive, the chosen index into neg_deg. Requires
    len(neg_deg) >= len(pos_deg).
    """
    used = np.zeros(len(neg_deg), bool)
    chosen = np.empty(len(pos_deg), int)
    for i in rng.permutation(len(pos_deg)):          # random positive order
        avail = np.where(~used)[0]
        gaps = np.abs(neg_deg[avail] - pos_deg[i])
        ties = avail[gaps == gaps.min()]
        j = ties[rng.randint(len(ties))]             # random among equal-degree
        used[j] = True
        chosen[i] = j
    return chosen


def balanced_auroc(g, arm, resid, y_prob, lib_deg, lib_col, query_col):
    """Query-anchored, library-degree-matched, 1:1 balanced AUROC.

    For every positive (query q, library+), draw a confident-zero partner of
    the SAME q whose library-gene degree is closest to library+'s. Sharing the
    query cancels query hubness by construction; matching library degree
    removes the remaining library-hub confound. Returns
    (auroc_residual, auroc_raw, n_pos_matched), averaged over N_BAL_DRAWS draws.
    """
    qgi, fdr = g['qGI'], g['FDR']
    if arm == 'neg':
        pos_mask = (qgi < -POS_QGI) & (fdr < POS_FDR)
        rscore, raw = -resid, -y_prob
    elif arm == 'pos':
        pos_mask = (qgi > POS_QGI) & (fdr < POS_FDR)
        rscore, raw = resid, y_prob
    else:
        pos_mask = (qgi.abs() > POS_QGI) & (fdr < POS_FDR)
        rscore, raw = np.abs(resid), np.abs(y_prob)
    neg_mask = (qgi.abs() < NEG_QGI) & (fdr > NEG_FDR)

    libdeg = g[lib_col].map(lib_deg[arm]).fillna(0).to_numpy()
    query = g[query_col].to_numpy()

    pos_idx = np.where(pos_mask.to_numpy())[0]
    neg_idx = np.where(neg_mask.to_numpy())[0]
    if len(pos_idx) < 1 or len(neg_idx) < 1:
        return np.nan, np.nan, 0

    pos_by_q, neg_by_q = {}, {}
    for j in pos_idx:
        pos_by_q.setdefault(query[j], []).append(j)
    for j in neg_idx:
        neg_by_q.setdefault(query[j], []).append(j)

    rng = np.random.RandomState(BAL_SEED)
    aurocs, raw_aurocs, n_matched = [], [], 0
    for _ in range(N_BAL_DRAWS):
        sel_pos, sel_neg = [], []
        for q, plist in pos_by_q.items():
            cand = neg_by_q.get(q)
            if not cand:
                continue
            cand = np.array(cand)
            p = np.array(plist)
            if len(cand) < len(p):                    # rare: subsample positives
                p = p[rng.choice(len(p), len(cand), replace=False)]
            ch = _match_negatives_one_draw(libdeg[p], libdeg[cand], rng)
            sel_pos.extend(p.tolist())
            sel_neg.extend(cand[ch].tolist())
        if not sel_pos:
            continue
        rows = np.array(sel_pos + sel_neg)
        labels = np.r_[np.ones(len(sel_pos)), np.zeros(len(sel_neg))]
        aurocs.append(_safe_auc(labels, rscore[rows]))
        raw_aurocs.append(_safe_auc(labels, raw[rows]))
        n_matched = len(sel_pos)
    return (float(np.nanmean(aurocs)) if aurocs else np.nan,
            float(np.nanmean(raw_aurocs)) if raw_aurocs else np.nan,
            n_matched)


# ---------------------------------------------------------------------------
# per-fold analysis
# ---------------------------------------------------------------------------

def analyze_fold(g, lib_deg, lib_col='library_gene', query_col='query_gene'):
    """Metrics for one fold. g has lib_col, query_col, y_prob, qGI, FDR with
    imputed rows already dropped."""
    y_prob = g['y_prob'].to_numpy()
    qgi, fdr = g['qGI'], g['FDR']
    resid = two_way_residual(g, lib_col, query_col)

    pearson, spearman, mse = regression_metrics(qgi.to_numpy(), y_prob)
    out = {'pearson': pearson, 'spearman': spearman, 'mse': mse,
           'resid_mean': float(np.mean(resid)), 'resid_std': float(np.std(resid))}

    arm_score = {'any': np.abs(resid), 'neg': -resid, 'pos': resid}        # residual
    arm_score_raw = {'any': np.abs(y_prob), 'neg': -y_prob, 'pos': y_prob}  # raw prediction

    for arm, (mask, labels) in build_labeled_subsets(qgi, fdr).items():
        m = mask.to_numpy()
        lab = labels[mask].to_numpy()
        au, ap = binary_metrics(lab, arm_score[arm][m])
        rau, rap = binary_metrics(lab, arm_score_raw[arm][m])
        out[f'{arm}_auroc'] = au
        out[f'{arm}_auprc'] = ap
        out[f'{arm}_raw_auroc'] = rau
        out[f'{arm}_raw_auprc'] = rap
        out[f'{arm}_n'] = int(m.sum())
        out[f'{arm}_n_pos'] = int(lab.sum())

    for arm in ARMS:
        bau, brau, bnp = balanced_auroc(g, arm, resid, y_prob, lib_deg, lib_col, query_col)
        out[f'{arm}_bal_auroc'] = bau
        out[f'{arm}_bal_raw_auroc'] = brau
        out[f'{arm}_bal_n_pos'] = bnp
    return out


def _present_genes(embed_path, cell_line=None):
    """Genes that have an embedding (for dropping imputed rows in legacy preds)."""
    with open(embed_path, 'rb') as f:
        emb = pickle.load(f)
    first = next(iter(emb.values()))
    if isinstance(first, dict):
        if cell_line in emb:
            return set(emb[cell_line].keys())
        return set().union(*(set(d.keys()) for d in emb.values()))
    return set(emb.keys())


def run(preds, dataset, out_csv=None, lib_col='library_gene', query_col='query_gene',
        embed_path=None):
    """Compute residual metrics per fold and write `out_csv` (+ mean/std rows).

    preds:   DataFrame or path of a `*_mlp_preds.csv` (needs fold, y_prob and
             the two gene columns; rows with source == 'imputed' are dropped).
    dataset: DataFrame or path of the task CSV supplying qGI and FDR.
    embed_path: only for preds files without a `source` column — imputed rows
             are then identified as pairs with a gene absent from the pickle.
    """
    if not isinstance(preds, pd.DataFrame):
        preds_path = Path(preds)
        preds = pd.read_csv(preds_path)
        if out_csv is None:
            out_csv = preds_path.with_name(preds_path.name.replace('_preds.csv', '_residual_metrics.csv'))
    if not isinstance(dataset, pd.DataFrame):
        dataset = pd.read_csv(dataset, usecols=[lib_col, query_col, 'qGI', 'FDR'])
    df_full = dataset[[lib_col, query_col, 'qGI', 'FDR']]
    lib_deg = arm_library_degree(df_full, lib_col)

    # drop train-mean-imputed rows (pairs with a gene lacking an embedding)
    n0 = len(preds)
    if 'source' in preds.columns:
        preds = preds[preds['source'] == 'model']
    elif embed_path is not None:
        cl = preds['cell_line'].iloc[0] if 'cell_line' in preds.columns else None
        genes = _present_genes(embed_path, cl)
        preds = preds[preds[lib_col].isin(genes) & preds[query_col].isin(genes)]
    else:
        print("  [warn] preds has no `source` column and no --embed-path given; "
              "imputed rows (if any) are NOT dropped")
    print(f"  dropped {n0 - len(preds):,} imputed rows ({len(preds):,} kept)")

    # attach qGI + FDR; (library, query) is unique per pair
    label_map = df_full.set_index([lib_col, query_col])[['qGI', 'FDR']]
    preds = preds.drop(columns=[c for c in ('qGI', 'FDR') if c in preds.columns])
    preds = preds.merge(label_map, on=[lib_col, query_col], how='left')
    n_nolabel = int(preds['qGI'].isna().sum())
    if n_nolabel:
        print(f"  dropped {n_nolabel:,} rows with no qGI/FDR label")
        preds = preds.dropna(subset=['qGI', 'FDR'])

    rows = []
    for fold in sorted(preds['fold'].unique(), key=lambda c: int(''.join(ch for ch in c if ch.isdigit()) or 0)):
        g = preds[preds['fold'] == fold].reset_index(drop=True)
        m = analyze_fold(g, lib_deg, lib_col, query_col)
        m['fold'] = fold
        rows.append(m)
        print(f"  {fold}: any_auprc={m['any_auprc']:.4f} any_auroc={m['any_auroc']:.4f} "
              f"(bal {m['any_bal_auroc']:.4f})  neg_auprc={m['neg_auprc']:.4f} "
              f"pos_auprc={m['pos_auprc']:.4f}")

    metrics_df = pd.DataFrame(rows)
    cols = ['fold'] + [c for c in metrics_df.columns if c != 'fold']
    metrics_df = metrics_df[cols]
    num = metrics_df.drop(columns=['fold']).select_dtypes(include=[np.number])
    agg = pd.DataFrame([num.mean(), num.std(ddof=0)])
    agg.insert(0, 'fold', ['mean', 'std'])
    metrics_df = pd.concat([metrics_df, agg], ignore_index=True)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_csv, index=False)
        print(f"Residual metrics saved to {out_csv}")

    print("\nResidual metrics (mean +/- std over folds):")
    mean_row = metrics_df[metrics_df['fold'] == 'mean'].iloc[0]
    std_row = metrics_df[metrics_df['fold'] == 'std'].iloc[0]
    for c in METRIC_COLS:
        if c in metrics_df.columns:
            print(f"  {c:18s}: {mean_row[c]:.4f} +/- {std_row[c]:.4f}")
    return metrics_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preds', required=True, help='*_mlp_preds.csv written by evaluate.py')
    ap.add_argument('--dataset', required=True, help='task CSV with qGI and FDR columns')
    ap.add_argument('--out', default=None, help='output CSV (default: next to --preds)')
    ap.add_argument('--lib-col', default='library_gene')
    ap.add_argument('--query-col', default='query_gene')
    ap.add_argument('--embed-path', default=None,
                    help='embedding pickle; only needed for preds without a `source` column')
    args = ap.parse_args()
    run(args.preds, args.dataset, args.out, args.lib_col, args.query_col, args.embed_path)


if __name__ == '__main__':
    main()
