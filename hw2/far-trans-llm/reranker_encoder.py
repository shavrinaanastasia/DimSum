"""HW2 Encoder inside a learned reranker (DimSum, A. Shavrina).

The compressed Qwen3.5 investor embedding (none / UMAP / PCA, d = 128, 32, 8, 2) produces a
user-kNN score for every asset; that score and its rank are added as features to a
LightGBM LambdaRank reranker (the team pipeline in ../../src). The control
`reranker_behaviour` is the identical reranker without the LLM feature, so the
difference between the two isolates what the LLM representation contributes.

Protocol (starter defaults): users, cutoff, eligible assets and nDCG@10 come from the
provided starter via data_adapter; the four provided baselines are written alongside.

Leakage rules:
  * Every feature for the test cutoff uses rows dated before 2021-06-01 only.
  * The reranker is trained on three earlier one-month windows (cutoffs 2021-03-01,
    2021-04-01, 2021-05-01; the last one is the early-stopping validation split). Their
    labels end on or before the test cutoff.
  * Training splits use the embedding file encoded at 2021-03-01 (texts built from rows
    before 2021-03-01), i.e. never later than their own cutoff. Users who first bought
    after 2021-03-01 have no embedding there and get a missing LLM feature (LightGBM
    handles NaN). The test split uses the file encoded at 2021-06-01.
  * Candidates = union of top-50 from own past buys, purchase-vector user-kNN, item-kNN,
    EASE, iALS and 90-day popularity; identical for every variant (LLM is a feature only).

python reranker_encoder.py --data-dir ../../data/FAR-Trans --output-dir ../outputs/reranker100 \
    --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from data import load as load_raw, make_split  # noqa: E402
from features import Signals, candidates, feature_frame  # noqa: E402

from data_adapter import arguments, load, save_results  # noqa: E402
from embedding_encoder import reduce  # noqa: E402

DIMS = (128, 32, 8, 2)
VARIANTS = [('none', 0)] + [('umap', d) for d in DIMS] + [('pca', d) for d in DIMS]
TRAIN = [('2021-05-01', '2021-06-01'), ('2021-04-01', '2021-05-01'), ('2021-03-01', '2021-04-01')]
DROP = ['user', 'item', 'label']


def vname(method, d):
    return 'reranker_llm_full' if method == 'none' else f'reranker_llm_{method}{d}'


def load_emb(emb_dir, cutoff):
    E = np.load(emb_dir / f'qwen35_{cutoff}.npy').astype(np.float32)
    users = (emb_dir / f'qwen35_{cutoff}.users.txt').read_text(encoding='utf-8').split()
    assert len(users) == len(E)
    return E, users


def reduced(E, tag, method, d, seed, cache):
    """Reduction is unsupervised and fit on all pre-cutoff users; cached across runs."""
    f = cache / f'{tag}_{method}{d}_s{seed}.npz'
    if f.exists():
        z = np.load(f)
        return z['Z'], str(z['metric'])
    Z, metric = reduce(E, method, d, seed)
    np.savez(f, Z=Z, metric=metric)
    return Z, metric


def llm_knn(sg, Z, metric, emb_users, query, k=50):
    """Neighbour score over items from the k most similar investors in Z (neighbours'
    365-day-decayed buys). NaN vector for users without an embedding at this cutoff."""
    pos = {u: i for i, u in enumerate(emb_users)}
    have = [u for u in sg.users if u in pos]
    Zs = Z[[pos[u] for u in have]]
    if metric == 'cos':
        Zs = Zs / (np.linalg.norm(Zs, axis=1, keepdims=True) + 1e-9)
    Xs = sg.Xd365[[sg.uix[u] for u in have]]
    hix = {u: i for i, u in enumerate(have)}
    out = {}
    nan = np.full(sg.Xd365.shape[1], np.nan, np.float32)
    for u in query:
        r = hix.get(u)
        if r is None:
            out[u] = nan
            continue
        sims = Zs @ Zs[r] if metric == 'cos' else 1.0 / (1.0 + np.linalg.norm(Zs - Zs[r], axis=1))
        sims[r] = -np.inf
        nn = np.argpartition(-sims, k)[:k]
        out[u] = sims[nn] @ Xs[nn]
    return out


def add_llm(df, scores, name):
    """Add score + rank columns for one LLM variant (rank NaN when the score is NaN)."""
    s_col, r_col = np.empty(len(df), np.float32), np.empty(len(df), np.float32)
    for u, idx in df.groupby('user', sort=False).indices.items():
        s = scores[u]
        items = df.item.values[idx]
        if np.isnan(s[0]):
            s_col[idx] = np.nan
            r_col[idx] = np.nan
        else:
            s_col[idx] = s[items]
            r_col[idx] = (-s).argsort().argsort()[items]
    df[name] = s_col
    df[name + '_rank'] = r_col


def main():
    parser = arguments(__doc__)
    parser.add_argument('--emb-dir', type=Path, default=ROOT / 'outputs' / 'emb')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lgb-seeds', type=int, default=3)
    parser.add_argument('--variants', default='all', help="'all' or 'behaviour' (control only)")
    args = parser.parse_args()
    t_all = time.perf_counter()
    bundle = load(args)
    starter, assets, history, truth, contexts, baseline, events, pop, manifest = bundle
    if any((args.output_dir / p).exists() for p in ('metrics.json', 'predictions.csv')):
        raise ValueError('Output directory already contains results; use a fresh directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'hw2' / 'outputs' / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    eval_users = manifest['users']
    assert args.cutoff == '2021-06-01', 'training windows are set for the starter default cutoff'
    timings = {}
    variants = VARIANTS if args.variants == 'all' else []

    d = load_raw(str(args.data_dir))
    E_tr, U_tr = load_emb(args.emb_dir, '2021-03-01')
    E_te, U_te = load_emb(args.emb_dir, '2021-06-01')

    def frame(cutoff, end, users_fn, E, U, tag):
        t = time.perf_counter()
        sp = make_split(d, cutoff, end)
        sg = Signals(sp, customers=d['customers'])
        users = users_fn(sp, sg)
        bknn = sg.userknn(sg.Xd365, users, k=args.k)
        df = feature_frame(sg, users, candidates(sg, users, bknn, n=50), {'uknn': bknn})
        for method, dim in variants:
            Z, metric = reduced(E, tag, method, dim, args.seed, cache)
            add_llm(df, llm_knn(sg, Z, metric, U, users, args.k), 'llm')
            df = df.rename(columns={'llm': vname(method, dim), 'llm_rank': vname(method, dim) + '_rank'})
        df['label'] = [int(i in sp.truth.get(u, ())) for u, i in zip(df.user, df.item)]
        timings[f'features_{cutoff}_seconds'] = time.perf_counter() - t
        print(f'split {cutoff}: users {len(users)}, rows {len(df)}, positives {df.label.sum()} '
              f'[{time.perf_counter() - t:.0f}s]', flush=True)
        return sp, sg, df

    built = [frame(c, e, lambda sp, sg: sp.warm, E_tr, U_tr, 'qwen35_2021-03-01')[2] for c, e in TRAIN]
    end = str((pd.Timestamp(args.cutoff) + pd.Timedelta(days=args.horizon_days)).date())
    sp, sg, test = frame(args.cutoff, end, lambda sp, sg: [u for u in eval_users if u in sg.uix],
                         E_te, U_te, 'qwen35_2021-06-01')
    no_signal = [u for u in eval_users if u not in sg.uix]
    print(f'{len(no_signal)} evaluated users without an eligible pre-cutoff buy -> popularity order')

    import lightgbm as lgb
    params = dict(objective='lambdarank', metric='ndcg', eval_at=[10], learning_rate=0.03,
                  num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambdarank_truncation_level=20, verbose=-1, seed=0)

    def ds(df):
        df = df.sort_values('user', kind='stable')
        return df, df.groupby('user', sort=False).size().values

    va, gva = ds(built[0])
    trn, gtr = ds(pd.concat(built[1:]))
    full, gfull = ds(pd.concat(built))
    base = [c for c in test.columns if c not in DROP and not c.startswith('reranker_llm_')]
    llm_cols = {vname(m, dd): [vname(m, dd), vname(m, dd) + '_rank'] for m, dd in variants}
    configs = [('reranker_behaviour', base)] + [(n, base + cols) for n, cols in llm_cols.items()]

    pop_order = sorted(pop, key=lambda i: (-pop[i], i))
    predictions = defaultdict(dict)
    info = {}
    for name, feats in configs:
        t = time.perf_counter()
        m = lgb.train(params, lgb.Dataset(trn[feats], trn.label, group=gtr), 2000,
                      valid_sets=[lgb.Dataset(va[feats], va.label, group=gva)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        best = max(m.best_iteration, 1)
        score = np.zeros(len(test))
        for s in range(args.lgb_seeds):  # seed-averaged refit on all three training windows
            ms = lgb.train({**params, 'seed': s}, lgb.Dataset(full[feats], full.label, group=gfull), best)
            score += ms.predict(test[feats]) / args.lgb_seeds
        imp = pd.Series(ms.feature_importance('gain'), feats)
        info[name] = {'best_iteration': best, 'valid_ndcg10_candidates': m.best_score['valid_0']['ndcg@10'],
                      'llm_feature_gain_share': float(imp[[f for f in feats if f.startswith('reranker_llm_')]].sum()
                                                      / imp.sum())}
        tt = test.assign(score=score)
        for u, g in tt.groupby('user', sort=False):
            elig = set(contexts[u]['eligible_asset_ids'])
            order = [sp.items[i] for i in g.item.values[np.argsort(-g.score.values, kind='stable')]]
            top = [i for i in order if i in elig][:10]
            for i in pop_order:          # pad with starter popularity if fewer than ten candidates
                if len(top) == 10:
                    break
                if i in elig and i not in top:
                    top.append(i)
            predictions[u][name] = top
        for u in no_signal:
            elig = set(contexts[u]['eligible_asset_ids'])
            predictions[u][name] = [i for i in pop_order if i in elig][:10]
        timings[name + '_train_predict_seconds'] = time.perf_counter() - t
        print(f'{name}: best it {best}, valid {info[name]["valid_ndcg10_candidates"]:.4f}, '
              f'LLM gain share {info[name]["llm_feature_gain_share"]:.3f} [{time.perf_counter() - t:.0f}s]',
              flush=True)

    timings['total_seconds'] = time.perf_counter() - t_all
    metadata = {
        'encoder_model': 'Qwen/Qwen3.5-0.8B', 'encoder_revision_local_snapshot': args.revision,
        'encoder_inference': 'frozen, zero-shot PromptEOL last-token, fp16 on Colab T4 '
                             '(notebooks/encode_qwen35_colab.ipynb); max_len 384; left padding',
        'prompt_template': 'This investor record:\\n"{text}"\\nmeans in one word:"',
        'embedding_files': {'train': 'qwen35_2021-03-01.npy', 'test': 'qwen35_2021-06-01.npy'},
        'reducers': {'umap': f'n_neighbors=30, min_dist=0, metric=cosine, random_state={args.seed}',
                     'pca': f'on L2-normalised vectors, random_state={args.seed}'},
        'reranker': {'model': 'LightGBM lambdarank', 'params': params, 'lgb_seeds': args.lgb_seeds,
                     'train_windows': TRAIN, 'validation_window': TRAIN[0]},
        'candidate_policy': 'union of top-50 per source: own past buys, purchase-vector user-kNN, item-kNN, '
                            'EASE, iALS, 90-day popularity; filtered to the starter eligible set; padded '
                            'with starter popularity',
        'users_without_signal': len(no_signal), 'variants': info,
        'timings_seconds': timings, 'invalid_outputs': 0,
    }
    save_results(args, bundle, predictions, [], 'reranker_encoder', metadata)
    print({k: round(v, 1) for k, v in timings.items()})


if __name__ == '__main__':
    main()
