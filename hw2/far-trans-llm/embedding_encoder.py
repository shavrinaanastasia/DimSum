"""HW2 Encoder (DimSum, A. Shavrina): compressed LLM investor embeddings -> user-kNN scores.

Research question: how does compressing frozen-LLM investor representations to
128, 32, 8 or 2 dimensions (UMAP main, PCA side) change Top-10 nDCG@10 compared with
the uncompressed representation?

Pipeline (all inputs strictly before the cutoff):
  1. Each investor's as-of profile + pre-cutoff purchase history is serialised as text and
     encoded by frozen Qwen/Qwen3.5-0.8B (zero-shot PromptEOL prompt, last-token hidden
     state, 1024-d). Inference code: ../../src/embed.py (stages export/encode/assemble),
     run on a Colab T4 via ../../notebooks/encode_qwen35_colab.ipynb. This script loads the
     resulting matrix (--emb) and does NOT call the model again.
  2. The matrix (all users with a pre-cutoff buy, ~25k) is reduced with UMAP or PCA.
  3. For each evaluated user, the k=50 nearest investors in the reduced space vote for
     assets: score(i) = sum_j sim(u, j) * decayed_buys(j, i)  (half-life 365 days).
  4. The full eligible universe is ranked (repeat purchases allowed); ties and zero scores
     are ordered by the starter's popularity, then ISIN.

Same-information control: `behaviour_uknn` uses the identical neighbour rule on the
investors' decayed purchase vectors instead of LLM embeddings.
Hybrid ablation: `own_history_pop` (no neighbours) vs `hybrid_behaviour` vs `hybrid_llm_umap8`.

The provided starter (../far-trans) supplies users, cutoff, eligible assets, truth and
nDCG@10; its four baselines are written alongside via data_adapter.save_results.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from data_adapter import arguments, load, save_results

DIMS = (128, 32, 8, 2)


def decayed_matrix(events, users, item_ix, cutoff, half_life=365.0):
    """Users x items matrix of time-decayed pre-cutoff Buy counts (starter row policy)."""
    t0 = date.fromisoformat(cutoff)
    uix = {u: r for r, u in enumerate(users)}
    X = np.zeros((len(users), len(item_ix)), np.float32)
    for uid, evs in events.items():
        r = uix.get(uid)
        if r is None:
            continue
        for ts, isin, kind in evs:
            if kind == 'Buy' and ts < cutoff and isin in item_ix:
                age = (t0 - date.fromisoformat(ts)).days
                X[r, item_ix[isin]] += 0.5 ** (age / half_life)
    return X


def l2(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def reduce(E, method, d, seed=42):
    """Return (Z, metric). Fit on all pre-cutoff users only (no evaluation labels)."""
    if method == 'none':
        return l2(E), 'cos'
    if method == 'pca':
        from sklearn.decomposition import PCA
        return l2(PCA(n_components=d, random_state=seed).fit_transform(l2(E))), 'cos'
    import umap
    Z = umap.UMAP(n_components=d, n_neighbors=30, min_dist=0.0, metric='cosine',
                  random_state=seed).fit_transform(E)
    return Z.astype(np.float32), 'euc'


def knn_scores(Z, metric, rows, X, k=50):
    """rows: query row indices. Returns len(rows) x items score matrix."""
    out = np.zeros((len(rows), X.shape[1]), np.float32)
    for n, r in enumerate(rows):
        if metric == 'cos':
            sims = Z @ Z[r]
        else:
            sims = 1.0 / (1.0 + np.linalg.norm(Z - Z[r], axis=1))
        sims[r] = -np.inf                       # never use the user as their own neighbour
        nn = np.argpartition(-sims, k)[:k]
        out[n] = sims[nn] @ X[nn]
    return out


def rank(scores, items, pop, eligible):
    """Full-universe top ten; ties broken by starter popularity, then ISIN."""
    s = {isin: float(scores[j]) for j, isin in enumerate(items) if isin in eligible}
    return sorted(s, key=lambda i: (-s[i], -pop.get(i, 0), i))[:10]


def main():
    parser = arguments(__doc__)
    parser.add_argument('--emb', type=Path, required=True, help='qwen35_<cutoff>.npy from src/embed.py')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    t_all = time.perf_counter()
    bundle = load(args)
    starter, assets, history, truth, contexts, baseline, events, pop, manifest = bundle
    if any((args.output_dir / p).exists() for p in ('metrics.json', 'predictions.csv')):
        raise ValueError('Output directory already contains results; use a fresh directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_users = manifest['users']
    items = sorted(assets)
    item_ix = {i: j for j, i in enumerate(items)}

    # Embedding matrix built from pre-cutoff information only (see src/embed.py).
    E = np.load(args.emb).astype(np.float32)
    emb_users = Path(str(args.emb).replace('.npy', '.users.txt')).read_text(encoding='utf-8').split('\n')
    assert len(emb_users) == len(E)
    missing = [u for u in eval_users if u not in set(emb_users)]
    if missing:
        raise ValueError(f'{len(missing)} evaluated users have no embedding')
    uix = {u: r for r, u in enumerate(emb_users)}
    rows = [uix[u] for u in eval_users]
    X = decayed_matrix(events, emb_users, item_ix, args.cutoff)
    timings = {'load_and_neighbour_matrix_seconds': time.perf_counter() - t_all}

    def score_block(name, Z, metric):
        t = time.perf_counter()
        S = knn_scores(Z, metric, rows, X, args.k)
        timings[name + '_knn_seconds'] = time.perf_counter() - t
        return S

    scores = {}
    scores['behaviour_uknn'] = score_block('behaviour_uknn', l2(X), 'cos')
    for method in ('none', 'umap', 'pca'):
        for d in ((0,) if method == 'none' else DIMS):
            name = 'llm_uknn_full' if method == 'none' else f'llm_uknn_{method}{d}'
            t = time.perf_counter()
            Z, metric = reduce(E, method, d, args.seed)
            timings[name + '_reduce_seconds'] = time.perf_counter() - t
            scores[name] = score_block(name, Z, metric)
            print(f'{name}: reduce {timings[name + "_reduce_seconds"]:.1f}s', flush=True)

    # hybrid: own decayed history + neighbour votes + popularity (fixed weights, set before this run)
    own = X[rows]
    pop_vec = np.array([pop.get(i, 0) for i in items], np.float32)
    nrm = lambda v: (v - v.min(1, keepdims=True)) / (v.max(1, keepdims=True) - v.min(1, keepdims=True) + 1e-9)
    P = np.tile(pop_vec / (pop_vec.max() + 1e-9), (len(rows), 1))
    scores['own_history_pop'] = nrm(own) + 0.1 * P
    scores['hybrid_behaviour'] = nrm(own) + 0.35 * nrm(scores['behaviour_uknn']) + 0.1 * P
    scores['hybrid_llm_umap8'] = nrm(own) + 0.35 * nrm(scores['llm_uknn_umap8']) + 0.1 * P

    predictions = defaultdict(dict)
    for name, S in scores.items():
        for n, uid in enumerate(eval_users):
            predictions[uid][name] = rank(S[n], items, pop, set(contexts[uid]['eligible_asset_ids']))
    timings['total_seconds'] = time.perf_counter() - t_all
    metadata = {
        'encoder_model': 'Qwen/Qwen3.5-0.8B', 'encoder_revision_local_snapshot': args.revision,
        'encoder_inference': 'frozen, zero-shot PromptEOL last-token, fp16 on Colab T4 '
                             '(notebooks/encode_qwen35_colab.ipynb); max_len 384; left padding',
        'prompt_template': 'This investor record:\\n"{text}"\\nmeans in one word:"',
        'embedding_file': str(args.emb), 'embedded_users': len(emb_users),
        'reducers': {'umap': 'n_neighbors=30, min_dist=0, metric=cosine, random_state=%d' % args.seed,
                     'pca': 'on L2-normalised vectors, random_state=%d' % args.seed},
        'knn': {'k': args.k, 'item_weight': 'Buy count decayed with 365-day half-life'},
        'hybrid_weights': 'own 1.0, neighbours 0.35, popularity 0.1 (fixed before this run)',
        'timings_seconds': timings, 'invalid_outputs': 0,
    }
    save_results(args, bundle, predictions, [], 'embedding_encoder', metadata)
    print(json.dumps({k: round(v, 1) for k, v in timings.items()}, indent=1))


if __name__ == '__main__':
    main()
