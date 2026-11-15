"""End-to-end pipeline: baselines, hybrid, learned reranker; nDCG@10 on the test split.

python src/run.py --data data/FAR-Trans --cutoff 2022-06-01 --end 2022-12-01 \
    --rep behaviour --reduce none --dim 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data import load, make_split, evaluate, bootstrap_ci  # noqa: E402
from features import Signals, candidates, feature_frame  # noqa: E402
import rep as reps  # noqa: E402

DROP = ['user', 'item', 'label']


def train_cutoffs(cutoff: pd.Timestamp, end: pd.Timestamp, n: int, step_months: int):
    """Earlier splits with the same window length whose windows end on/before the test cut-off.
    The first one (closest to the test) is the validation split."""
    months = (end.year - cutoff.year) * 12 + end.month - cutoff.month
    assert months >= 1 and cutoff.day == end.day, 'window must be whole months'
    win = pd.DateOffset(months=months)
    return [(cutoff - win - pd.DateOffset(months=step_months * k), cutoff - pd.DateOffset(months=step_months * k))
            for k in range(n)]


def build(d, cutoff, end, args, log):
    t = time.time()
    sp = make_split(d, str(cutoff.date()), str(end.date()))
    sg = Signals(sp, customers=d['customers'], mf=not args.no_mf)
    if args.protocol == 'new':
        # FAR-Trans paper protocol: a purchase counts only if the user never bought that asset
        # before the cut-off; already-bought assets are never recommended
        for u in list(sp.truth):
            if u in sg.uix:
                sp.truth[u] = sp.truth[u] - set(np.flatnonzero(sg.B[sg.uix[u]]))
        sp.warm = [u for u in sp.warm if sp.truth[u]]
    users = sp.warm
    R = reps.behaviour(sg) if args.rep == 'behaviour' else reps.llm(sg, args.emb_dir, args.rep)
    Z, metric = reps.reduce(R, args.reduce, args.dim)
    knn = sg.userknn(Z, users, k=args.k, metric=metric)
    feats = {'uknn': knn}
    if args.rep != 'behaviour':          # keep the classical neighbour signal as a separate feature
        feats['uknn_behaviour'] = sg.userknn(reps.behaviour(sg), users, k=args.k)
    cands = candidates(sg, users, knn, n=args.n_cand, new_only=args.protocol == 'new')
    df = feature_frame(sg, users, cands, feats)
    df['label'] = [int(i in sp.truth[u]) for u, i in zip(df.user, df.item)]
    rec = df.groupby('user').label.sum()
    recall = np.mean([rec[u] / len(sp.truth[u]) for u in users])
    any_hit = np.mean([rec[u] > 0 for u in users])
    log(f'split {cutoff.date()}..{end.date()}: items {len(sp.items)}, warm {len(users)}, cold {len(sp.cold)}, '
        f'rows {len(df)}, cand recall {recall:.3f}, users with a hit {any_hit:.3f} [{time.time()-t:.0f}s]')
    return sp, sg, df, knn


def topk(scores_by_user: dict, n_items: int, fallback: np.ndarray, excl: dict | None = None):
    """excl: user -> set of item indices that must not be recommended (paper protocol)."""
    out = {}
    for u, (items, s) in scores_by_user.items():
        ex = excl.get(u, set()) if excl else set()
        if ex:
            keep = np.array([i not in ex for i in items])
            items, s = items[keep], s[keep]
        order = list(items[np.argsort(-s)][:10])
        for f in fallback:              # pad with popular items if fewer than 10 candidates
            if len(order) == 10:
                break
            if f not in order and f not in ex:
                order.append(int(f))
        out[u] = [int(i) for i in order]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/FAR-Trans')
    ap.add_argument('--cutoff', default='2022-06-01')
    ap.add_argument('--end', default='2022-12-01')
    ap.add_argument('--rep', default='behaviour', help="'behaviour' or an embedding tag from embed.py")
    ap.add_argument('--emb-dir', default='outputs/emb')
    ap.add_argument('--reduce', default='none', choices=['none', 'pca', 'umap'])
    ap.add_argument('--dim', type=int, default=0)
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--n-train', type=int, default=4)
    ap.add_argument('--step-months', type=int, default=3)
    ap.add_argument('--out', default='outputs/runs')
    ap.add_argument('--baselines', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--protocol', default='all', choices=['all', 'new'],
                    help="'all': every purchase in the window counts (repeats included); "
                         "'new': FAR-Trans paper protocol, only assets the user never bought before")
    ap.add_argument('--n-cand', type=int, default=50, help='top-n per candidate source')
    ap.add_argument('--no-mf', action='store_true', help='disable the iALS source/feature')
    ap.add_argument('--tag', default='', help='suffix for the output name')
    args = ap.parse_args()
    np.random.seed(0)
    name = f'{args.rep}_{args.reduce}{args.dim or ""}_{args.cutoff}'
    if args.protocol == 'new':
        name += '_new'
    if args.tag:
        name += '_' + args.tag
    os.makedirs(args.out, exist_ok=True)
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)

    d = load(args.data)
    t0, te = pd.Timestamp(args.cutoff), pd.Timestamp(args.end)
    tr = train_cutoffs(t0, te, args.n_train, args.step_months)
    log(f'run {name}; train/valid cut-offs: {[str(c.date()) for c, _ in tr]} (first = validation)')
    built = [build(d, c, e, args, log) for c, e in tr]
    sp, sg, test_df, knn = build(d, t0, te, args, log)
    users = sp.warm
    fallback = np.argsort(-sg.pop[90])[:60]
    excl = ({u: set(np.flatnonzero(sg.B[sg.uix[u]])) for u in users} if args.protocol == 'new' else None)
    res = {}

    def report(key, recs):
        v = evaluate(recs, sp, users)
        lo, hi = bootstrap_ci(v)
        res[key] = {'ndcg10': float(v.mean()), 'ci95': [lo, hi], 'per_user': v.tolist()}
        log(f'  {key:28s} nDCG@10 {v.mean():.4f}  [{lo:.4f}, {hi:.4f}]')
        return v

    if args.baselines:
        N = len(sp.items)
        rng = np.random.default_rng(0)
        ex = excl or {}
        report('random', {u: [int(i) for i in rng.permutation(N) if i not in ex.get(u, ())][:10] for u in users})
        report('popularity_90d', topk({u: (np.arange(N), sg.pop[90].copy()) for u in users}, N, fallback, excl))
        report('cbf_attributes', topk({u: (np.arange(N), sg.F @ (sg.X[sg.uix[u]] @ sg.F) + 1e-6 * sg.pop[100000])
                                       for u in users}, N, fallback, excl))
        if args.protocol == 'all':
            report('repeat', topk({u: (np.arange(N), sg.X[sg.uix[u]] + 1e-3 * sg.pop[90]) for u in users},
                                  N, fallback))
        report('userknn_only', topk({u: (np.arange(N), knn[u] + 1e-6 * sg.pop[100000]) for u in users},
                                    N, fallback, excl))
        report('itemknn_only', topk({u: (np.arange(N), sg.B[sg.uix[u]] @ sg.S_item + 1e-6 * sg.pop[100000])
                                     for u in users}, N, fallback, excl))
        report('ease_only', topk({u: (np.arange(N), sg.B[sg.uix[u]] @ sg.W_ease + 1e-6 * sg.pop[100000])
                                  for u in users}, N, fallback, excl))
        if sg.mf is not None:
            report('ials_only', topk({u: (np.arange(N), sg.mf[sg.uix[u]]) for u in users}, N, fallback, excl))

    # hand-weighted hybrid (no learning): decayed repeat + neighbour score + recent popularity
    nrm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-9)
    report('hybrid_fixed', topk({u: (np.arange(len(sp.items)),
                                     nrm(sg.Xd365[sg.uix[u]]) + 0.35 * nrm(knn[u]) + 0.1 * nrm(sg.pop[90]))
                                 for u in users}, len(sp.items), fallback, excl))

    # learned reranker: LightGBM LambdaRank on earlier splits
    import lightgbm as lgb
    feats = [c for c in test_df.columns if c not in DROP]
    def ds(df):
        df = df.sort_values('user', kind='stable')
        return df, df.groupby('user', sort=False).size().values
    params = dict(objective='lambdarank', metric='ndcg', eval_at=[10], learning_rate=0.03,
                  num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambdarank_truncation_level=20, verbose=-1, seed=0)
    va, gva = ds(built[0][2])
    trn = pd.concat([b[2] for b in built[1:]]); trn, gtr = ds(trn)
    m = lgb.train(params, lgb.Dataset(trn[feats], trn.label, group=gtr), 2000,
                  valid_sets=[lgb.Dataset(va[feats], va.label, group=gva)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    best = m.best_iteration
    log(f'  reranker: best iteration {best}, validation nDCG@10 (candidate-level) {m.best_score["valid_0"]["ndcg@10"]:.4f}')
    full = pd.concat([b[2] for b in built]); full, gfull = ds(full)
    test_df['score'] = 0.0
    for seed in range(args.seeds):        # seed-averaged ensemble, refit on all training splits
        m = lgb.train({**params, 'seed': seed}, lgb.Dataset(full[feats], full.label, group=gfull), best)
        test_df['score'] += m.predict(test_df[feats]) / args.seeds
    recs = topk({u: (g.item.values, g.score.values) for u, g in test_df.groupby('user')}, len(sp.items),
                fallback, excl)
    report('reranker', recs)
    imp = pd.Series(m.feature_importance('gain'), feats).sort_values(ascending=False)
    log('  top features (gain): ' + ', '.join(f'{k} {v/imp.sum():.2f}' for k, v in imp.head(10).items()))

    pd.DataFrame([{'user': u, 'rank': r + 1, 'ISIN': sp.items[i]} for u, l in recs.items() for r, i in enumerate(l)]
                 ).to_csv(os.path.join(args.out, f'{name}_top10.csv'), index=False)
    json.dump({'args': vars(args), 'n_warm': len(users), 'n_cold_excluded': len(sp.cold), 'results': res,
               'log': lines}, open(os.path.join(args.out, f'{name}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
