"""HW3 hyper-parameter / feature-set selection on the VALIDATION window only.

Uses the cached D0 training frames (12 monthly windows before 2021-06-01): fits on windows
2..12 and scores the most recent window (2021-05-01..2021-05-31). Test truth is never read.
Metric: mean nDCG@10 over validation users with >= 1 positive among candidates.

python hw3/tune.py
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank3 as R  # noqa: E402

DISC = 1 / np.log2(np.arange(2, 12))


def ndcg_users(df, score):
    out = []
    d = df.assign(s=score)
    for _, g in d.groupby('user', sort=False):
        lab = g.label.values
        if lab.sum() == 0:
            continue
        top = lab[np.argsort(-g.s.values, kind='stable')][:10]
        out.append((top * DISC[:len(top)]).sum() / DISC[:min(int(lab.sum()), 10)].sum())
    return float(np.mean(out))


def main():
    cutoff = sys.argv[1] if len(sys.argv) > 1 else '2021-06-01'
    wins = R.windows(cutoff, 12)
    from data import load
    d = load(str(R.ROOT / 'data' / 'FAR-Trans'))
    # investor embedding never newer than the window's own cut-off (same rule as rerank3.py)
    frames = [R.train_frame(d, c, e, R.inv_for(c, cutoff)) for c, e in wins]
    va = frames[0].sort_values('user', kind='stable')
    tr = pd.concat(frames[1:], ignore_index=True).sort_values('user', kind='stable')
    gva = va.groupby('user', sort=False).size().values
    gtr = tr.groupby('user', sort=False).size().values
    cols = list(va.columns)
    base = dict(objective='lambdarank', metric='ndcg', eval_at=[10], learning_rate=0.03, num_leaves=31,
                min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambdarank_truncation_level=20, verbose=-1, seed=0, num_threads=8)
    specs = {'rebuy_tfidf': 'base+rebuy+asset_tfidf', 'full': 'base+rebuy+asset_llm+inv_llm',
             'all': 'base+rebuy+asset_tfidf+asset_llm+inv_llm'}
    grid = [
        {},
        {'num_leaves': 15}, {'num_leaves': 63}, {'num_leaves': 127, 'min_data_in_leaf': 100},
        {'min_data_in_leaf': 200}, {'min_data_in_leaf': 20},
        {'feature_fraction': 0.5}, {'lambdarank_truncation_level': 10},
        {'lambdarank_truncation_level': 50}, {'objective': 'rank_xendcg'},
        {'learning_rate': 0.01}, {'lambda_l2': 10.0},
    ]
    out = []
    for (sname, spec), over in itertools.product(specs.items(), grid):
        if sname != 'all' and over:
            continue
        feats = R.feats_for(spec, cols)
        p = {**base, **over}
        t = time.perf_counter()
        m = lgb.train(p, lgb.Dataset(tr[feats], tr.label, group=gtr), 4000,
                      valid_sets=[lgb.Dataset(va[feats], va.label, group=gva)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        s = ndcg_users(va, m.predict(va[feats], num_iteration=m.best_iteration))
        out.append({'features': sname, 'override': json.dumps(over), 'best_it': m.best_iteration,
                    'valid_ndcg': round(s, 5), 'sec': round(time.perf_counter() - t)})
        print(out[-1], flush=True)
    df = pd.DataFrame(out).sort_values('valid_ndcg', ascending=False)
    print(df.to_string(index=False))
    df.to_csv(Path(__file__).resolve().parent / 'outputs' / f'tune_{cutoff}.csv', index=False)


if __name__ == '__main__':
    main()
