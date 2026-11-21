"""Pooled paired bootstrap over the four dev sets (nonempty user-windows, resampled within dev set).

python hw3/pooled.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'outputs'
CUTS = ['2021-06-01', '2021-09-01', '2021-12-01', '2022-06-01']
PAIRS = [('main', 'latest_buy'), ('main', 'behaviour_hw2'), ('main', 'rebuy_tfidf'), ('main', 'rebuy'),
         ('rebuy_assetllm', 'rebuy_tfidf'), ('rebuy_assetllm', 'rebuy'), ('rebuy_invllm', 'rebuy'),
         ('full', 'rebuy'), ('rebuy', 'behaviour_hw2')]
W = {}
for c in CUTS:
    per = pd.read_csv(OUT / f'dev_fast_{c}' / 'per_user.csv')
    w = per.pivot_table(index='user_id', columns='method', values='ndcg_at_10')
    rel = per.groupby('user_id').relevant_count.first().reindex(w.index)
    W[c] = w[rel > 0]
rng = np.random.default_rng(0)
idx = {c: rng.integers(0, len(W[c]), (10000, len(W[c]))) for c in CUTS}
rows = []
for a, b in PAIRS:
    diffs = {c: (W[c][a] - W[c][b]).values for c in CUTS}
    n = sum(len(v) for v in diffs.values())
    boot = sum(diffs[c][idx[c]].sum(1) for c in CUTS) / n
    rows.append({'a': a, 'b': b, 'n': n, 'mean_diff': sum(v.sum() for v in diffs.values()) / n,
                 'ci_lo': np.quantile(boot, 0.025), 'ci_hi': np.quantile(boot, 0.975),
                 'wins_a': int(sum((v > 1e-12).sum() for v in diffs.values())),
                 'wins_b': int(sum((v < -1e-12).sum() for v in diffs.values()))})
print(pd.DataFrame(rows).round(4).to_string(index=False))
