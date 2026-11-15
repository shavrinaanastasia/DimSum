"""Paired bootstrap 95% CI of one method minus each provided baseline (all users and
nonempty-truth users).

python hw2/vs_baselines.py hw2/outputs/reranker_fullwarm reranker_llm_umap8
"""
import json
import sys
from pathlib import Path

import numpy as np

out, method = Path(sys.argv[1]), sys.argv[2]
rec = json.loads((out / 'per_user.json').read_text(encoding='utf-8'))
V, rel = {}, {}
for r in rec:
    V.setdefault(r['method'], {})[r['user_id']] = r['ndcg_at_10']
    rel[r['user_id']] = r['relevant_count']
rng = np.random.default_rng(0)
for scope in ('all', 'nonempty'):
    users = sorted(u for u in rel if scope == 'all' or rel[u] > 0)
    B = rng.integers(0, len(users), (10000, len(users)))
    m = np.array([V[method][u] for u in users])
    print(f'{scope} users ({len(users)}): {method} = {m.mean():.4f}')
    for b in ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell', *sys.argv[3:]):
        d = m - np.array([V[b][u] for u in users])
        lo, hi = np.quantile(d[B].mean(1), [0.025, 0.975])
        print(f'  minus {b:22s} {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]')
