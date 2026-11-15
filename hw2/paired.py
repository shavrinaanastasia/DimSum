"""Paired comparison of each compressed variant with the uncompressed LLM embedding,
on users with nonempty truth (the HW1 population), using the HW1 pre-registered rule:
non-inferior at d if the lower bound of the one-sided bootstrap CI of the mean paired
difference, at level 1 - 0.05/4 (Bonferroni over four d), exceeds -0.01 nDCG@10.

python hw2/paired.py hw2/outputs/encoder_fullwarm
"""
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
rec = json.loads((out / 'per_user.json').read_text(encoding='utf-8'))
users = sorted({r['user_id'] for r in rec if r['relevant_count'] > 0})
ix = {u: n for n, u in enumerate(users)}
V = {}
for r in rec:
    if r['user_id'] in ix:
        V.setdefault(r['method'], np.zeros(len(users)))[ix[r['user_id']]] = r['ndcg_at_10']
ref_name = sys.argv[2] if len(sys.argv) > 2 else 'llm_uknn_full'
ref = V[ref_name]
rng = np.random.default_rng(0)
B = rng.integers(0, len(users), (10000, len(users)))
alpha = 0.05 / 4
fam = ref_name.rsplit('_', 1)[0]          # llm_uknn or reranker_llm
names = [f'{fam}_{m}{d}' for m in ('umap', 'pca') for d in (128, 32, 8, 2)]
names += ['behaviour_uknn' if fam == 'llm_uknn' else 'reranker_behaviour']
print(f'{len(users)} users with nonempty truth; reference {ref_name} = {ref.mean():.4f}')
print(f'{"variant":22s} {"nDCG":>7s} {"delta":>8s} {"one-sided LB":>13s} {"two-sided 95% CI":>19s} {"non-inferior (-0.01)":>21s}')
for name in names:
    d = V[name] - ref
    bm = d[B].mean(1)
    lb = np.quantile(bm, alpha)
    lo, hi = np.quantile(bm, [0.025, 0.975])
    print(f'{name:22s} {V[name].mean():7.4f} {d.mean():+8.4f} {lb:+13.4f} [{lo:+.4f}, {hi:+.4f}] '
          f'{"yes" if lb > -0.01 else "NO":>21s}')
