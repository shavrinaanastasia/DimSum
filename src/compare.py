"""Paired comparison of reranker runs on the same users (2022-06-01 split).

Prints each run's nDCG@10 and the paired bootstrap 95% CI of its difference
against a reference run (default: behaviour_none reranker) and the CBF baseline.

python src/compare.py
"""
import glob
import json
import os

import numpy as np

RUNS = 'outputs/runs'
REF = 'behaviour_none_2022-06-01'


def per_user(name, method='reranker'):
    return np.array(json.load(open(os.path.join(RUNS, name + '.json')))['results'][method]['per_user'])


def paired_ci(a, b, n=2000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ref = per_user(REF)
    cbf = per_user(REF, 'cbf_attributes')
    print(f'{"run":34s} {"nDCG@10":>8s}  {"vs behaviour (paired 95% CI)":>32s}  {"vs CBF":>8s}')
    # outputs/smoke holds the random-vector control (same pipeline, random 'embeddings')
    for f in sorted(glob.glob(os.path.join(RUNS, '*_2022-06-01.json'))) + glob.glob('outputs/smoke/*.json'):
        name = os.path.basename(f)[:-5]
        x = np.array(json.load(open(f))['results']['reranker']['per_user'])
        assert len(x) == len(ref)
        m, lo, hi = paired_ci(x, ref)
        print(f'{name:34s} {x.mean():8.4f}  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  {x.mean() - cbf.mean():+8.4f}')


if __name__ == '__main__':
    main()
