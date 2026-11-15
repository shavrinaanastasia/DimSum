"""Tune iALS alone under the paper protocol (new assets only) on a non-test date.

python scripts/tune_ials.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data import load, make_split, ndcg10  # noqa: E402
from features import Signals, ials  # noqa: E402

d = load('data/FAR-Trans')
for cut, end in [('2020-08-23', '2021-02-23'), ('2021-08-01', '2022-02-01')]:
    sp = make_split(d, cut, end)
    sg = Signals(sp, mf=False)
    users = [u for u in sp.warm if sp.truth[u] - set(np.flatnonzero(sg.B[sg.uix[u]]))]

    def score(S):
        v = []
        for u in users:
            r = sg.uix[u]
            s = np.where(sg.B[r] > 0, -np.inf, S[r])
            v.append(ndcg10(list(np.argsort(-s)[:10]), sp.truth[u] - set(np.flatnonzero(sg.B[r]))))
        return np.mean(v)

    print(cut, 'popularity', round(score(np.tile(sg.pop[90], (len(sg.users), 1))), 4), flush=True)
    for k in (16, 64):
        for reg in (1, 10, 100):
            for alpha in (1, 10):
                for conf in ('bin', 'dec'):
                    t = time.time()
                    C = alpha * (sg.B if conf == 'bin' else np.log1p(sg.Xd365 * 4))
                    P, Q = ials(sg.B, C, k=k, reg=reg, iters=10)
                    print(cut, f'k={k} reg={reg} alpha={alpha} conf={conf}: {score(P @ Q.T):.4f} '
                          f'[{time.time() - t:.0f}s]', flush=True)
