"""Paired bootstrap comparison of a reference method against all others, per dev run.

python hw3/compare.py hw3/outputs/dev_fast_2021-06-01 [reference=main]
Prints all-user and nonempty means (4 dp) and the paired 95% CI of (reference - other).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


def main():
    run = Path(sys.argv[1])
    ref = sys.argv[2] if len(sys.argv) > 2 else 'main'
    f = run / 'per_user.csv'
    per = pd.read_csv(f) if f.exists() else pd.DataFrame(
        __import__('json').loads((run / 'per_user.json').read_text(encoding='utf-8')))
    W = per.pivot_table(index='user_id', columns='method', values='ndcg_at_10')
    rel = per.groupby('user_id').relevant_count.first().reindex(W.index)
    ne = rel > 0
    rows = []
    for m in W.columns:
        r = {'method': m, 'all': W[m].mean(), 'nonempty': W.loc[ne, m].mean()}
        if m != ref:
            dn = (W.loc[ne, ref] - W.loc[ne, m]).values
            lo, hi = ci(dn)
            r.update({'d_nonempty': dn.mean(), 'ci_lo': lo, 'ci_hi': hi,
                      'd_all': dn.sum() / len(W)})
        rows.append(r)
    out = pd.DataFrame(rows).sort_values('all', ascending=False)
    print(f'{run.name}: users {len(W)}, nonempty {int(ne.sum())}, reference {ref}')
    print(out.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
