"""HW3 failure analysis on the dev runs (nonempty users only; uses the saved per-user scores).

Groups each future buyer by (a) number of distinct assets bought before the cut-off
(1 / 2-4 / 5-9 / 10+) and (b) what they bought next: only repeats, mix, or only new assets.
Reports main vs Latest buy per group and how often main loses to Latest buy.

python hw3/failure.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'hw3' / 'outputs'
t = pd.read_csv(ROOT / 'data' / 'FAR-Trans' / 'transactions.csv', parse_dates=['timestamp'])
t = t[t.transactionType.str.lower().str.startswith('b')]

rows, loss_rows = [], []
for c in ['2021-06-01', '2021-09-01', '2021-12-01', '2022-06-01']:
    per = pd.read_csv(OUT / f'dev_fast_{c}' / 'per_user.csv')
    W = per.pivot_table(index='user_id', columns='method', values='ndcg_at_10')
    rel = per.groupby('user_id').relevant_count.first()
    W = W[rel.reindex(W.index) > 0]
    cut = pd.Timestamp(c)
    past = t[t.timestamp < cut].groupby('customerID').ISIN.apply(set)
    fut = t[(t.timestamp >= cut) & (t.timestamp < cut + pd.Timedelta(days=30))].groupby('customerID').ISIN.apply(set)
    g = pd.DataFrame(index=W.index)
    g['n_hist'] = [len(past.get(u, set())) for u in W.index]
    g['hist_bin'] = pd.cut(g.n_hist, [0, 1, 4, 9, 10 ** 6], labels=['1', '2-4', '5-9', '10+'])
    rep = [len(fut.get(u, set()) & past.get(u, set())) for u in W.index]
    new = [len(fut.get(u, set()) - past.get(u, set())) for u in W.index]
    g['next'] = np.select([np.array(new) == 0, np.array(rep) == 0], ['repeat only', 'new only'], 'mixed')
    g['main'], g['latest'] = W['main'], W['latest_buy']
    g['best_var'] = W[['rebuy', 'rebuy_tfidf', 'rebuy_assetllm', 'full']].max(1)
    for key in ['hist_bin', 'next']:
        for lvl, s in g.groupby(key, observed=True):
            rows.append({'cutoff': c, 'by': key, 'group': lvl, 'users': len(s),
                         'main': s.main.mean(), 'latest_buy': s.latest.mean(),
                         'diff': (s.main - s.latest).mean(),
                         'main_worse_share': (s.main < s.latest - 1e-12).mean(),
                         'main_better_share': (s.main > s.latest + 1e-12).mean()})
    loss_rows.append({'cutoff': c, 'users': len(g), 'main_worse': int((g.main < g.latest - 1e-12).sum()),
                      'main_better': int((g.main > g.latest + 1e-12).sum()),
                      'main_zero': int((g.main == 0).sum()),
                      'new_only_users': int((g.next == 'new only').sum()),
                      'new_only_main_zero': int(((g.next == 'new only') & (g.main == 0)).sum())})

res = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(res.round(4).to_string(index=False))
print(pd.DataFrame(loss_rows).to_string(index=False))
# pooled over the four dev sets (each user-window counted once)
pool = res.assign(wm=res.main * res.users, wl=res.latest_buy * res.users).groupby(['by', 'group'], observed=True)[
    ['users', 'wm', 'wl']].sum()
pool['main'], pool['latest_buy'] = pool.wm / pool.users, pool.wl / pool.users
pool['diff'] = pool.main - pool.latest_buy
print(pool[['users', 'main', 'latest_buy', 'diff']].round(4).to_string())
res.to_csv(OUT / 'failure_by_group.csv', index=False)
