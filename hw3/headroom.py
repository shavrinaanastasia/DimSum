"""HW3 headroom analysis: where can nDCG@10 still come from?

For several 30-day windows (repeat purchases allowed, warm users = bought before cut-off)
it reports: share of users with future buys, share of future-bought items the user had
bought before (repeat), the Latest-buy nDCG, and two oracles:
  * repeat oracle  - perfect ordering of the user's own past buys only
  * full oracle    - perfect ranking (=1 for every nonempty user)
Approximate teaching protocol (all assets eligible); for planning only, not a reported result.
"""
import sys
import numpy as np
import pandas as pd

t = pd.read_csv('data/FAR-Trans/transactions.csv', parse_dates=['timestamp'])
t = t[t.transactionType.str.lower().str.startswith('b')]


def ndcg(rank, truth, k=10):
    g = sum(1 / np.log2(i + 2) for i, a in enumerate(rank[:k]) if a in truth)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(truth))))
    return g / ideal if ideal else 0.0


cuts = sys.argv[1:] or ['2021-03-01', '2021-06-01', '2021-09-01', '2021-12-01', '2022-03-01', '2022-06-01', '2022-09-01']
rows = []
for c in cuts:
    c = pd.Timestamp(c)
    e = c + pd.Timedelta(days=30)
    past = t[t.timestamp < c]
    fut = t[(t.timestamp >= c) & (t.timestamp < e)]
    users = past.customerID.unique()
    truth = fut[fut.customerID.isin(users)].groupby('customerID').ISIN.apply(set)
    last = past.sort_values('timestamp').groupby('customerID').ISIN.apply(lambda s: list(dict.fromkeys(s[::-1])))
    rep_items = new_items = 0
    lb, ro = [], []
    for u, tr in truth.items():
        hist = last.get(u, [])
        hs = set(hist)
        rep_items += len(tr & hs)
        new_items += len(tr - hs)
        lb.append(ndcg(hist, tr))
        rep = [a for a in hist if a in tr]
        ro.append(ndcg(rep, tr))
    n, ne = len(users), len(truth)
    rows.append(dict(cutoff=c.date(), warm=n, nonempty=ne, share=ne / n,
                     repeat_item_share=rep_items / max(1, rep_items + new_items),
                     users_any_repeat=np.mean([r > 0 for r in ro]),
                     latest_nonempty=np.mean(lb), latest_all=np.sum(lb) / n,
                     repeat_oracle_nonempty=np.mean(ro), repeat_oracle_all=np.sum(ro) / n,
                     full_oracle_all=ne / n))
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).round(4).to_string(index=False))
