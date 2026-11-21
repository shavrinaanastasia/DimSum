"""HW3 planning probe: do simple repeat-ordering signals beat Latest buy?
Approximate protocol (all buyers before cut-off warm, all assets eligible); planning only."""
import sys
import numpy as np
import pandas as pd

t = pd.read_csv('data/FAR-Trans/transactions.csv', parse_dates=['timestamp'])
t = t[t.transactionType.str.lower().str.startswith('b')]


def ndcg(rank, truth, k=10):
    g = sum(1 / np.log2(i + 2) for i, a in enumerate(rank[:k]) if a in truth)
    return g / sum(1 / np.log2(i + 2) for i in range(min(k, len(truth))))


cuts = sys.argv[1:] or ['2021-03-01', '2021-06-01', '2021-09-01', '2021-12-01', '2022-03-01', '2022-06-01']
res = []
for c in cuts:
    c = pd.Timestamp(c)
    past = t[t.timestamp < c].copy()
    fut = t[(t.timestamp >= c) & (t.timestamp < c + pd.Timedelta(days=30))]
    truth = fut[fut.customerID.isin(past.customerID.unique())].groupby('customerID').ISIN.apply(set)
    past = past[past.customerID.isin(truth.index)]
    past['age'] = (c - past.timestamp).dt.days
    days = past.groupby(['customerID', 'ISIN']).timestamp.apply(lambda s: np.sort(s.dt.normalize().unique()))
    g = past.groupby(['customerID', 'ISIN']).agg(last=('age', 'min'), n=('age', 'size'),
                                                n90=('age', lambda a: (a < 90).sum()),
                                                dec=('age', lambda a: np.exp(-a / 60).sum()))
    gaps = days.apply(lambda d: np.median(np.diff(d).astype('timedelta64[D]').astype(float)) if len(d) > 2 else np.nan)
    g['gap'] = gaps
    # expected next buy inside the window: last + gap - 30 <= age... i.e. due = gap - last in [-15, 30+15]
    due = g.gap - g['last']
    g['periodic'] = np.where(g.gap.notna() & (due > -g.gap) & (due < 45), 1.0, 0.0)
    g = g.reset_index()
    out = {'cutoff': c.date(), 'nonempty': len(truth)}
    scorers = {
        'latest_buy': -g['last'],
        'freq90_then_recency': g.n90 * 1000 - g['last'],
        'decay60': g.dec,
        'periodic_then_decay': g.periodic * 1000 + g.dec,
        'periodic_then_recency': g.periodic * 1000 - g['last'],
    }
    for name, s in scorers.items():
        g['s'] = s.values
        ranks = g.sort_values(['customerID', 's'], ascending=[True, False]).groupby('customerID').ISIN.apply(list)
        out[name] = np.mean([ndcg(ranks.get(u, []), tr) for u, tr in truth.items()])
    res.append(out)
pd.set_option('display.width', 250)
df = pd.DataFrame(res)
print(df.round(4).to_string(index=False))
print('mean', df.drop(columns=['cutoff', 'nonempty']).mean().round(4).to_dict())
