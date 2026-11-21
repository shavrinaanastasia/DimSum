"""HW3 personal check: one user's nDCG@10 recomputed independently (pandas, not the starter).

Picks the nonempty user of `method` with the most relevant items (ties: highest score) unless a
user ID is given, recomputes truth from transactions.csv, reads the saved Top-10 and computes
binary nDCG@10 by hand. Also prints the user's latest input date (must be < cut-off).

python hw3/check_one_user.py hw3/outputs/T100_full main [user_id]
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

out, method = Path(sys.argv[1]), sys.argv[2]
cutoff, end = '2021-06-01', '2021-07-01'
per_user = json.loads((out / 'per_user.json').read_text(encoding='utf-8'))
rows = [r for r in per_user if r['method'] == method and r['relevant_count'] > 0]
if len(sys.argv) > 3:
    uid = sys.argv[3]
else:
    uid = sorted(rows, key=lambda r: (-r['relevant_count'], -r['ndcg_at_10'], r['user_id']))[0]['user_id']
saved = next(r for r in rows if r['user_id'] == uid)

tx = pd.read_csv('data/FAR-Trans/transactions.csv').drop_duplicates()
tx = tx[tx.transactionID >= 0]
pred = pd.read_csv(out / 'predictions.csv')
top10 = pred[(pred.method == method) & (pred.user_id == uid)].sort_values('rank').ISIN.tolist()
eligible = set(json.loads(Path('hw2/outputs/baseline/protocol.json').read_text())['eligible_assets'])
fut = tx[(tx.customerID == uid) & (tx.transactionType == 'Buy') &
         (tx.timestamp.str[:10] >= cutoff) & (tx.timestamp.str[:10] < end)]
truth = set(fut.ISIN) & eligible
past = tx[(tx.customerID == uid) & (tx.timestamp.str[:10] < cutoff)]

print(f'user {uid[:8]}..., method {method}')
print('past Buy assets (before cutoff):', sorted(set(past[past.transactionType == "Buy"].ISIN)))
print('future Buy assets in window (eligible):', sorted(truth))
for r, isin in enumerate(top10, 1):
    hit = isin in truth
    print(f'  rank {r:2d} {isin} {"HIT  gain 1/log2(%d) = %.6f" % (r + 1, 1 / math.log2(r + 1)) if hit else ""}')
dcg = sum(1 / math.log2(r + 1) for r, i in enumerate(top10, 1) if i in truth)
idcg = sum(1 / math.log2(r + 1) for r in range(1, min(10, len(truth)) + 1))
print(f'hand DCG {dcg:.6f} / IDCG {idcg:.6f} = nDCG@10 {dcg / idcg:.6f}; saved by starter: {saved["ndcg_at_10"]:.6f}')
print('relevant_count saved', saved['relevant_count'], '| recomputed', len(truth))
print('latest input transaction date for this user:', past.timestamp.max(), '(must be < cutoff)')
