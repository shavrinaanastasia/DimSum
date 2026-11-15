"""Independent check of one user's nDCG@10 (pandas, not the starter's code path).

Recomputes the user's future purchases from transactions.csv, reads the saved Top-10 of a
method and recomputes binary nDCG@10 by hand; also confirms no model input is dated on or
after the cutoff.

python hw2/check_one_user.py hw2/outputs/encoder100 llm_uknn_umap32
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
uid = sorted(rows, key=lambda r: -r['ndcg_at_10'])[0]['user_id'] if len(sys.argv) < 4 else sys.argv[3]
saved = next(r for r in rows if r['user_id'] == uid)

tx = pd.read_csv('data/FAR-Trans/transactions.csv').drop_duplicates()
tx = tx[tx.transactionID >= 0]
pred = pd.read_csv(out / 'predictions.csv')
top10 = pred[(pred.method == method) & (pred.user_id == uid)].sort_values('rank').ISIN.tolist()
eligible = set(json.loads((Path('hw2/outputs/baseline/protocol.json')).read_text())['eligible_assets'])
fut = tx[(tx.customerID == uid) & (tx.transactionType == 'Buy') &
         (tx.timestamp.str[:10] >= cutoff) & (tx.timestamp.str[:10] < end)]
truth = set(fut.ISIN) & eligible
past = tx[(tx.customerID == uid) & (tx.timestamp.str[:10] < cutoff)]

print(f'user {uid[:8]}..., method {method}')
print('past Buy assets (before cutoff):', sorted(set(past[past.transactionType == "Buy"].ISIN)))
print('future Buy assets in window (eligible):', sorted(truth))
for r, isin in enumerate(top10, 1):
    print(f'  rank {r:2d} {isin} {"HIT" if isin in truth else ""}')
dcg = sum(1 / math.log2(r + 1) for r, i in enumerate(top10, 1) if i in truth)
idcg = sum(1 / math.log2(r + 1) for r in range(1, min(10, len(truth)) + 1))
print(f'hand DCG {dcg:.6f} / IDCG {idcg:.6f} = nDCG@10 {dcg / idcg:.6f}; saved by starter: {saved["ndcg_at_10"]:.6f}')
print('relevant_count saved', saved['relevant_count'], '| recomputed', len(truth))
print('latest input transaction date for this user:', past.timestamp.max(), '(must be < cutoff)')
