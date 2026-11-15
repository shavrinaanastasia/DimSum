"""Print the HW2 results table for one encoder output directory and check the baselines
against the unchanged starter run.

python hw2/summarize.py hw2/outputs/encoder100 [hw2/outputs/baseline]
"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
m = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
if len(sys.argv) > 2:
    b = json.loads((Path(sys.argv[2]) / 'metrics.json').read_text(encoding='utf-8'))['methods']
    for k in ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell'):
        assert m[k] == b[k], k
    print('four baselines identical to the unchanged starter run: OK')
man = json.loads((out / 'run_manifest.json').read_text(encoding='utf-8'))
t = man['runtime']['timings_seconds']
base = ['popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell']
order = base + [k for k in m if k not in base and isinstance(m[k], dict) and 'ndcg_at_10' in m[k]]
print(f'{"method":22s} {"users":>6s} {"nonempty":>8s} {"all":>7s} {"nonempty":>8s} {"reduce s":>9s} {"kNN/fit s":>9s}')
for k in order:
    v = m[k]
    r = t.get(k + '_reduce_seconds')
    q = t.get(k + '_knn_seconds', t.get(k + '_train_predict_seconds'))
    print(f'{k:22s} {v["users"]:6d} {v["nonempty_truth_users"]:8d} {v["ndcg_at_10"]:7.4f} '
          f'{v["ndcg_at_10_nonempty"]:8.4f} {"" if r is None else f"{r:9.1f}"} {"" if q is None else f"{q:9.1f}"}')
print('failures:', len(man['failures']), '| total run seconds:', round(t['total_seconds'], 1))
