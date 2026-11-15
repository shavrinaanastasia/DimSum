"""Compare our paper-protocol runs with the public FAR-Trans benchmark on the same dates.

python src/paper_compare.py
"""
import glob
import json
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else 'v3'
ref = json.load(open('outputs/reference/far_trans_benchmark.json'))
rows = []
for f in sorted(glob.glob(f'outputs/runs/behaviour_none_*_new_{tag}.json')):
    r = json.load(open(f))
    date = r['args']['cutoff']
    if date not in ref['best4_per_date']:
        continue
    best_name, best = max(ref['best4_per_date'][date].items(), key=lambda kv: kv[1])
    res = {k: v['ndcg10'] for k, v in r['results'].items()}
    rows.append((date, r['n_warm'], ref['n_eval_users'][date], res['popularity_90d'], ref['popularity'][date],
                 best_name, best, res['reranker'], ref['mf_bpr_per_date'][date]))
print('best public = best of ~50 benchmark methods on that date (an oracle choice); MF-BPR = best single method')
print(f'{"date":11s} {"users ours/theirs":>18s} {"pop ours/theirs":>16s} {"best public":>24s} {"MF-BPR":>7s} '
      f'{"ours":>7s} {"vs best":>8s} {"vs BPR":>7s}')
for d, nu, nt, po, pt, bn, b, rr, bpr in rows:
    print(f'{d:11s} {nu:>8d}/{nt:<9d} {po:7.4f}/{pt:<8.4f} {bn:>17s} {b:.4f} {bpr:7.4f} {rr:7.4f} '
          f'{rr - b:+8.4f} {rr - bpr:+7.4f}')
if rows:
    n = len(rows)
    m = lambda i: sum(r[i] for r in rows) / n
    print(f'{"mean":11s} {"":18s} {"":16s} {"":>17s} {m(6):.4f} {m(8):7.4f} {m(7):7.4f} '
          f'{m(7) - m(6):+8.4f} {m(7) - m(8):+7.4f}')
