"""Candidate coverage (recall of future purchases by the fixed candidate set) and investor-embedding
coverage, per dev set. Uses the starter truth restricted to eligible assets; same candidate code as rerank3.

python hw3/coverage.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank3 as R  # noqa: E402
from features import Signals, candidates  # noqa: E402

d = R.load_raw(str(R.ROOT / 'data' / 'FAR-Trans'))
rows = []
for c in ['2021-06-01', '2021-09-01', '2021-12-01', '2022-06-01']:
    light = pickle.loads((R.CACHE / f'bundle_{c}.pkl').read_bytes())
    elig = set(light['eligible'])
    users = [u for u in light['manifest']['users'] if light['truth'].get(u, set()) & elig]
    end = str((pd.Timestamp(c) + pd.Timedelta(days=30)).date())
    sp = R.make_split(d, c, end)
    sg = Signals(sp, customers=d['customers'])
    users = [u for u in users if u in sg.uix]
    cand = candidates(sg, users, sg.userknn(sg.Xd365, users, k=50), n=50)
    tot = hit = tot_new = hit_new = full_users = 0
    for u in users:
        cs = {sp.items[i] for i in cand[u]}
        past = {sp.items[i] for i in np.flatnonzero(sg.B[sg.uix[u]])}
        tr = light['truth'][u] & elig
        tot += len(tr); hit += len(tr & cs)
        new = tr - past
        tot_new += len(new); hit_new += len(new & cs)
        full_users += tr <= cs
    inv = R.inv_for(c, c)
    n_inv = 'n/a'
    if inv:
        ids = set((R.ROOT / 'outputs' / 'emb' / f'qwen35_{inv}.users.txt').read_text(encoding='utf-8').split())
        n_inv = f'{sum(u in ids for u in users) / len(users):.3f}'
    wins = R.windows(c, 12)
    rows.append({'cutoff': c, 'nonempty_users': len(users), 'truth_items': tot,
                 'recall_all': hit / tot, 'recall_new_items': hit_new / max(tot_new, 1), 'new_items': tot_new,
                 'users_all_truth_covered': full_users / len(users),
                 'train_windows_with_inv_emb': sum(R.inv_for(w, c) is not None for w, _ in wins),
                 'test_users_with_inv_emb': n_inv})
    print(rows[-1], flush=True)
out = pd.DataFrame(rows)
print(out.round(4).to_string(index=False))
out.to_csv(Path(__file__).resolve().parent / 'outputs' / 'coverage.csv', index=False)
