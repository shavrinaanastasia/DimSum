"""Validation-only round 2: combine the best single changes, score 3-seed averages
(the final pipeline averages 3 seeds), and compare 12 vs 24 training windows."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank3 as R  # noqa: E402
from tune import ndcg_users  # noqa: E402
from data import load  # noqa: E402

cutoff = '2021-06-01'
nwin = int(sys.argv[1]) if len(sys.argv) > 1 else 12
d = load(str(R.ROOT / 'data' / 'FAR-Trans'))
frames = [R.train_frame(d, c, e, R.inv_for(c, cutoff)) for c, e in R.windows(cutoff, nwin)]
va = frames[0].sort_values('user', kind='stable')
tr = pd.concat(frames[1:], ignore_index=True).sort_values('user', kind='stable')
gva, gtr = va.groupby('user', sort=False).size().values, tr.groupby('user', sort=False).size().values
base = dict(objective='lambdarank', metric='ndcg', eval_at=[10], learning_rate=0.03, num_leaves=31,
            min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambdarank_truncation_level=20, verbose=-1, seed=0, num_threads=8)
spec = 'base+rebuy+asset_tfidf+asset_llm+inv_llm'
feats = R.feats_for(spec, list(va.columns))
combos = [{}, {'feature_fraction': 0.5}, {'feature_fraction': 0.5, 'num_leaves': 15},
          {'feature_fraction': 0.5, 'num_leaves': 15, 'lambda_l2': 10.0}, {'feature_fraction': 0.3}]
if nwin != 12:
    combos = combos[:3]
for over in combos:
    p = {**base, **over}
    preds = []
    for s in range(3):
        m = lgb.train({**p, 'seed': s}, lgb.Dataset(tr[feats], tr.label, group=gtr), 4000,
                      valid_sets=[lgb.Dataset(va[feats], va.label, group=gva)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        preds.append(m.predict(va[feats], num_iteration=m.best_iteration))
    single = [ndcg_users(va, x) for x in preds]
    print(nwin, json.dumps(over), 'single', np.round(single, 4).tolist(),
          'avg3', round(ndcg_users(va, np.mean(preds, 0)), 5), flush=True)
