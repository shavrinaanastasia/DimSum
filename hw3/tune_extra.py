"""Validation-only check of the v2 'extra' feature group (same split as tune.py)."""
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank3 as R  # noqa: E402
from tune import ndcg_users  # noqa: E402
from data import load  # noqa: E402

cutoff = '2021-06-01'
d = load(str(R.ROOT / 'data' / 'FAR-Trans'))
frames = [R.train_frame(d, c, e, R.inv_for(c, cutoff)) for c, e in R.windows(cutoff, 12)]
va = frames[0].sort_values('user', kind='stable')
tr = pd.concat(frames[1:], ignore_index=True).sort_values('user', kind='stable')
gva, gtr = va.groupby('user', sort=False).size().values, tr.groupby('user', sort=False).size().values
base = dict(objective='lambdarank', metric='ndcg', eval_at=[10], learning_rate=0.03, num_leaves=15,
            min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambdarank_truncation_level=20, verbose=-1, seed=0, num_threads=4)
for spec in sys.argv[1:] or ['base+rebuy+asset_tfidf+asset_llm+inv_llm',
                              'base+rebuy+asset_tfidf+asset_llm+inv_llm+extra', 'base+rebuy+extra']:
    feats = R.feats_for(spec, list(va.columns))
    m = lgb.train(base, lgb.Dataset(tr[feats], tr.label, group=gtr), 4000,
                  valid_sets=[lgb.Dataset(va[feats], va.label, group=gva)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    print(spec, m.best_iteration, round(ndcg_users(va, m.predict(va[feats], num_iteration=m.best_iteration)), 5),
          flush=True)
