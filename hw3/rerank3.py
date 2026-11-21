"""HW3 main experiment (DimSum, A. Shavrina): multi-window LambdaRank reranker with
re-buy features and LLM asset/investor Encoders, on the provided starter protocol.

Protocol: users, eligible assets, truth and nDCG@10 come from the provided starter via
hw2/far-trans-llm/data_adapter.py (warm cohort, repeats allowed, 30-day window).

Leakage rules
  * Every feature at a cut-off c uses rows dated before c only (team loader src/data.py).
  * Training windows are monthly cut-offs c_k = c - k months with label window
    [c_k, min(c_k + 30 days, c)); so every training label is dated before the test cut-off.
  * The most recent window is the early-stopping validation set; the final model is refit
    on all windows with the chosen number of trees, averaged over --lgb-seeds seeds.
  * Asset embeddings use only asset attributes dated before each cut-off (newest record per
    ISIN before c), so a renamed or re-described asset never leaks future text.

Modes
  --fast  : rankings only for evaluated users whose starter truth is nonempty. Users with
            empty truth score 0 for every method, so the all-user mean is identical; this is
            a compute shortcut for development, never used for reported runtime.
  default : rankings for every evaluated user, written with the starter's save_results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'hw2' / 'far-trans-llm'))
from data import load as load_raw, make_split  # noqa: E402
from features import Signals, candidates, feature_frame  # noqa: E402

CACHE = ROOT / 'hw3' / 'cache'
FEAT_VERSION = 'v1'
DROP = ['user', 'item', 'label']
REBUY = ['gap_med', 'gap_cv', 'n_days', 'due', 'due_ratio', 'n30', 'n60', 'n180', 'last_year',
         'sold_after', 'rec_rank', 'freq_rank', 'u_rebuy_rate', 'u_gap_med', 'u_n90',
         'u_days_last_new', 'u_new_share180']
ASSET_LLM = ['a_llm_prof', 'a_llm_max', 'a_llm_prof_rank']
ASSET_TFIDF = ['a_tfidf_prof', 'a_tfidf_max', 'a_tfidf_prof_rank']
INV_LLM = ['inv_llm', 'inv_llm_rank']


# ----------------------------------------------------------------------------- re-buy
def rebuy_features(sp, sg):
    """Per (user, item) inter-purchase statistics + user-level re-buy behaviour."""
    t0 = sp.cutoff
    b = sp.past[sp.past.transactionType == 'Buy'][['customerID', 'ISIN', 'ts']].copy()
    b['u'] = b.customerID.map(sg.uix)
    b['i'] = b.ISIN.map(sp.item_ix)
    b = b.dropna(subset=['u', 'i'])
    b['u'] = b.u.astype(int)
    b['i'] = b.i.astype(int)
    b['age'] = (t0 - b.ts).dt.days.astype(float)
    b['day'] = b.ts.dt.normalize()
    b = b.sort_values(['u', 'ts'])
    # first purchase of each (u,i) marks a "new" buy event
    b['is_new'] = ~b.duplicated(['u', 'i'])
    days = b.drop_duplicates(['u', 'i', 'day'])
    g = days.groupby(['u', 'i']).age
    gaps = days.assign(gap=-g.diff()).groupby(['u', 'i']).gap
    pair = pd.DataFrame({
        'gap_med': gaps.median(), 'gap_mean': gaps.mean(), 'gap_std': gaps.std(),
        'n_days': g.size(), 'last': g.min()})
    pair['gap_cv'] = pair.gap_std / (pair.gap_mean + 1e-9)
    pair['due'] = pair.gap_med - pair['last']
    pair['due_ratio'] = pair['last'] / (pair.gap_med + 1e-9)
    flags = pd.DataFrame({'u': b.u, 'i': b.i, 'n30': b.age < 30, 'n60': b.age < 60, 'n180': b.age < 180,
                          'last_year': (b.age >= 335) & (b.age < 395)}).groupby(['u', 'i']).sum()
    pair = pair.join(flags.astype(float))
    s = sp.past[sp.past.transactionType == 'Sell'][['customerID', 'ISIN', 'ts']]
    s = s.assign(u=s.customerID.map(sg.uix), i=s.ISIN.map(sp.item_ix)).dropna(subset=['u', 'i'])
    s_last = s.assign(age=(t0 - s.ts).dt.days).groupby([s.u.astype(int), s.i.astype(int)]).age.min()
    pair['sold_after'] = (s_last.reindex(pair.index) < pair['last']).astype(float).fillna(0).values
    pair['rec_rank'] = pair.groupby(level=0)['last'].rank(method='min')
    pair['freq_rank'] = pair.groupby(level=0)['n_days'].rank(method='min', ascending=False)
    # user level
    ub = b.groupby('u')
    user = pd.DataFrame({
        'u_rebuy_rate': 1 - ub.is_new.mean(),
        'u_n90': (b.age < 90).groupby(b.u).sum().astype(float),
        'u_days_last_new': b[b.is_new].groupby('u').age.min(),
    })
    ud = b.drop_duplicates(['u', 'day']).groupby('u').age
    user['u_gap_med'] = (-ud.diff()).groupby(b.drop_duplicates(['u', 'day']).u).median()
    r180 = b[b.age < 180]
    user['u_new_share180'] = r180.groupby('u').is_new.mean()
    return pair.drop(columns=['gap_mean', 'gap_std', 'last']), user


def add_rebuy(df, sg, pair, user):
    ui = df.user.map(sg.uix).values
    key = pd.MultiIndex.from_arrays([ui, df.item.values])
    P = pair.reindex(key)
    for c in pair.columns:
        df[c] = P[c].values.astype(np.float32)
    U = user.reindex(ui)
    for c in user.columns:
        df[c] = U[c].values.astype(np.float32)


EXTRA = ['pop7', 'pop14', 'item_age', 'i_rebuy30', 'i_multi_share', 'ui_months6', 'u_months6',
         'i_rebuy30_x_seen']


def extra_features(tx, cutoff):
    """v2 features from buys before the cut-off: item-level re-buy propensity (savings-plan-like
    assets), short-term popularity, item age, and monthly regularity of (user, item) and user."""
    t0 = pd.Timestamp(cutoff)
    b = tx[(tx.ts < t0) & (tx.transactionType == 'Buy')][['customerID', 'ISIN', 'ts']].copy()
    b['age'] = (t0 - b.ts).dt.days
    item = pd.DataFrame({
        'pop7': b[b.age < 7].groupby('ISIN').customerID.nunique(),
        'pop14': b[b.age < 14].groupby('ISIN').customerID.nunique(),
        'item_age': b.groupby('ISIN').age.max(),
    })
    # P(same user buys the same asset again within 30 days | a buy), from buys >= 30 days old
    s = b.sort_values(['customerID', 'ISIN', 'ts'])
    nxt = s.groupby(['customerID', 'ISIN']).ts.shift(-1)
    s['re30'] = ((nxt - s.ts).dt.days <= 30).astype(float)
    old = s[s.age >= 30]
    item['i_rebuy30'] = old.groupby('ISIN').re30.mean()
    cnt = b.groupby(['ISIN', 'customerID']).size()
    item['i_multi_share'] = (cnt >= 2).groupby(level=0).mean()
    item[['pop7', 'pop14']] = item[['pop7', 'pop14']].fillna(0)
    r6 = b[b.age < 182].assign(m=lambda x: x.ts.dt.to_period('M'))
    pair = r6.groupby(['customerID', 'ISIN']).m.nunique().rename('ui_months6').to_frame()
    user = r6.groupby('customerID').m.nunique().rename('u_months6').to_frame()
    return item, pair, user


def add_extra(df, items, item, pair, user):
    isin = np.asarray(items)[df.item.values]
    I = item.reindex(isin)
    for c in item.columns:
        df[c] = I[c].values.astype(np.float32)
    P = pair.reindex(pd.MultiIndex.from_arrays([df.user.values, isin]))
    df['ui_months6'] = P.ui_months6.fillna(0).values.astype(np.float32)
    df['u_months6'] = user.reindex(df.user.values).u_months6.fillna(0).values.astype(np.float32)
    df['i_rebuy30_x_seen'] = (df.i_rebuy30.fillna(0) * (df.n_buy > 0)).astype(np.float32)


# ----------------------------------------------------------------------------- assets
def asset_text(row):
    parts = [f'Asset: {row.assetName}', f'category: {row.assetCategory}',
             f'subcategory: {row.assetSubCategory}']
    for c in ('sector', 'industry'):
        v = row[c]
        if isinstance(v, str) and v.strip():
            parts.append(f'{c}: {v}')
    return '; '.join(parts)


def asset_vectors(sp, kind):
    """Rows aligned with sp.items. kind: 'llm' (cached Qwen3.5 PromptEOL) or 'tfidf'."""
    texts = [asset_text(r) for _, r in sp.assets.iterrows()]
    if kind == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        V = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        M = V.fit_transform(texts).astype(np.float32).toarray()
    else:
        store = CACHE / 'asset_emb.pkl'
        emb = pickle.loads(store.read_bytes()) if store.exists() else {}
        missing = [t for t in texts if t not in emb]
        if missing:
            raise RuntimeError(f'{len(missing)} asset texts not embedded; run hw3/embed_assets.py')
        M = np.stack([emb[t] for t in texts]).astype(np.float32)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def add_asset_sim(df, sg, M, prefix):
    prof_col = np.empty(len(df), np.float32)
    max_col = np.empty(len(df), np.float32)
    rank_col = np.empty(len(df), np.float32)
    S = M @ M.T
    for u, idx in df.groupby('user', sort=False).indices.items():
        r = sg.uix[u]
        w = sg.Xd365[r]
        prof = w @ M
        prof /= np.linalg.norm(prof) + 1e-9
        full = M @ prof
        items = df.item.values[idx]
        past = np.flatnonzero(sg.B[r])
        prof_col[idx] = full[items]
        Sx = S[np.ix_(items, past)]
        Sx[items[:, None] == past[None, :]] = -np.inf  # exclude self-similarity
        mx = Sx.max(1) if len(past) else np.full(len(items), -np.inf)
        max_col[idx] = np.where(np.isfinite(mx), mx, 0.0)
        rank_col[idx] = (-full).argsort().argsort()[items]
    df[f'{prefix}_prof'] = prof_col
    df[f'{prefix}_max'] = max_col
    df[f'{prefix}_prof_rank'] = rank_col


# ----------------------------------------------------------------------------- investor LLM
def add_investor_llm(df, sg, cutoff, k=50):
    """HW2 Encoder: UMAP-32 Qwen3.5 investor embedding -> user-kNN score (only where the
    embedding file for this cut-off exists)."""
    from reranker_encoder import load_emb, reduced, llm_knn, add_llm
    emb_dir = ROOT / 'outputs' / 'emb'
    if not (emb_dir / f'qwen35_{cutoff}.npy').exists():
        df['inv_llm'] = np.nan
        df['inv_llm_rank'] = np.nan
        return False
    E, U = load_emb(emb_dir, cutoff)
    cache = ROOT / 'hw2' / 'outputs' / 'cache'
    Z, metric = reduced(E, f'qwen35_{cutoff}', 'umap', 32, 42, cache)
    add_llm(df, llm_knn(sg, Z, metric, U, sorted(set(df.user)), k), 'inv_llm')
    return True


# ----------------------------------------------------------------------------- frames
def build_frame(d, cutoff, end, users_fn, inv_emb_cutoff=None):
    sp = make_split(d, cutoff, end)
    sg = Signals(sp, customers=d['customers'])
    users = users_fn(sp, sg)
    bknn = sg.userknn(sg.Xd365, users, k=50)
    df = feature_frame(sg, users, candidates(sg, users, bknn, n=50), {'uknn': bknn})
    pair, user = rebuy_features(sp, sg)
    add_rebuy(df, sg, pair, user)
    add_asset_sim(df, sg, asset_vectors(sp, 'tfidf'), 'a_tfidf')
    try:
        add_asset_sim(df, sg, asset_vectors(sp, 'llm'), 'a_llm')
    except RuntimeError as e:
        print('asset LLM:', e)
        for c in ASSET_LLM:
            df[c] = np.nan
    if inv_emb_cutoff:
        add_investor_llm(df, sg, inv_emb_cutoff)
    else:
        df['inv_llm'] = np.nan
        df['inv_llm_rank'] = np.nan
    df['label'] = [int(i in sp.truth.get(u, ())) for u, i in zip(df.user, df.item)]
    add_extra(df, sp.items, *extra_features(d['tx'], cutoff))
    return sp, sg, df


def train_frame(d, c, end, inv_cut):
    f = CACHE / 'frames' / f'{FEAT_VERSION}_{c}_{end}_{inv_cut or "noinv"}.pkl'
    if f.exists():
        df = pd.read_pickle(f)
        if 'pop7' not in df:   # v1 cache: add the v2 columns (items list from the same split)
            add_extra(df, make_split(d, c, end).items, *extra_features(d['tx'], c))
            df.to_pickle(f)
        return df
    t = time.perf_counter()
    _, _, df = build_frame(d, c, end, lambda sp, sg: sp.warm, inv_cut)
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(f)
    print(f'  train window {c}..{end}: users {df.user.nunique()}, rows {len(df)}, '
          f'pos {df.label.sum()} [{time.perf_counter() - t:.0f}s]', flush=True)
    return df


EMB_CUTS = sorted(p.stem.replace('qwen35_', '') for p in (ROOT / 'outputs' / 'emb').glob('qwen35_*.npy'))


def inv_for(c, test_cutoff):
    """Newest investor-embedding cut-off <= c (texts built from rows before it), or None.
    None everywhere when the test cut-off itself has no embedding file."""
    if test_cutoff not in EMB_CUTS:
        return None
    ok = [e for e in EMB_CUTS if e <= c]
    return ok[-1] if ok else None


def windows(cutoff, n):
    c0 = pd.Timestamp(cutoff)
    out = []
    for k in range(1, n + 1):
        c = c0 - pd.DateOffset(months=k)
        e = min(c + pd.Timedelta(days=30), c0)
        out.append((str(c.date()), str(e.date())))
    return out


# ----------------------------------------------------------------------------- starter bundle
def starter_bundle(cutoff, data_dir, fast, max_users=10 ** 7):
    """Load the starter protocol once per cut-off and cache what models may not see apart."""
    from data_adapter import load
    f = CACHE / (f'bundle_{cutoff}.pkl' if max_users >= 10 ** 7 else f'bundle_{cutoff}_n{max_users}.pkl')
    ns = argparse.Namespace(starter_dir=ROOT / 'hw2' / 'far-trans', data_dir=Path(data_dir),
                            cutoff=cutoff, horizon_days=30, max_users=max_users, candidate_count=50)
    if f.exists() and fast:
        return None, pickle.loads(f.read_bytes()), ns
    bundle = load(ns)
    starter, assets, history, truth, contexts, baseline, events, pop, manifest = bundle
    elig = sorted(next(iter(contexts.values()))['eligible_asset_ids'])
    assert all(sorted(c['eligible_asset_ids']) == elig for c in contexts.values())
    light = dict(truth=truth, eligible=elig, baseline=baseline, pop=pop, manifest=manifest)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(light))
    return bundle, light, ns


# ----------------------------------------------------------------------------- main
CONFIGS = {
    'behaviour_hw2': 'base',                         # HW2 reranker features, multi-window
    'rebuy': 'base+rebuy',
    'rebuy_tfidf': 'base+rebuy+asset_tfidf',         # matched text control for the asset Encoder
    'rebuy_assetllm': 'base+rebuy+asset_llm',        # asset Encoder (LLM)
    'full': 'base+rebuy+asset_llm+inv_llm',          # both Encoders
    'main': 'base+rebuy+asset_tfidf+asset_llm+inv_llm',  # HW3 main model (chosen on validation)
    'rebuy_invllm': 'base+rebuy+inv_llm',            # investor Encoder only
}


def feats_for(spec, cols):
    base = [c for c in cols if c not in DROP + REBUY + ASSET_LLM + ASSET_TFIDF + INV_LLM + EXTRA]
    groups = {'base': base, 'rebuy': REBUY, 'asset_llm': ASSET_LLM, 'asset_tfidf': ASSET_TFIDF,
              'inv_llm': INV_LLM, 'extra': EXTRA}
    out = []
    for g in spec.split('+'):
        out += groups[g]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', default=str(ROOT / 'data' / 'FAR-Trans'))
    ap.add_argument('--cutoff', default='2021-06-01')
    ap.add_argument('--windows', type=int, default=12)
    ap.add_argument('--configs', default='all')
    ap.add_argument('--lgb-seeds', type=int, default=3)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--max-users', type=int, default=10 ** 7, help='100 = starter teaching sample')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--params', default='{}', help='JSON overrides for LightGBM params')
    args = ap.parse_args()
    t_all = time.perf_counter()
    timings = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle, light, ns = starter_bundle(args.cutoff, args.data_dir, args.fast, args.max_users)
    truth, elig, pop, manifest = light['truth'], light['eligible'], light['pop'], light['manifest']
    eval_users = manifest['users']
    eligset = set(elig)
    d = load_raw(args.data_dir)

    inv_test = inv_for(args.cutoff, args.cutoff)
    t = time.perf_counter()
    wins = windows(args.cutoff, args.windows)
    built = [train_frame(d, c, e, inv_for(c, args.cutoff)) for c, e in wins]
    timings['train_features_seconds'] = time.perf_counter() - t

    t = time.perf_counter()
    end = str((pd.Timestamp(args.cutoff) + pd.Timedelta(days=30)).date())
    rel = {u: truth.get(u, set()) & eligset for u in eval_users}
    if args.fast:
        users_fn = lambda sp, sg: [u for u in eval_users if rel[u] and u in sg.uix]
    else:
        users_fn = lambda sp, sg: [u for u in eval_users if u in sg.uix]
    sp, sg, test = build_frame(d, args.cutoff, end, users_fn, inv_test)
    timings['test_features_seconds'] = time.perf_counter() - t
    print(f'test {args.cutoff}: users {test.user.nunique()}, rows {len(test)}', flush=True)

    import lightgbm as lgb
    params = dict(objective='lambdarank', metric='ndcg', eval_at=[10], learning_rate=0.03,
                  num_leaves=31, min_data_in_leaf=50, feature_fraction=0.5, bagging_fraction=0.8,
                  bagging_freq=1, lambdarank_truncation_level=20, verbose=-1, seed=0)
    # feature_fraction 0.5 chosen on the 2021-05 validation window only (hw3/tune*.py)
    params.update(json.loads(args.params))

    def ds(df):
        df = df.sort_values('user', kind='stable')
        return df, df.groupby('user', sort=False).size().values

    va, gva = ds(built[0])
    trn, gtr = ds(pd.concat(built[1:], ignore_index=True))
    full, gfull = ds(pd.concat(built, ignore_index=True))

    configs = CONFIGS if args.configs == 'all' else {k: CONFIGS[k] for k in args.configs.split(',')}
    if inv_test is None:
        configs = {k: v for k, v in configs.items() if 'inv_llm' not in v}
    pop_order = sorted(pop, key=lambda i: (-pop[i], i))
    predictions = defaultdict(dict)
    info = {}
    for name, spec in configs.items():
        feats = feats_for(spec, test.columns)
        t = time.perf_counter()
        m = lgb.train(params, lgb.Dataset(trn[feats], trn.label, group=gtr), 3000,
                      valid_sets=[lgb.Dataset(va[feats], va.label, group=gva)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        best = max(m.best_iteration, 1)
        t_fit = time.perf_counter()
        models = [lgb.train({**params, 'seed': s}, lgb.Dataset(full[feats], full.label, group=gfull), best)
                  for s in range(args.lgb_seeds)]
        timings[name + '_train_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        score = sum(ms.predict(test[feats]) for ms in models) / len(models)
        imp = sum(pd.Series(ms.feature_importance('gain'), feats) for ms in models)
        tt = test.assign(score=score)
        for u, g in tt.groupby('user', sort=False):
            order = [sp.items[i] for i in g.item.values[np.argsort(-g.score.values, kind='stable')]]
            top = [i for i in order if i in eligset][:10]
            for i in pop_order:
                if len(top) == 10:
                    break
                if i in eligset and i not in top:
                    top.append(i)
            predictions[u][name] = top
        for u in eval_users:
            if name not in predictions[u] and (not args.fast or rel[u]):
                predictions[u][name] = [i for i in pop_order if i in eligset][:10]
        timings[name + '_predict_seconds'] = time.perf_counter() - t
        share = lambda cols: float(imp[[c for c in feats if c in cols]].sum() / imp.sum())
        info[name] = {'best_iteration': best, 'valid_ndcg10_candidates': m.best_score['valid_0']['ndcg@10'],
                      'gain_share': {'rebuy': share(REBUY), 'asset_llm': share(ASSET_LLM),
                                     'asset_tfidf': share(ASSET_TFIDF), 'inv_llm': share(INV_LLM)},
                      'top_features': imp.sort_values(ascending=False).head(12).round(0).to_dict()}
        print(f'{name}: it {best}, valid {info[name]["valid_ndcg10_candidates"]:.4f}, '
              f'shares {info[name]["gain_share"]}', flush=True)

    timings['total_seconds'] = time.perf_counter() - t_all
    meta = {'cutoff': args.cutoff, 'windows': wins, 'params': params, 'lgb_seeds': args.lgb_seeds,
            'configs': configs, 'variants': info, 'timings_seconds': timings, 'fast': args.fast,
            'feature_version': FEAT_VERSION, 'investor_embedding_test': inv_test}
    if args.fast:
        import importlib.util
        spec = importlib.util.spec_from_file_location('st', ROOT / 'hw2' / 'far-trans' / 'starter.py')
        st = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(st)
        rows = []
        for u in eval_users:
            meths = dict(light['baseline'][u])
            meths.update(predictions.get(u, {}))
            for nm in configs:  # empty-truth users were skipped (score 0 for any ranking)
                meths.setdefault(nm, None)
            for nm, rk in meths.items():
                rows.append((u, nm, len(rel[u]), st.ndcg10(rk, rel[u]) if rel[u] else 0.0))
        per = pd.DataFrame(rows, columns=['user_id', 'method', 'relevant_count', 'ndcg_at_10'])
        per.to_csv(args.output_dir / 'per_user.csv', index=False)
        summ = per.groupby('method').apply(lambda g: pd.Series({
            'users': len(g), 'nonempty': int((g.relevant_count > 0).sum()),
            'all': g.ndcg_at_10.mean(), 'nonempty_mean': g[g.relevant_count > 0].ndcg_at_10.mean()}))
        print(summ.sort_values('all', ascending=False).round(4).to_string())
        meta['summary'] = summ.round(6).to_dict()
        (args.output_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    else:
        from data_adapter import save_results
        ns.output_dir = args.output_dir
        save_results(ns, bundle, predictions, [], 'hw3_rerank3', meta)


if __name__ == '__main__':
    main()
