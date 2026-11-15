"""Frozen-LLM investor embeddings (prompted encoder, zero-shot, no weight updates).

For each split cut-off, every user with a pre-cut-off buy is serialised as text from
information dated before the cut-off only, then encoded:

  --model Qwen/Qwen3.5-0.8B          generative LLM; prompt-based last-token embedding
                                     (PromptEOL: '... means in one word:"')
  --model Qwen/Qwen3-Embedding-0.6B  dedicated embedding model; last-token pooling

Texts are deduplicated across cut-offs (inactive users' texts repeat), so each unique
text is encoded once. Output: outputs/emb/<tag>_<cutoff>.npy + .users.txt

python src/embed.py --model Qwen/Qwen3.5-0.8B --tag qwen35 --cutoffs 2022-06-01 2021-12-01 ...
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data import load, make_split  # noqa: E402


def profile_at(ci: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    c = ci[ci.ts < cutoff].sort_values('ts').drop_duplicates('customerID', keep='last')
    return c.set_index('customerID').to_dict('index')


def user_texts(sp, ci) -> dict[str, str]:
    t0 = sp.cutoff
    prof = profile_at(ci, t0)
    a = sp.assets
    past = sp.past.copy()
    past['age'] = (t0 - past.ts).dt.days
    b = past[past.transactionType == 'Buy'].assign(w=lambda x: 0.5 ** (x.age / 365))
    n_sell = past[past.transactionType == 'Sell'].groupby('customerID').size().to_dict()
    ub = b.groupby('customerID').agg(n=('age', 'size'), last=('age', 'min'))
    agg = (b.groupby(['customerID', 'ISIN']).agg(n=('age', 'size'), last=('age', 'min'), w=('w', 'sum'))
           .reset_index().sort_values(['customerID', 'w'], ascending=[True, False]))
    agg['cat'] = a.assetCategory.fillna('unknown').reindex(agg.ISIN).values
    n_assets = agg.groupby('customerID').size().to_dict()
    catmix = (b.assign(cat=a.assetCategory.fillna('unknown').reindex(b.ISIN).values)
              .groupby('customerID').cat.value_counts(normalize=True))
    desc = {i: ', '.join(str(v) for v in (x.assetCategory, x.assetSubCategory, x.sector, x.industry)
                         if isinstance(v, str)) for i, x in a.iterrows()}
    name = a.assetName.to_dict()
    top = {u: g.head(12)[['ISIN', 'n', 'last']].values.tolist() for u, g in agg.groupby('customerID', sort=False)}
    mix = {u: s for u, s in catmix.groupby(level=0)}
    out = {}
    for u, r in ub.iterrows():
        p = prof.get(u)
        if p:
            risk = p['riskLevel']
            src = 'estimated' if str(risk).startswith('Predicted_') else 'questionnaire'
            head = (f"Customer type {p['customerType']}; risk level {str(risk).replace('Predicted_', '')} ({src}); "
                    f"investment capacity {p['investmentCapacity']}.")
        else:
            head = 'Profile unknown.'
        lines = [head,
                 f"{int(r.n)} buys and {n_sell.get(u, 0)} sells of {n_assets[u]} assets; "
                 f"last buy {int(r['last'])} days ago.",
                 'Category mix: ' + ', '.join(f'{k[1]} {v:.0%}' for k, v in mix[u].items()) + '.',
                 'Assets bought (most weight first):']
        for i, n, last in top[u]:
            lines.append(f"- {name[i]} ({desc[i]}): {int(n)} buys, last {int(last)} days ago")
        out[u] = '\n'.join(lines)
    return out


class Encoder:
    def __init__(self, model: str, max_len: int, threads: int):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = 'left'
        dtype = torch.float16 if self.dev == 'cuda' else torch.float32
        self.m = AutoModel.from_pretrained(model, torch_dtype=dtype).to(self.dev).eval()
        self.max_len = max_len
        self.prompt_eol = 'Embedding' not in model

    def wrap(self, t: str) -> str:
        if self.prompt_eol:     # zero-shot prompt that makes the last token summarise the investor
            return f'This investor record:\n"{t}"\nmeans in one word:"'
        return f'Instruct: Represent this investor to find investors with similar future purchases\nQuery: {t}'

    def __call__(self, texts: list[str], bs: int) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), bs):
                enc = self.tok([self.wrap(t) for t in texts[i:i + bs]], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors='pt').to(self.dev)
                h = self.m(**enc, output_hidden_states=False)
                h = h.last_hidden_state if hasattr(h, 'last_hidden_state') else h[0]
                out.append(h[:, -1].float().cpu().numpy())   # left padding: last position is the last token
        return np.concatenate(out)


TEXTS = 'texts.json.gz'      # {"unique": {hash: text}, "cutoffs": {cutoff: {user: hash}}}


def export(args):
    """Local, CPU: build the leakage-free texts for every cut-off."""
    d = load(args.data)
    per_cut, uniq = {}, {}
    for c in args.cutoffs:
        sp = make_split(d, c, str((pd.Timestamp(c) + pd.DateOffset(months=1)).date()))
        tx = user_texts(sp, d['customers'])
        per_cut[c] = {}
        for u, t in tx.items():
            h = hashlib.md5(t.encode()).hexdigest()
            uniq.setdefault(h, t)
            per_cut[c][u] = h
        print(f'{c}: {len(tx)} users', flush=True)
    print(f'unique texts: {len(uniq)}', flush=True)
    with gzip.open(os.path.join(args.out, TEXTS), 'wt', encoding='utf-8') as f:
        json.dump({'unique': uniq, 'cutoffs': per_cut}, f)
    u, h = next(iter(per_cut[args.cutoffs[0]].items()))
    open(os.path.join(args.out, f'example_text_{args.cutoffs[0]}.txt'), 'w', encoding='utf-8').write(
        f'user {u}\n\n{uniq[h]}\n')


def encode(args):
    """GPU recommended (Colab/Kaggle): encode every unique text once; resumable."""
    with gzip.open(os.path.join(args.out, TEXTS), 'rt', encoding='utf-8') as f:
        uniq = json.load(f)['unique']
    cache_f = os.path.join(args.out, f'{args.tag}_cache.npz')
    cache = dict(np.load(cache_f)) if os.path.exists(cache_f) else {}
    todo = sorted((k for k in uniq if k not in cache), key=lambda k: len(uniq[k]))
    enc = Encoder(args.model, args.max_len, args.threads)
    print(f'device {enc.dev}; {len(todo)} texts to encode', flush=True)
    t = time.time()
    for i in range(0, len(todo), 2048):
        ks = todo[i:i + 2048]
        for k, v in zip(ks, enc([uniq[k] for k in ks], args.bs)):
            cache[k] = v.astype(np.float16)
        np.savez(cache_f, **cache)
        print(f'  encoded {i + len(ks)}/{len(todo)} [{time.time() - t:.0f}s]', flush=True)


def assemble(args):
    """Local: per-cut-off matrices aligned to user lists, read by rep.llm()."""
    with gzip.open(os.path.join(args.out, TEXTS), 'rt', encoding='utf-8') as f:
        per_cut = json.load(f)['cutoffs']
    q8 = os.path.join(args.out, f'{args.tag}_q8.npz')
    if os.path.exists(q8):
        # int8 copy made on the GPU machine to shrink the download (per-vector scale;
        # cosine to the fp16 original >= 0.9996 for every vector)
        z = np.load(q8)
        V = z['q'].astype(np.float32) * z['s']
        cache = {k: V[i] for i, k in enumerate(z['keys'].tolist())}
    else:
        cache = np.load(os.path.join(args.out, f'{args.tag}_cache.npz'))
    for c, m in per_cut.items():
        users = sorted(m)
        E = np.stack([cache[m[u]] for u in users]).astype(np.float32)
        np.save(os.path.join(args.out, f'{args.tag}_{c}.npy'), E)
        open(os.path.join(args.out, f'{args.tag}_{c}.users.txt'), 'w', encoding='utf-8').write('\n'.join(users))
        print(f'saved {args.tag}_{c}: {E.shape}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['export', 'encode', 'assemble'])
    ap.add_argument('--data', default='data/FAR-Trans')
    ap.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    ap.add_argument('--tag', default='qwen35')
    ap.add_argument('--cutoffs', nargs='+', default=['2022-06-01', '2021-12-01', '2021-09-01', '2021-06-01', '2021-03-01'])
    ap.add_argument('--out', default='outputs/emb')
    ap.add_argument('--max-len', type=int, default=384)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {'export': export, 'encode': encode, 'assemble': assemble}[args.stage](args)


if __name__ == '__main__':
    main()
