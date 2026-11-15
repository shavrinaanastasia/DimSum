"""Reproducibility check: re-encode a few evaluated users' texts locally (CPU) with the
pinned Qwen3.5-0.8B snapshot and compare with the Colab GPU vectors used in the experiment.

python verify_embeddings.py --n 5
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from embed import Encoder  # noqa: E402  (same prompt/pooling code that produced the vectors)

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=5)
ap.add_argument('--cutoff', default='2021-06-01')
ap.add_argument('--protocol', default=str(ROOT / 'hw2/outputs/baseline/protocol.json'))
args = ap.parse_args()

users = json.load(open(args.protocol))['users'][:args.n]
with gzip.open(ROOT / 'outputs/emb/texts.json.gz', 'rt', encoding='utf-8') as f:
    T = json.load(f)
texts = [T['unique'][T['cutoffs'][args.cutoff][u]] for u in users]
emb_users = (ROOT / f'outputs/emb/qwen35_{args.cutoff}.users.txt').read_text(encoding='utf-8').split('\n')
E = np.load(ROOT / f'outputs/emb/qwen35_{args.cutoff}.npy')
stored = E[[emb_users.index(u) for u in users]]

os.environ.setdefault('HF_HUB_OFFLINE', '1')
enc = Encoder('Qwen/Qwen3.5-0.8B', 384, 8)
local = enc(texts, 1)
cos = (local * stored).sum(1) / (np.linalg.norm(local, axis=1) * np.linalg.norm(stored, axis=1))
for u, c, t in zip(users, cos, texts):
    print(f'user {u[:6]}...  tokens {len(enc.tok(enc.wrap(t)).input_ids):4d}  cosine(local CPU fp32, Colab T4 fp16+int8) = {c:.5f}')
# also check: are the stored vectors distinct from each other (not a constant output)?
S = stored / np.linalg.norm(stored, axis=1, keepdims=True)
print('mean pairwise cosine among these stored users:', round(float((S @ S.T)[np.triu_indices(len(S), 1)].mean()), 4))
print('first text:\n' + texts[0][:600])
