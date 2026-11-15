"""Measure encoder throughput on the real texts before launching the full run."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from data import load, make_split
from embed import user_texts, Encoder

model = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen3.5-0.8B'
d = load('data/FAR-Trans')
sp = make_split(d, '2022-06-01', '2022-07-01')
texts = list(user_texts(sp, d['customers']).values())[:64]
enc = Encoder(model, 384, 8)
n_tok = sum(len(enc.tok(enc.wrap(t)).input_ids) for t in texts) / len(texts)
t = time.time(); E = enc(texts, 16); dt = time.time() - t
print(f'{model}: dim {E.shape[1]}, mean tokens {n_tok:.0f}, {len(texts)/dt:.1f} texts/s')
