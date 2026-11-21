"""HW3 asset-side LLM Encoder: frozen Qwen3.5-0.8B PromptEOL embedding of each asset's
supplied attributes (name, category, subcategory, sector, industry).

Every distinct attribute record in asset_information.csv is embedded once (CPU). rerank3.py
picks, for each cut-off, the newest record per ISIN dated before that cut-off, so the text
used at a cut-off never contains later information. Output: hw3/cache/asset_emb.pkl
({text: float32 vector}).

python hw3/embed_assets.py --revision 2fc06364715b967f1860aea9cf38778875588b17
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hw3'))
from rerank3 import asset_text, CACHE  # noqa: E402

PROMPT = 'This financial asset:\n"{t}"\nmeans in one word:"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    ap.add_argument('--revision', default='2fc06364715b967f1860aea9cf38778875588b17')
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(args.threads)
    a = pd.read_csv(ROOT / 'data' / 'FAR-Trans' / 'asset_information.csv')
    texts = sorted({asset_text(r) for _, r in a.iterrows()})
    store = CACHE / 'asset_emb.pkl'
    emb = pickle.loads(store.read_bytes()) if store.exists() else {}
    todo = [t for t in texts if t not in emb]
    print(f'{len(texts)} distinct asset texts, {len(todo)} to encode', flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tok.padding_side = 'left'
    m = AutoModel.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float32).eval()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(todo), args.bs):
            batch = todo[i:i + args.bs]
            enc = tok([PROMPT.format(t=t) for t in batch], padding=True, truncation=True,
                      max_length=128, return_tensors='pt')
            h = m(**enc)
            h = h.last_hidden_state if hasattr(h, 'last_hidden_state') else h[0]
            for t, v in zip(batch, h[:, -1].float().numpy()):
                emb[t] = v.astype(np.float32)
            if (i // args.bs) % 10 == 0:
                print(f'  {i + len(batch)}/{len(todo)} [{time.perf_counter() - t0:.0f}s]', flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    store.write_bytes(pickle.dumps(emb))
    print(f'saved {len(emb)} vectors to {store} in {time.perf_counter() - t0:.0f}s')
    print('example:', PROMPT.format(t=texts[0]))


if __name__ == '__main__':
    main()
