"""User representations and dimensionality reduction (none / PCA / UMAP)."""
from __future__ import annotations

import os

import numpy as np

from features import Signals


def behaviour(sg: Signals) -> np.ndarray:
    """Classical representation: time-decayed purchase vector over eligible items."""
    return sg.Xd365.copy()


def llm(sg: Signals, emb_dir: str, tag: str) -> np.ndarray:
    """Frozen-LLM embedding of each user's pre-cut-off text, produced by embed.py.
    Rows are aligned to sg.users via the saved user list."""
    stem = os.path.join(emb_dir, f'{tag}_{sg.sp.cutoff.date()}')
    E = np.load(stem + '.npy')
    users = open(stem + '.users.txt', encoding='utf-8').read().split()
    pos = {u: i for i, u in enumerate(users)}
    missing = [u for u in sg.users if u not in pos]
    assert not missing, f'{len(missing)} users missing from {stem}'
    return E[[pos[u] for u in sg.users]].astype(np.float32)


def reduce(R: np.ndarray, method: str, d: int, seed: int = 0) -> tuple[np.ndarray, str]:
    """Returns (reduced matrix, metric for kNN). Fitting uses only pre-cut-off users,
    so it is leakage-free."""
    if method == 'none' or d >= R.shape[1]:
        return R, 'cos'
    if method == 'pca':
        from sklearn.decomposition import PCA
        Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
        return PCA(n_components=d, random_state=seed).fit_transform(Rn).astype(np.float32), 'cos'
    if method == 'umap':
        import umap
        Z = umap.UMAP(n_components=d, metric='cosine', n_neighbors=30, min_dist=0.0,
                      random_state=seed, compatibility_layout=True).fit_transform(R)
        return Z.astype(np.float32), 'euclid'   # UMAP coordinates are compared by distance
    raise ValueError(method)
