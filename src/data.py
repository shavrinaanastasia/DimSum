"""FAR-Trans loading, temporal split and evaluation.

Everything a model may see is computed from rows dated strictly before the
cut-off; the evaluation window [cutoff, end) is used only for ground truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def load(data_dir: str) -> dict[str, pd.DataFrame]:
    r = lambda n: pd.read_csv(os.path.join(data_dir, n))
    tx = r('transactions.csv')
    tx['ts'] = pd.to_datetime(tx.timestamp)
    ai = r('asset_information.csv')
    ai['ts'] = pd.to_datetime(ai.timestamp)
    ci = r('customer_information.csv')
    ci['ts'] = pd.to_datetime(ci.timestamp)
    cp = r('close_prices.csv')
    cp['ts'] = pd.to_datetime(cp.timestamp)
    return {'tx': tx, 'assets': ai, 'customers': ci, 'prices': cp}


@dataclass
class Split:
    """One temporal split. `items` is the eligible universe at the cut-off."""
    cutoff: pd.Timestamp
    end: pd.Timestamp
    items: list[str]
    item_ix: dict[str, int]
    past: pd.DataFrame            # all transactions (Buy and Sell) before cut-off, eligible items only
    truth: dict[str, set[int]]    # user -> eligible items bought in the window
    warm: list[str]               # evaluated users with >= 1 pre-cut-off buy
    cold: list[str]               # evaluated users without one (handled by teammate; popularity fallback here)
    assets: pd.DataFrame = field(repr=False)   # attributes known at cut-off, indexed by ISIN, in item order
    prices: pd.DataFrame = field(repr=False)   # close prices before cut-off


def make_split(d: dict, cutoff: str, end: str) -> Split:
    t0, te = pd.Timestamp(cutoff), pd.Timestamp(end)
    pr = d['prices'][d['prices'].ts < t0]
    a = d['assets'][d['assets'].ts < t0].sort_values('ts').drop_duplicates('ISIN', keep='last')
    items = sorted(set(pr.ISIN) & set(a.ISIN))           # priced and described before cut-off
    ix = {s: i for i, s in enumerate(items)}
    tx = d['tx'][d['tx'].ISIN.isin(ix)]
    past = tx[tx.ts < t0]
    fut = tx[(tx.ts >= t0) & (tx.ts < te) & (tx.transactionType == 'Buy')]
    truth = {u: {ix[s] for s in g.ISIN} for u, g in fut.groupby('customerID')}
    buyers = set(past[past.transactionType == 'Buy'].customerID)
    warm = sorted(u for u in truth if u in buyers)
    cold = sorted(u for u in truth if u not in buyers)
    return Split(t0, te, items, ix, past, truth, warm, cold,
                 a.set_index('ISIN').loc[items], pr)


_DISC = 1.0 / np.log2(np.arange(2, 12))


def ndcg10(ranked: list[int], rel: set[int]) -> float:
    dcg = sum(_DISC[i] for i, it in enumerate(ranked[:10]) if it in rel)
    return dcg / _DISC[:min(len(rel), 10)].sum()


def check_top10(ranked, n_items: int) -> None:
    assert len(ranked) == 10 and len(set(ranked)) == 10, 'need ten unique items'
    assert all(0 <= i < n_items for i in ranked), 'item outside eligible universe'


def evaluate(recs: dict[str, list[int]], sp: Split, users: list[str]) -> np.ndarray:
    out = []
    for u in users:
        check_top10(recs[u], len(sp.items))
        out.append(ndcg10(recs[u], sp.truth[u]))
    return np.array(out)


def bootstrap_ci(x: np.ndarray, n: int = 10_000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))
