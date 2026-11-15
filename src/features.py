"""Signals, candidate generation and (user, item) features for one split.

All quantities use only `Split.past` (before the cut-off).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import Split

ATTRS = ['assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID']


def _l2(m):
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


RISK = {'Conservative': 0, 'Income': 1, 'Balanced': 2, 'Aggressive': 3}
CAP = {'CAP_LT30K': 0, 'CAP_30K_80K': 1, 'CAP_80K_300K': 2, 'CAP_GT300K': 3}


def ials(B: np.ndarray, conf: np.ndarray, k: int = 64, reg: float = 100.0, iters: int = 12, seed: int = 0):
    # reg/alpha/k tuned under the paper protocol on non-test dates (scripts/tune_ials.py):
    # reg=0.1 overfits (nDCG@10 ~0.04 on new assets); reg=100, alpha=10, decayed confidence is best
    """Implicit ALS (Hu, Koren, Volinsky 2008) in numpy. B: binary preferences (U x N),
    conf: confidence weights c_ui - 1 on the non-zeros. Returns user and item factors."""
    rng = np.random.default_rng(seed)
    U, N = B.shape
    P, Q = rng.normal(0, 0.01, (U, k)), rng.normal(0, 0.01, (N, k))
    nz_u = [np.flatnonzero(B[u]) for u in range(U)]
    nz_i = [np.flatnonzero(B[:, i]) for i in range(N)]
    I = reg * np.eye(k)

    def solve(F, nz, c_rows):
        G = F.T @ F + I
        out = np.zeros((len(nz), k))
        for r, idx in enumerate(nz):
            if len(idx) == 0:
                continue
            c = c_rows(r, idx)
            Fi = F[idx]
            out[r] = np.linalg.solve(G + (Fi.T * c) @ Fi, Fi.T @ (1 + c))
        return out

    for _ in range(iters):
        P = solve(Q, nz_u, lambda r, idx: conf[r, idx])
        Q = solve(P, nz_i, lambda r, idx: conf[idx, r])
    return P.astype(np.float32), Q.astype(np.float32)


def _ord(v, m):
    return m.get(str(v).replace('Predicted_', ''), -1)


class Signals:
    def __init__(self, sp: Split, ease_lambda: float = 200.0, customers: pd.DataFrame | None = None,
                 mf: bool = True):
        self.sp = sp
        t0, N = sp.cutoff, len(sp.items)
        buys = sp.past[sp.past.transactionType == 'Buy']
        sells = sp.past[sp.past.transactionType == 'Sell']
        self.users = sorted(buys.customerID.unique())
        self.uix = {u: i for i, u in enumerate(self.users)}
        U = len(self.users)
        ui = buys.customerID.map(self.uix).values
        ii = buys.ISIN.map(sp.item_ix).values
        age = (t0 - buys.ts).dt.days.values.astype(np.float32)

        def acc(w):
            m = np.zeros((U, N), np.float32); np.add.at(m, (ui, ii), w); return m

        self.X = acc(1.0)                          # buy counts
        self.Xd90 = acc(0.5 ** (age / 90))         # time-decayed counts
        self.Xd365 = acc(0.5 ** (age / 365))
        self.Xd30 = acc(0.5 ** (age / 30))
        # net units still held = bought - sold (FAR-Trans has no opening positions; approximate)
        self.units = acc(buys.units.values.astype(np.float32))
        self.val = acc(buys.totalValue.values.astype(np.float32))
        self.last = np.full((U, N), 9999.0, np.float32)
        np.minimum.at(self.last, (ui, ii), age)
        self.first = np.zeros((U, N), np.float32)
        np.maximum.at(self.first, (ui, ii), age)
        su = sells.customerID.map(self.uix); ok = su.notna().values
        self.S = np.zeros((U, N), np.float32)
        np.add.at(self.S, (su.values[ok].astype(int), sells.ISIN.map(sp.item_ix).values[ok]), 1.0)
        np.subtract.at(self.units, (su.values[ok].astype(int), sells.ISIN.map(sp.item_ix).values[ok]),
                       sells.units.values[ok].astype(np.float32))
        self.held = (self.units > 1e-6).astype(np.float32)
        self.B = (self.X > 0).astype(np.float32)
        self.u_last = np.where(self.X > 0, self.last, 9999).min(1)
        self.u_n = self.X.sum(1)

        # item popularity (distinct buyers) over several windows
        def pop(days):
            b = buys[buys.ts >= t0 - pd.Timedelta(days=days)]
            p = np.zeros(N, np.float32)
            np.add.at(p, b.ISIN.map(sp.item_ix).values, 1.0)
            return p
        self.pop = {d: pop(d) for d in (30, 90, 180, 100000)}
        self.trend = (self.pop[30] + 1) / (self.pop[180] / 6 + 1)

        # price features from close prices before cut-off
        px = sp.prices.pivot_table(index='ts', columns='ISIN', values='closePrice').reindex(columns=sp.items).ffill()
        lastp = px.iloc[-1].values
        def ret(days):
            ref = px[px.index <= t0 - pd.Timedelta(days=days)]
            base = ref.iloc[-1].values if len(ref) else np.full(N, np.nan)
            return np.nan_to_num(lastp / base - 1, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.ret = {d: ret(d) for d in (30, 90, 365)}
        daily = px[px.index > t0 - pd.Timedelta(days=90)].pct_change()
        self.vol = np.nan_to_num(daily.std().values, nan=0.0).astype(np.float32)

        # content: one-hot attributes (classical CBF of the course slides)
        A = sp.assets[ATTRS].fillna('unknown').astype(str)
        self.F = _l2(pd.get_dummies(A).values.astype(np.float32))
        self.cat_codes = {c: pd.factorize(A[c].values)[0].astype(int) for c in ATTRS[:3]}

        # collaborative: item-kNN cosine and EASE
        n = np.sqrt(self.B.sum(0)) + 1e-9
        self.S_item = (self.B.T @ self.B) / n[:, None] / n[None, :]
        np.fill_diagonal(self.S_item, 0)
        G = self.B.T @ self.B + ease_lambda * np.eye(N, dtype=np.float32)
        P = np.linalg.inv(G)
        self.W_ease = (-P / np.diag(P)[None, :]).astype(np.float32)
        np.fill_diagonal(self.W_ease, 0)

        # collaborative: implicit ALS on decayed purchase confidence
        self.mf = None
        if mf:
            Pf, Qf = ials(self.B, 10 * np.log1p(self.Xd365 * 4))
            self.mf = Pf @ Qf.T

        # investor profile at the cut-off and risk fit (suitability-style features)
        U = len(self.users)
        self.u_risk = np.full(U, -1, np.int8)
        self.u_cap = np.full(U, -1, np.int8)
        self.u_type = np.full(U, -1, np.int8)
        self.u_risk_pred = np.zeros(U, np.int8)
        if customers is not None:
            c = customers[customers.ts < t0].sort_values('ts').drop_duplicates('customerID', keep='last')
            c = c.set_index('customerID').reindex(self.users)
            self.u_risk = np.array([_ord(v, RISK) for v in c.riskLevel], np.int8)
            self.u_cap = np.array([_ord(v, CAP) for v in c.investmentCapacity], np.int8)
            self.u_type = pd.Categorical(
                c.customerType.fillna('unknown'),
                categories=['Mass', 'Premium', 'Professional', 'Inactive', 'Legal Entity', 'unknown']).codes.astype(np.int8)
            self.u_risk_pred = c.riskLevel.astype(str).str.startswith('Predicted_').values.astype(np.int8)
        # mean risk level of each item's past buyers, and popularity within each risk segment (180 days)
        known = self.u_risk >= 0
        w = self.B[known]
        self.i_risk = (self.u_risk[known] @ w) / (w.sum(0) + 1e-9)
        recent = (self.last < 180).astype(np.float32)
        self.pop_seg = {}
        for lvl in range(-1, 4):
            m = self.u_risk == lvl
            p = recent[m].sum(0)
            self.pop_seg[lvl] = p / (p.sum() + 1e-9)

    # ---- representation-based user-kNN ---------------------------------
    def userknn(self, R: np.ndarray, users: list[str], k: int = 50, metric: str = 'cos') -> dict[str, np.ndarray]:
        """Score items by the purchases of the k nearest users in representation R
        (rows aligned with self.users). Neighbours' purchases are weighted by similarity."""
        out = {}
        Rn = _l2(R) if metric == 'cos' else R
        for u in users:
            q = Rn[self.uix[u]]
            sims = Rn @ q if metric == 'cos' else 1.0 / (1.0 + np.linalg.norm(Rn - q, axis=1))
            sims[self.uix[u]] = -np.inf
            nn = np.argpartition(-sims, k)[:k]
            out[u] = sims[nn] @ self.Xd365[nn]
        return out


def candidates(sg: Signals, users, knn: dict, n=50, new_only=False) -> dict[str, np.ndarray]:
    """Union of top-n per source: own past buys, user-kNN, item-kNN, EASE, iALS, 90-day popularity.
    new_only: drop assets the user bought before (paper protocol)."""
    top = lambda s, k: np.argpartition(-s, min(k, len(s) - 1))[:k]
    p90 = sg.pop[90]
    out = {}
    for u in users:
        r = sg.uix[u]
        b = sg.B[r]
        c = set() if new_only else set(np.flatnonzero(b))
        srcs = [knn[u], b @ sg.S_item, b @ sg.W_ease, p90]
        if sg.mf is not None:
            srcs.append(sg.mf[r])
        for s in srcs:
            s = np.where(b > 0, -np.inf, s) if new_only else s
            c |= set(top(s, n))
        if new_only:
            c -= set(np.flatnonzero(b))
        out[u] = np.array(sorted(c))
    return out


def feature_frame(sg: Signals, users, cands, knn_feats: dict[str, dict]) -> pd.DataFrame:
    """One row per (user, candidate). knn_feats: name -> {user: score vector}."""
    rows = []
    for u in users:
        r, c = sg.uix[u], cands[u]
        x, b = sg.X[r], sg.B[r]
        prof = x @ sg.F
        f = {
            'user': u, 'item': c,
            'n_buy': x[c], 'dec90': sg.Xd90[r, c], 'dec365': sg.Xd365[r, c],
            'days_last': sg.last[r, c], 'days_first': sg.first[r, c],
            'n_sell': sg.S[r, c], 'net_buys': x[c] - sg.S[r, c],
            'dec30': sg.Xd30[r, c], 'held': sg.held[r, c], 'u_n_held': np.full(len(c), sg.held[r].sum()),
            'val_share': sg.val[r, c] / (sg.val[r].sum() + 1e-9),
            'u_n': np.full(len(c), sg.u_n[r]), 'u_distinct': np.full(len(c), b.sum()),
            'u_days_last': np.full(len(c), sg.u_last[r]),
            'pop30': sg.pop[30][c], 'pop90': sg.pop[90][c], 'pop180': sg.pop[180][c],
            'pop_all': sg.pop[100000][c], 'trend': sg.trend[c],
            'ret30': sg.ret[30][c], 'ret90': sg.ret[90][c], 'ret365': sg.ret[365][c], 'vol90': sg.vol[c],
            'cbf': sg.F[c] @ prof / (np.linalg.norm(prof) + 1e-9),
            'itemknn': (b @ sg.S_item)[c], 'ease': (b @ sg.W_ease)[c],
            'u_risk': np.full(len(c), sg.u_risk[r]), 'u_cap': np.full(len(c), sg.u_cap[r]),
            'u_type': np.full(len(c), sg.u_type[r]), 'u_risk_pred': np.full(len(c), sg.u_risk_pred[r]),
            'i_risk': sg.i_risk[c], 'risk_gap': sg.i_risk[c] - sg.u_risk[r],
            'pop_seg': sg.pop_seg[int(sg.u_risk[r])][c],
        }
        for nm, s in (('itemknn', b @ sg.S_item), ('ease', b @ sg.W_ease)) + (
                (('mf', sg.mf[r]),) if sg.mf is not None else ()):
            if nm == 'mf':
                f['mf'] = s[c]
            f[nm + '_rank'] = (-s).argsort().argsort()[c]
        for a, codes in sg.cat_codes.items():
            share = np.bincount(codes, weights=x, minlength=codes.max() + 1) / (x.sum() + 1e-9)
            f[f'share_{a}'] = share[codes[c]]
            f[f'code_{a}'] = codes[c]
        for name, sc in knn_feats.items():
            s = sc[u]
            f[name] = s[c]
            f[name + '_rank'] = (-s).argsort().argsort()[c]
        rows.append(pd.DataFrame(f))
    return pd.concat(rows, ignore_index=True)
