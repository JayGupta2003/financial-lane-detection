"""LOBSTER loader, level-indexed LOB tensors, spoof-aware feature construction.

Grid: H=20 rows = 10 ask levels (row 0 = level 10 ... row 9 = level 1) then 10 bid levels
(row 10 = level 1 ... row 19 = level 10). One column per LOB event (event-based time axis).
Channels: 0 ask vol, 1 bid vol, 2 cancel rate | 3 arrival, 4 execution, 5 order age.
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import lfilter

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "lobster")
TICKERS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
L = 10          # levels per side
H = 2 * L
DECAY = 0.9     # per-event exponential decay for rate channels
SPLITS = {"train": (0.0, 0.60), "val": (0.60, 0.75), "test": (0.75, 1.0)}


def _paths(t):
    d = os.path.join(DATA, t)
    f = os.listdir(d)
    m = [x for x in f if "message" in x][0]
    o = [x for x in f if "orderbook" in x][0]
    return os.path.join(d, m), os.path.join(d, o)


def ask_row(level):  # level 1..10
    return L - level


def bid_row(level):
    return L - 1 + level


def _row_of(price, side, ask_p, bid_p):
    """Row index of `price` in the book (side -1=ask, 1=bid), or -1 if not among the 10 levels."""
    out = np.full(len(price), -1, dtype=np.int16)
    am = ask_p == price[:, None]
    bm = bid_p == price[:, None]
    a_hit = am.any(1) & (side == -1)
    b_hit = bm.any(1) & (side == 1)
    out[a_hit] = (L - 1 - am[a_hit].argmax(1))          # level idx j -> row L-1-j
    out[b_hit] = L + bm[b_hit].argmax(1)
    return out


def load_base(ticker):
    mp, op = _paths(ticker)
    msg = pd.read_csv(mp, header=None, names=["t", "type", "id", "size", "price", "dir"])
    ob = pd.read_csv(op, header=None).to_numpy(dtype=np.int64)
    ask_p, ask_v = ob[:, 0::4], ob[:, 1::4]
    bid_p, bid_v = ob[:, 2::4], ob[:, 3::4]
    t = msg["t"].to_numpy()
    typ = msg["type"].to_numpy()
    size = msg["size"].to_numpy().astype(np.float64)
    price = msg["price"].to_numpy()
    side = msg["dir"].to_numpy()
    N = len(t)
    # book before / after each event
    ap_prev = np.vstack([ask_p[:1], ask_p[:-1]])
    bp_prev = np.vstack([bid_p[:1], bid_p[:-1]])
    vol = np.zeros((N, H), np.float32)
    vol[:, :L] = ask_v[:, ::-1]
    vol[:, L:] = bid_v
    px = np.zeros((N, H), np.int64)
    px[:, :L] = ask_p[:, ::-1]
    px[:, L:] = bid_p
    cancel = np.zeros((N, H), np.float32)
    arrive = np.zeros((N, H), np.float32)
    execu = np.zeros((N, H), np.float32)
    idx = np.arange(N)
    m = (typ == 2) | (typ == 3)
    r = _row_of(price, side, ap_prev, bp_prev)
    ok = m & (r >= 0)
    cancel[idx[ok], r[ok]] = size[ok]
    m = (typ == 4) | (typ == 5)
    ok = m & (r >= 0)
    execu[idx[ok], r[ok]] = size[ok]
    m = typ == 1
    r2 = _row_of(price, side, ask_p, bid_p)
    ok = m & (r2 >= 0)
    arrive[idx[ok], r2[ok]] = size[ok]
    # age: seconds since the price at that row last changed
    changed = np.ones((N, H), bool)
    changed[1:] = px[1:] != px[:-1]
    tt = np.broadcast_to(t[:, None], (N, H))
    last = np.maximum.accumulate(np.where(changed, tt, -np.inf), axis=0)
    age = (tt - last).astype(np.float32)
    mid = (ask_p[:, 0] + bid_p[:, 0]) / 2.0
    return dict(t=t, vol=vol, px=px, cancel=cancel, arrive=arrive, execu=execu, age=age, mid=mid, ticker=ticker)


def split_range(N, split, gap=256):
    a, b = SPLITS[split]
    s, e = int(a * N), int(b * N)
    if split != "train":
        s += gap
    return s, e


def features(base, s, e, spoof=None):
    """Return X (6,H,n) float32 un-normalized (log1p applied later) and mask (H,n) for events s:e.
    `spoof` is the output of inject.make_spoofs (relative to the slice)."""
    vol = base["vol"][s:e].copy()
    canc = base["cancel"][s:e].copy()
    arr = base["arrive"][s:e].copy()
    exe = base["execu"][s:e].copy()
    age = base["age"][s:e].copy()
    t = base["t"][s:e]
    n = e - s
    mask = np.zeros((n, H), np.uint8)
    if spoof is not None:
        for sp in spoof:
            k0, k1 = sp["t0"], sp["t1"]
            decoy = sp.get("decoy", False)            # hard negative: genuine large order, label 0 (mask value 2)
            for row, q in zip(sp["rows"], sp["qs"]):
                vol[k0:k1, row] += q
                arr[k0, row] += q
                if k1 < n:
                    if sp.get("end") == "exec":       # the order trades instead of being pulled
                        exe[k1, row] += q
                    else:
                        canc[k1, row] += q
                if decoy:
                    sl = mask[k0:k1, row]; sl[sl == 0] = 2
                else:
                    mask[k0:k1, row] = 1
    rate = lambda x: lfilter([1.0], [1.0, -DECAY], x, axis=0).astype(np.float32)
    ask = vol.copy(); ask[:, L:] = 0
    bid = vol.copy(); bid[:, :L] = 0
    X = np.stack([ask, bid, rate(canc), rate(arr), rate(exe), age], 0)   # (6,n,H)
    return np.ascontiguousarray(X.transpose(0, 2, 1)), np.ascontiguousarray(mask.T)


def fit_scale(X):
    """99.9th percentile of log1p per channel (computed on clean train data only)."""
    return np.array([np.percentile(np.log1p(X[c]), 99.9) + 1e-6 for c in range(X.shape[0])], np.float32)


def normalize(X, scale):
    return np.clip(np.log1p(X) / scale[:, None, None], 0, 1).astype(np.float32)


class Windows:
    """Sliding windows over a (C,H,n) tensor and (H,n) mask."""

    def __init__(self, X, M, W=256, stride=128):
        self.X, self.M, self.W = X, M, W
        self.starts = np.arange(0, X.shape[-1] - W + 1, stride)

    def __len__(self):
        return len(self.starts)

    def batch(self, ids, channels):
        xs = np.stack([self.X[channels, :, s:s + self.W] for s in self.starts[ids]])
        ms = np.stack([self.M[:, s:s + self.W] for s in self.starts[ids]])[:, None]
        return xs, (ms == 1).astype(np.float32)          # value 2 marks decoys (label 0)

    def batch_decoy(self, ids):
        ms = np.stack([self.M[:, s:s + self.W] for s in self.starts[ids]])[:, None]
        return (ms == 2).astype(np.float32)


CH3, CH6, CH2 = [0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1]

if __name__ == "__main__":
    for tk in TICKERS:
        b = load_base(tk)
        N = len(b["t"])
        dur = b["t"][-1] - b["t"][0]
        print(tk, N, "events", f"{N/dur:.1f}/s", "nonzero cancel/arr/exec:",
              [(b[k] > 0).any(1).mean().round(3) for k in ("cancel", "arrive", "execu")])
