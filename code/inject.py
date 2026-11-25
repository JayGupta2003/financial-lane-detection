"""Spoof-like liquidity injector on the level-indexed LOB (parameters follow the paper).
Size ratio r in [2,20] x local level volume (log-uniform), distance = level 2..10,
residence 5..100 events (log-uniform), 35% price-triggered early cancel, 25% multi-level."""
import numpy as np
from data import L, ask_row, bid_row

TICK = 100  # 0.01 USD in LOBSTER price units


def make_spoofs(base, s, e, density, rng, r=None, res=None, level=None):
    n = e - s
    vol, mid = base["vol"][s:e], base["mid"][s:e]
    med = np.median(vol[vol > 0]) if (vol > 0).any() else 100.0
    k = int(density * n / 1000)
    out = []
    for t0 in rng.integers(200, n - 120, size=k):
        side = rng.integers(0, 2)                              # 0 ask, 1 bid
        lv = int(level if level is not None else rng.integers(2, L + 1))
        multi = rng.random() < 0.25
        nl = int(rng.integers(2, 4)) if multi else 1
        lvls = [l for l in range(lv, min(lv + nl, L) + 1)]
        rows = [(ask_row if side == 0 else bid_row)(l) for l in lvls]
        local = vol[t0 - 200:t0, rows[0]]
        local = local[local > 0].mean() if (local > 0).any() else med
        rr = r if r is not None else np.exp(rng.uniform(np.log(2), np.log(20)))
        q = rr * local
        dur = int(res if res is not None else round(np.exp(rng.uniform(np.log(5), np.log(100)))))
        t1 = min(t0 + dur, n)
        if rng.random() < 0.35:                                # price-triggered cancel
            mv = (mid[t0:t1] - mid[t0]) * (1 if side == 0 else -1)   # move toward the order
            hit = np.nonzero(mv >= TICK)[0]
            if len(hit) and hit[0] >= 2:
                t1 = t0 + int(hit[0])
        out.append(dict(t0=int(t0), t1=int(max(t1, t0 + 2)), rows=rows, qs=[q / len(rows)] * len(rows),
                        ratio=float(rr), dur=int(t1 - t0), side=int(side), level=lv))
    return out


# ---- out-of-generator families (never seen in training) ----
def _one(base_vol, t0, side, lv, q_ratio, dur, n, nl=1):
    lvls = [l for l in range(lv, min(lv + nl, L) + 1)]
    rows = [(ask_row if side == 0 else bid_row)(l) for l in lvls]
    local = base_vol[max(t0 - 200, 0):t0, rows[0]]
    local = local[local > 0].mean() if (local > 0).any() else 100.0
    t1 = min(t0 + dur, n)
    return dict(t0=int(t0), t1=int(max(t1, t0 + 2)), rows=rows, qs=[q_ratio * local / len(rows)] * len(rows),
                ratio=float(q_ratio), dur=int(t1 - t0), side=int(side), level=lv)


def gen_oversized_long(b, s, e, density, rng):
    """size 20-40x (beyond training range 2-20x), residence 100-300 events (beyond 5-100)."""
    n = e - s; vol = b["vol"][s:e]
    return [_one(vol, t0, rng.integers(0, 2), int(rng.integers(2, L + 1)),
                 np.exp(rng.uniform(np.log(20), np.log(40))), int(np.exp(rng.uniform(np.log(100), np.log(300)))), n)
            for t0 in rng.integers(200, n - 320, size=int(density * n / 1000))]


def gen_staggered(b, s, e, density, rng):
    """staggered layering: 3-5 consecutive levels posted 8 events apart, pulled together."""
    n = e - s; vol = b["vol"][s:e]; out = []
    for t0 in rng.integers(200, n - 200, size=int(density * n / 1000 / 2)):
        side = rng.integers(0, 2); k = int(rng.integers(3, 6)); lv = int(rng.integers(2, L - k + 2))
        dur = int(rng.integers(30, 90)); end = t0 + dur + 8 * k
        for i in range(k):
            out.append(_one(vol, t0 + 8 * i, side, lv + i, np.exp(rng.uniform(np.log(3), np.log(8))), end - (t0 + 8 * i), n))
    return out


def gen_flicker(b, s, e, density, rng):
    """flickering: the same level posted for 5 events and pulled, every 15 events (6 times)."""
    n = e - s; vol = b["vol"][s:e]; out = []
    for t0 in rng.integers(200, n - 150, size=int(density * n / 1000 / 3)):
        side = rng.integers(0, 2); lv = int(rng.integers(2, L + 1)); r = np.exp(rng.uniform(np.log(4), np.log(15)))
        for i in range(6):
            out.append(_one(vol, t0 + 15 * i, side, lv, r, 5, n))
    return out


OOD = {"oversized_long": gen_oversized_long, "staggered": gen_staggered, "flicker": gen_flicker}


# ---- hard negatives (decoys): genuine-looking large orders that must NOT be flagged ----
def make_decoys(base, s, e, density, rng):
    """Half 'executed' decoys (same size/level/lifetime law as spoofs, but the order trades -> execution impulse, no
    cancel impulse) and half 'persistent' decoys (150-400 events, cancelled late). Label 0 (mask value 2)."""
    n = e - s; vol = base["vol"][s:e]; out = []
    for t0 in rng.integers(200, n - 420, size=int(density * n / 1000)):
        side = int(rng.integers(0, 2)); lv = int(rng.integers(2, L + 1)); r = float(np.exp(rng.uniform(np.log(2), np.log(20))))
        if rng.random() < 0.5:
            dur = int(round(np.exp(rng.uniform(np.log(5), np.log(100))))); end = "exec"
        else:
            dur = int(round(np.exp(rng.uniform(np.log(150), np.log(400))))); end = "cancel"
        d = _one(vol, int(t0), side, lv, r, dur, n)
        d["decoy"] = True; d["end"] = end
        out.append(d)
    return out
