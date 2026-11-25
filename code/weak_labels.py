"""Weak labels from REAL order lifecycles (LOBSTER order IDs). A candidate spoof-like order is a real limit order that
 (i) arrives at book level >= 2, (ii) has size >= R x the recent mean volume of its level, (iii) is fully deleted within
 <= 100 events of arrival, and (iv) never trades. These are rule-based candidates, NOT verified manipulation.
Compares model flags on clean test books with the candidates (pixel IoU, component overlap) against a volume-threshold
detector and a random-flag control. Usage: python weak_labels.py <tag> [<tag> ...]"""
import json, os, sys
import numpy as np, pandas as pd, torch
from scipy import ndimage
import experiment as E, models as M, data as D, modelio

OUT = os.path.join(os.path.dirname(__file__), "..", "results"); T = D.TICKERS
MAXLIFE, RATIOS = 100, (3.0, 5.0)


def candidates(t, ratio):
    """Returns weak-label mask (H, N) over all events of ticker t plus list of candidate dicts."""
    mp, _ = D._paths(t)
    msg = pd.read_csv(mp, header=None, names=["t", "type", "id", "size", "price", "dir"])
    b = E.base(t); N = len(msg); vol, px = b["vol"], b["px"]
    typ = msg["type"].to_numpy(); oid = msg["id"].to_numpy(); size = msg["size"].to_numpy(); price = msg["price"].to_numpy(); side = msg["dir"].to_numpy()
    arrive = {}
    life = {}
    for k in range(N):
        ty = typ[k]
        if ty == 1:
            arrive[oid[k]] = k
        elif oid[k] in arrive:
            if ty == 3:
                life.setdefault(oid[k], {})["del"] = k
            elif ty in (4, 5):
                life.setdefault(oid[k], {})["exec"] = True
    mask = np.zeros((D.H, N), np.uint8); cands = []
    for o, k0 in arrive.items():
        d = life.get(o, {})
        if "del" not in d or d.get("exec") or d["del"] - k0 > MAXLIFE or d["del"] - k0 < 2:
            continue
        k1 = d["del"]; sd = side[k0]; rows = range(0, D.L) if sd == -1 else range(D.L, D.H)
        row_hit = np.nonzero(px[k0, list(rows)] == price[k0])[0]
        if len(row_hit) == 0:
            continue
        row0 = list(rows)[row_hit[0]]
        level = (D.L - row0) if sd == -1 else (row0 - D.L + 1)
        if level < 2:
            continue
        loc = vol[max(k0 - 200, 0):k0, row0]; loc = loc[loc > 0].mean() if (loc > 0).any() else None
        if loc is None or size[k0] < ratio * loc:
            continue
        for k in range(k0, k1):
            hit = np.nonzero(px[k, list(rows)] == price[k0])[0]
            if len(hit):
                mask[list(rows)[hit[0]], k] = 1
        cands.append(dict(k0=int(k0), k1=int(k1), level=int(level), ratio=float(size[k0] / loc)))
    return mask, cands


def stitch(m, ch, wins):
    P, _ = E.predict(m, [wins], ch)
    n = wins.X.shape[-1]; out = np.zeros((D.H, n), np.float32)
    for st, p in zip(wins.starts, P[:, 0].numpy()):
        out[:, st:st + 256] = p
    return out


def compare(pred, weak, s):
    """pred (H, n) boolean over test slice; weak (H, N) full-day mask -> restrict to [s, s+n)."""
    n = pred.shape[1]; wk = weak[:, s:s + n] > 0
    inter = (pred & wk).sum(); union = (pred | wk).sum()
    lp, npred = ndimage.label(pred); lw, nweak = ndimage.label(wk)
    pred_hit = sum(1 for c in range(1, npred + 1) if wk[lp == c].any())
    weak_hit = sum(1 for c in range(1, nweak + 1) if pred[lw == c].any())
    return dict(iou=float(inter / max(union, 1)), pred_components=int(npred), weak_components=int(nweak),
                pred_overlapping_weak=pred_hit / max(npred, 1), weak_overlapping_pred=weak_hit / max(nweak, 1), flagged_px=float(pred.mean()))


def main(tags):
    res = {}
    cache = {}
    for t in T:
        for r in RATIOS:
            cache[(t, r)] = candidates(t, r)
    for tag in tags:
        m, ch = modelio.load_model(tag); R = res.setdefault(tag, {})
        for r in RATIOS:
            rows = {}
            for t in T:
                b = E.base(t); s, e = D.split_range(len(b["t"]), "test")
                w, _ = E.build(t, "test", 0, density=0.0)
                w = D.Windows(w.X, w.M, stride=256)
                pred = stitch(m, ch, w) > 0.5
                weak, cands = cache[(t, r)]
                n = pred.shape[1]
                # volume-threshold control tuned on validation in main.json; random control with same flagged-pixel rate
                thr = json.load(open(f"{OUT}/main.json"))["volume_threshold"]["param"]
                vt = (np.maximum(w.X[0], w.X[1])[:, :n] > thr)
                rng = np.random.default_rng(0)
                rnd = ndimage.binary_dilation(rng.random(pred.shape) < pred.mean() / 30, structure=np.ones((3, 10)))
                rows[t] = dict(model=compare(pred, weak, s), volume_threshold=compare(vt, weak, s), random=compare(rnd, weak, s),
                               n_candidates_test=sum(1 for c in cands if s <= c["k0"] < s + n))
            R[str(r)] = rows
            print(tag, "ratio", r, {t: (round(v["model"]["pred_overlapping_weak"], 2), round(v["model"]["weak_overlapping_pred"], 2),
                                          v["n_candidates_test"]) for t, v in rows.items()}, flush=True)
    json.dump(res, open(f"{OUT}/weak_labels.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["unet3_bce", "unet6"])
