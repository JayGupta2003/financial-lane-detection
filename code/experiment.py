"""Dataset construction, training, evaluation helpers."""
import os, time, json
import numpy as np
import torch
from scipy import ndimage
import data as D
from inject import make_spoofs
import models as M

dev = "cuda" if torch.cuda.is_available() else "cpu"
DENSITY = 3.0  # injected spoofs per 1000 events
_base_cache = {}


def base(t):
    if t not in _base_cache:
        _base_cache[t] = D.load_base(t)
    return _base_cache[t]


_scale_cache = {}


def scale(t):
    if t not in _scale_cache:
        b = base(t)
        s, e = D.split_range(len(b["t"]), "train")
        X, _ = D.features(b, s, e, None)
        _scale_cache[t] = D.fit_scale(X)
    return _scale_cache[t]


def build(t, split, seed, density=DENSITY, r=None, res=None, level=None, whole=False, gen=None, decoys=0.0):
    """Returns Windows object (+ spoof list). Inject independently per split via the seed."""
    b = base(t)
    N = len(b["t"])
    s, e = (0, N) if whole else D.split_range(N, split)
    rng = np.random.default_rng(seed)
    sp = gen(b, s, e, density, rng) if gen else make_spoofs(b, s, e, density, rng, r=r, res=res, level=level)
    if decoys > 0:
        from inject import make_decoys
        sp = sp + make_decoys(b, s, e, decoys, np.random.default_rng(seed + 777))
    X, Mk = D.features(b, s, e, sp)
    return D.Windows(D.normalize(X, scale(t)), Mk), sp


def build_multi(ts, split, seed, **kw):
    return [build(t, split, seed + i, **kw)[0] for i, t in enumerate(ts)]


def iterate(wins, ch, bs, shuffle, rng=None, with_decoy=False):
    ids = [(i, j) for i, w in enumerate(wins) for j in range(len(w))]
    if shuffle:
        rng.shuffle(ids)
    for k in range(0, len(ids), bs):
        chunk = ids[k:k + bs]
        xs, ms, ds = [], [], []
        for i in sorted({c[0] for c in chunk}):
            sel = np.array([c[1] for c in chunk if c[0] == i])
            x, m = wins[i].batch(sel, ch)
            xs.append(x); ms.append(m)
            if with_decoy:
                ds.append(wins[i].batch_decoy(sel))
        if with_decoy:
            yield torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ms)), torch.from_numpy(np.concatenate(ds))
        else:
            yield torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ms))


@torch.no_grad()
def predict(model, wins, ch, bs=256):
    model.eval()
    P, Y = [], []
    for x, y in iterate(wins, ch, bs, False):
        P.append(torch.sigmoid(model(x.to(dev))).cpu()); Y.append(y)
    return torch.cat(P), torch.cat(Y)


@torch.no_grad()
def predict_decoy(model, wins, ch, bs=256):
    model.eval()
    P, Y, Dm = [], [], []
    for x, y, d in iterate(wins, ch, bs, False, with_decoy=True):
        P.append(torch.sigmoid(model(x.to(dev))).cpu()); Y.append(y); Dm.append(d)
    return torch.cat(P), torch.cat(Y), torch.cat(Dm)


def decoy_report(model, wins, ch, thr=0.5):
    """Metrics on a test set that contains decoys: spoof IoU/precision/recall plus the fraction of decoy cells flagged."""
    P, Y, Dm = predict_decoy(model, wins, ch)
    r = seg_metrics(P, Y, thr)
    flagged = ((P > thr).float() * Dm).sum().item()
    r["decoy_flag_rate"] = flagged / max(Dm.sum().item(), 1)
    r["decoy_cells"] = int(Dm.sum().item())
    return r


def train(ch, tr, va, seed, use_dice=True, use_bce=True, max_ep=50, patience=10, bs=64, log=print, loss=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    loss_fn = M.get_loss(loss) if loss else None
    m = M.UNet(len(ch)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    best, bad, state, best_ep = -1, 0, None, 0
    for ep in range(max_ep):
        m.train(); t0 = time.time(); tl = 0; nb = 0
        for x, y in iterate(tr, ch, bs, True, rng):
            x, y = x.to(dev), y.to(dev)
            lv = loss_fn(m(x), y) if loss_fn else M.dice_bce_loss(m(x), y, use_dice=use_dice, use_bce=use_bce)
            opt.zero_grad(); lv.backward(); opt.step(); tl += lv.item(); nb += 1
        P, Y = predict(m, va, ch)
        iou = seg_metrics(P, Y)["iou"]
        sch.step(-iou)
        log(f"ep{ep} loss {tl/nb:.4f} val_iou {iou:.4f} {time.time()-t0:.0f}s")
        if iou > best:
            best, bad, best_ep = iou, 0, ep
            state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    return m, dict(best_val_iou=best, best_epoch=best_ep, epochs=ep + 1)


def seg_metrics(P, Y, thr=0.5):
    pr = (P > thr).float()
    tp = (pr * Y).sum().item(); fp = (pr * (1 - Y)).sum().item(); fn = ((1 - pr) * Y).sum().item()
    return dict(iou=tp / (tp + fp + fn + 1e-9), dice=2 * tp / (2 * tp + fp + fn + 1e-9),
                precision=tp / (tp + fp + 1e-9), recall=tp / (tp + fn + 1e-9))


def average_precision(P, Y):
    p = P.flatten().to(dev); y = Y.flatten().to(dev)
    o = torch.argsort(p, descending=True); y = y[o]
    tp = torch.cumsum(y, 0); prec = tp / torch.arange(1, len(y) + 1, device=dev)
    return ((prec * y).sum() / y.sum().clamp_min(1)).item()


def component_recall(P, Y, thr=0.5, min_iou=0.5):
    """Connected-component matching at IoU>=min_iou, within each window (4-connectivity)."""
    n_gt = matched = 0
    errs = []
    pb = (P[:, 0] > thr).numpy(); yb = (Y[:, 0] > 0.5).numpy()
    for p, y in zip(pb, yb):
        gl, gn = ndimage.label(y)
        if gn == 0:
            continue
        pl, pn = ndimage.label(p)
        n_gt += gn
        for g in range(1, gn + 1):
            gm = gl == g
            ids = np.unique(pl[gm]); ids = ids[ids > 0]
            best = 0; bc = None
            for i in ids:
                pm = pl == i
                iou = (gm & pm).sum() / (gm | pm).sum()
                if iou > best:
                    best, bc = iou, pm
            if best >= min_iou:
                matched += 1
                gy, gx = ndimage.center_of_mass(gm); py, px = ndimage.center_of_mass(bc)
                errs.append((abs(gy - py), abs(gx - px)))
    return dict(n_gt=n_gt, matched=matched, recall=matched / max(n_gt, 1),
                med_err_levels=float(np.median([e[0] for e in errs])) if errs else None,
                med_err_cols=float(np.median([e[1] for e in errs])) if errs else None)


def full_eval(model, wins, ch):
    P, Y = predict(model, wins, ch)
    r = seg_metrics(P, Y); r["ap"] = average_precision(P, Y)
    return r, P, Y


def build_time(t, split, seed, dt=1.0, density=DENSITY):
    """Time-based columns: resample the event-level tensor to dt-second bins (last state per bin;
    mask = any spoof event in the bin)."""
    b = base(t); N = len(b["t"]); s, e = D.split_range(N, split)
    sp = make_spoofs(b, s, e, density, np.random.default_rng(seed))
    X, Mk = D.features(b, s, e, sp)
    Xn = D.normalize(X, scale(t))
    tt = b["t"][s:e]; bins = np.floor((tt - tt[0]) / dt).astype(np.int64); nb = int(bins[-1]) + 1
    g = np.arange(nb)
    hi = np.searchsorted(bins, g, side="right"); lo = np.searchsorted(bins, g, side="left")
    idx = np.clip(hi - 1, 0, None)
    cs = np.concatenate([np.zeros((D.H, 1)), np.cumsum(Mk, axis=1)], 1)
    Mt = ((cs[:, hi] - cs[:, lo]) > 0).astype(np.uint8)
    return D.Windows(np.ascontiguousarray(Xn[:, :, idx]), Mt, stride=32)
