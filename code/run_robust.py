"""(1) out-of-generator test, (2) real-data sanity check on clean books, (3) time- vs event-based column ablation.
Usage: python run_robust.py ood|real <tag>   or   python run_robust.py timecols"""
import json, os, sys
import numpy as np, torch
from scipy import ndimage
import experiment as E, models as M, data as D
from inject import OOD

OUT = os.path.join(os.path.dirname(__file__), "..", "results"); T = D.TICKERS
CH = {"unet2": D.CH2, "unet3": D.CH3, "unet3_bce": D.CH3, "unet6": D.CH6}
log = lambda s: print(s, flush=True)


def load(tag):
    import modelio
    return modelio.load_model(tag)


def ood(tag):
    m, ch = load(tag); res = {}
    ref = json.load(open(f"{OUT}/main.json"))["volume_threshold"]["param"]
    for name, gen in OOD.items():
        te = [E.build(t, "test", 5000 + i, density=3.0, gen=gen)[0] for i, t in enumerate(T)]
        r, P, Y = E.full_eval(m, te, ch)
        vt = torch.cat([M.volume_threshold_mask(x, ref) for x, _ in E.iterate(te, D.CH3, 256, False)])
        res[name] = {**r, "volume_threshold_iou": E.seg_metrics(vt, Y)["iou"], "pos_pixels": float(Y.mean())}
        log(f"{tag} OOD {name} {res[name]}")
    json.dump(res, open(f"{OUT}/ood_{tag}.json", "w"), indent=1)


def real(tag):
    """Detections on clean (uninjected) real books: rates, component stats, 'posted-then-pulled' share, top examples."""
    m, ch = load(tag); res = {"per_asset": {}}; examples = []
    for i, t in enumerate(T):
        b = E.base(t); s, e = D.split_range(len(b["t"]), "test")
        w, _ = E.build(t, "test", 0, density=0.0)
        w = D.Windows(w.X, w.M, stride=256)
        P, _ = E.predict(m, [w], ch)
        hours = (b["t"][e - 1] - b["t"][s]) / 3600
        comps, pulled = [], 0
        for k, p in enumerate(P[:, 0].numpy()):
            lab, n = ndimage.label(p > 0.5)
            x = w.X[:, :, w.starts[k]:w.starts[k] + 256]
            raw = np.expm1(x[:2] * E.scale(t)[:2, None, None]).max(0)    # back to raw share volume
            for c in range(1, n + 1):
                cm = lab == c; rows, cols = np.nonzero(cm)
                vin = raw[cm].mean()
                end = cols.max()
                after = raw[rows, end + 3].mean() if end + 3 < 256 else vin
                pul = bool(after < 0.3 * vin); pulled += pul
                comps.append(dict(win=k, n_cells=int(cm.sum()), width=int(cols.max() - cols.min() + 1),
                                  peak=float(p[rows, cols].max()), pulled=pul))
        for c in sorted(comps, key=lambda c: -(c["peak"] * c["n_cells"]))[:2]:
            st = w.starts[c["win"]]
            examples.append(dict(ticker=t, X=w.X[:, :, st:st + 256][ch].copy(), P=P[c["win"], 0].numpy().copy(),
                                 n_cells=c["n_cells"], peak=c["peak"]))
        res["per_asset"][t] = dict(windows=len(P), hours=float(hours), components=len(comps), per_hour=len(comps) / float(hours),
                                   window_rate=float((P.amax((1, 2, 3)) > 0.5).float().mean()),
                                   pixel_rate=float((P > 0.5).float().mean()),
                                   median_width_cols=float(np.median([c["width"] for c in comps])) if comps else None,
                                   pulled_share=pulled / max(len(comps), 1))
        log(f"{tag} real {t} {res['per_asset'][t]}")
    examples = sorted(examples, key=lambda d: -(d["peak"] * d["n_cells"]))[:6]
    np.savez(f"{OUT}/real_examples_{tag}.npz", X=np.stack([d["X"] for d in examples]), P=np.stack([d["P"] for d in examples]),
             tick=np.array([d["ticker"] for d in examples]))
    json.dump(res, open(f"{OUT}/real_{tag}.json", "w"), indent=1)


def timecols(nseeds=3):
    """Train unet3 on 1-second time bins vs the event-based baseline (compare with main.json unet3)."""
    tr = [E.build_time(t, "train", 100 + i) for i, t in enumerate(T)]
    va = [E.build_time(t, "val", 200 + i) for i, t in enumerate(T)]
    te = [E.build_time(t, "test", 300 + i) for i, t in enumerate(T)]
    log(f"time-based windows train {sum(map(len,tr))} val {sum(map(len,va))} test {sum(map(len,te))}")
    runs = []
    for seed in range(nseeds):
        m, info = E.train(D.CH3, tr, va, seed, log=lambda s: log(f"  [time s{seed}] {s}"))
        r, _, _ = E.full_eval(m, te, D.CH3); r.update(info); runs.append(r); log(f"time seed{seed} {r}")
        json.dump(runs, open(f"{OUT}/timecols.json", "w"), indent=1)


if __name__ == "__main__":
    st = sys.argv[1]
    {"ood": lambda: ood(sys.argv[2]), "real": lambda: real(sys.argv[2]), "timecols": timecols}[st]()
