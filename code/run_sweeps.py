"""Sweeps (spoof size, residence), volatility regimes, per-asset / held-out asset, latency, components.
Usage: python run_sweeps.py [model_tag ...]   (default unet3 unet6; needs results/<tag>_seed0.pt)"""
import json, os, sys, time
import numpy as np, torch
import experiment as E, models as M, data as D

OUT = os.path.join(os.path.dirname(__file__), "..", "results"); T = D.TICKERS
CH = {"unet2": D.CH2, "unet3": D.CH3, "unet3_bce": D.CH3, "unet6": D.CH6}
log = lambda s: print(s, flush=True)


def load(tag):
    import modelio
    return modelio.load_model(tag)


def window_vol(ws):
    """realized volatility proxy per window: std of mid-price change (ticks) over the window's events."""
    v = []
    for t, w in zip(T, ws):
        b = E.base(t); s, _ = D.split_range(len(b["t"]), "test")
        mid = b["mid"][s:]
        v += [np.std(np.diff(mid[st:st + 256])) / 100.0 for st in w.starts]
    return np.array(v)


def run(tags):
    res = json.load(open(f"{OUT}/sweeps.json")) if os.path.exists(f"{OUT}/sweeps.json") else {}
    for tag in tags:
        m, ch = load(tag); R = res.setdefault(tag, {})
        # size sweep / residence sweep (fresh test injections with fixed parameter)
        R["size"] = {}
        for r in (2, 3, 5, 10, 20):
            te = E.build_multi(T, "test", 1000 + r, r=r)
            R["size"][r] = E.full_eval(m, te, ch)[0]; log(f"{tag} size {r}x {R['size'][r]['iou']:.3f}")
        R["duration"] = {}
        for d in (5, 10, 20, 50, 100):
            te = E.build_multi(T, "test", 2000 + d, res=d)
            R["duration"][d] = E.full_eval(m, te, ch)[0]; log(f"{tag} dur {d} {R['duration'][d]['iou']:.3f}")
        # volatility regimes on the standard test windows (quartiles of window volatility)
        te = E.build_multi(T, "test", 300)
        P, Y = E.predict(m, te, ch); vol = window_vol(te)
        # rank-based quartiles with a random tie-break (many windows have zero price change, so quantile edges coincide)
        jit = vol + np.random.default_rng(0).uniform(0, 1e-9, len(vol))
        reg = np.minimum((np.argsort(np.argsort(jit)) * 4) // len(vol), 3); R["regime"] = {}
        for k, name in enumerate(["Q1 (calm)", "Q2", "Q3", "Q4 (volatile)"]):
            sel = torch.from_numpy(reg == k)
            R["regime"][name] = {**E.seg_metrics(P[sel], Y[sel]), "n_windows": int(sel.sum()),
                                 "vol_range_ticks": [float(vol[reg == k].min()), float(vol[reg == k].max())]}
            log(f"{tag} {name} {R['regime'][name]['iou']:.3f}")
        # per-asset test performance (model trained on all five train splits)
        R["per_asset"] = {t: E.full_eval(m, [te[i]], ch)[0] for i, t in enumerate(T)}
        # latency, batch 1
        x = torch.randn(1, len(ch), 20, 256, device=E.dev)
        with torch.no_grad():
            for _ in range(50): m(x)
            ts = []
            for _ in range(500):
                torch.cuda.synchronize(); t0 = time.perf_counter(); m(x); torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
        R["latency_ms"] = dict(mean=float(np.mean(ts)), p95=float(np.percentile(ts, 95)), params=M.count_params(m))
        log(f"{tag} latency {R['latency_ms']}")
        json.dump(res, open(f"{OUT}/sweeps.json", "w"), indent=1)
    # held-out asset: train without INTC, test on INTC (compare to the all-asset model's INTC test)
    for key, dice in (("heldout", True), ("heldout_bce", False)):
        if key in res or (key == "heldout_bce" and "--bce-heldout" not in sys.argv and "heldout" not in res):
            continue
        tr = E.build_multi([t for t in T if t != "INTC"], "train", 100)
        va = E.build_multi([t for t in T if t != "INTC"], "val", 200)
        m, info = E.train(D.CH3, tr, va, 0, use_dice=dice, log=lambda s: log(f"  [{key}] " + s))
        te = [E.build("INTC", "test", 300 + T.index("INTC"))[0]]
        res[key] = {"INTC_unseen": E.full_eval(m, te, D.CH3)[0], **info}
        json.dump(res, open(f"{OUT}/sweeps.json", "w"), indent=1)
    log("SWEEPS DONE")


if __name__ == "__main__":
    run([a for a in sys.argv[1:] if not a.startswith("--")] or ["unet3", "unet6"])
