"""Stylized execution simulation: buy parent order, TWAP-like slices sent as IOC limit orders.
The algorithm believes the *visible* ask depth. Injected spoof volume is visible but vanishes the instant
a marketable order reaches it (real depth R = base volume; visible V = R + spoof volume).
Arms: unfiltered (believes V), filtered (believes V with model-masked cells removed), oracle (true mask), clean (no spoofs).
Each slice: sends (target cumulative - filled), limit level k = shallowest level whose believed cumulative depth
covers the slice (cap 10); fills only against real depth up to level k; unfilled quantity carries over. The final
slice sweeps all 10 levels; any remainder pays last-level price + PENALTY ticks.
Usage: python execsim.py [model_tag] [n_orders]"""
import json, os, sys
import numpy as np, torch
import experiment as E, models as M, data as D

OUT = os.path.join(os.path.dirname(__file__), "..", "results"); T = D.TICKERS
CH = {"unet2": D.CH2, "unet3": D.CH3, "unet3_bce": D.CH3, "unet6": D.CH6}
NSLICE, HORIZON, SLICE_FRAC, PENALTY = 50, 6000, 0.4, 10 * 100   # slice = SLICE_FRAC x median 10-level ask depth; penalty = 10 ticks
log = lambda s: print(s, flush=True)


def simulate(tick_prices, R, B, mid0, Q, slice_ev):
    """tick_prices (n,10) ask prices level1..10 per event; R real depth (n,10); B belief depth (n,10)."""
    filled, cost, sends, first_fill, first_sent, done_at = 0.0, 0.0, 0, 0.0, 0.0, None
    for i in range(NSLICE):
        k0 = i * slice_ev
        if filled >= Q - 1e-9:
            break
        send = Q * (i + 1) / NSLICE - filled
        last = i == NSLICE - 1
        if last:
            send = Q - filled
        cum = np.cumsum(B[k0])
        lvl = 10 if last else (int(np.searchsorted(cum, send)) + 1 if cum[-1] >= send else 10)
        lvl = min(lvl, 10)
        got, rem = 0.0, send
        for l in range(lvl):
            take = min(R[k0, l], rem)
            cost += take * tick_prices[k0, l]; got += take; rem -= take
        sends += 1; first_sent += send; first_fill += got; filled += got
        if last and rem > 1e-9:
            cost += rem * (tick_prices[k0, 9] + PENALTY); filled += rem
        if filled >= Q - 1e-9 and done_at is None:
            done_at = k0
    return cost / filled, sends, first_fill / max(first_sent, 1e-9), (done_at if done_at is not None else HORIZON)


def stitch_masks(m, ch, t, seed, density):
    """Model mask per (H, n) over the whole test split of ticker t, using non-overlapping 256-event windows."""
    b = E.base(t); s, e = D.split_range(len(b["t"]), "test")
    rng = np.random.default_rng(seed)
    from inject import make_spoofs
    sp = make_spoofs(b, s, e, density, rng)
    X, Mk = D.features(b, s, e, sp)
    Xn = D.normalize(X, E.scale(t))
    w = D.Windows(Xn, Mk, stride=256)
    P, _ = E.predict(m, [w], ch)
    pm = np.zeros(Mk.shape, np.float32)
    for st, p in zip(w.starts, P[:, 0].numpy()):
        pm[:, st:st + 256] = p
    spoof = np.zeros((e - s, D.H), np.float32)
    for q in sp:
        for row, qty in zip(q["rows"], q["qs"]):
            spoof[q["t0"]:q["t1"], row] += qty
    return s, e, spoof, Mk.T.astype(np.float32), pm.T


def run(tag="unet3", n_orders=2500, densities=(1.5, 3.0, 6.0), thr=0.5, out_path=None):
    import modelio
    m, ch = modelio.load_model(tag); out = {}
    for dens in densities:
        rng = np.random.default_rng(int(dens * 10))
        arms = {a: dict(IS=[], sends=[], fill=[], done=[]) for a in ("unfiltered", "filtered", "oracle", "clean")}
        per = n_orders // len(T)
        for ti, t in enumerate(T):
            s, e, spoof, truth, pm = stitch_masks(m, ch, t, 7000 + ti, dens)
            b = E.base(t)
            ask = np.stack([b["px"][s:e, D.ask_row(l)] for l in range(1, 11)], 1).astype(np.float64)
            Rv = np.stack([b["vol"][s:e, D.ask_row(l)] for l in range(1, 11)], 1).astype(np.float64)
            Sp = np.stack([spoof[:, D.ask_row(l)] for l in range(1, 11)], 1)
            Tm = np.stack([truth[:, D.ask_row(l)] for l in range(1, 11)], 1)
            Pm = np.stack([(pm[:, D.ask_row(l)] > thr) for l in range(1, 11)], 1).astype(np.float64)
            Q = NSLICE * SLICE_FRAC * np.median(Rv.sum(1))
            mid = b["mid"][s:e]
            for k0 in rng.integers(300, e - s - HORIZON - 1, size=per):
                sl = slice(k0, k0 + HORIZON); mid0 = mid[k0]
                V = Rv[sl] + Sp[sl]
                beliefs = {"unfiltered": V, "filtered": V * (1 - Pm[sl]), "oracle": V * (1 - Tm[sl]), "clean": Rv[sl]}
                for a, B in beliefs.items():
                    # clean arm: real book only (no spoof volume in the visible book either)
                    avg, sends, fr, done = simulate(ask[sl], Rv[sl], B, mid0, Q, HORIZON // NSLICE)
                    arms[a]["IS"].append((avg - mid0) / mid0 * 1e4); arms[a]["sends"].append(sends)
                    arms[a]["fill"].append(fr); arms[a]["done"].append(done)
        res = {}
        for a, d in arms.items():
            res[a] = {k: float(np.mean(v)) for k, v in d.items()}
        IS_u, IS_f = np.array(arms["unfiltered"]["IS"]), np.array(arms["filtered"]["IS"])
        diff = IS_f - IS_u; bs = np.random.default_rng(0)
        boots = np.array([diff[bs.integers(0, len(diff), len(diff))].mean() for _ in range(5000)])
        res["delta_IS_bps"] = dict(mean=float(diff.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                   p_two_sided=float(2 * min((boots >= 0).mean(), (boots <= 0).mean())), n=len(diff))
        out[str(dens)] = res; log(f"density {dens}: {json.dumps(res)[:400]}")
        json.dump(out, open(out_path or f"{OUT}/execution_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "unet3", int(sys.argv[2]) if len(sys.argv) > 2 else 2500)
