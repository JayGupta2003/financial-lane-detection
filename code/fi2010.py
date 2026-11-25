"""External background-structure test on FI-2010 (NoAuction, DecPre). No spoof labels -> no IoU.
Only ask/bid volume channels exist (2-channel model). Volumes are rescaled so the median non-zero volume matches
the LOBSTER train median, then log1p / pooled LOBSTER scale. Reports flagged-pixel rate and flagged-window rate,
compared with the same statistics on clean (uninjected) LOBSTER test windows, plus batch-1 latency."""
import json, os, sys, glob
import numpy as np, torch
import experiment as E, models as M, data as D

OUT = os.path.join(os.path.dirname(__file__), "..", "results"); T = D.TICKERS
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "fi2010", "raw")


def fi_tensor(path):
    a = np.loadtxt(path)[:40]                       # (40, n) time steps in columns
    ask_v = a[1::4][:10]; bid_v = a[3::4][:10]      # levels 1..10
    vol = np.concatenate([ask_v[::-1], bid_v], 0)   # rows: ask lvl10..1, bid lvl1..10
    return vol.astype(np.float32)


def stats(m, X, W=256, stride=256):
    starts = np.arange(0, X.shape[-1] - W + 1, stride)
    flagged_px, flagged_win = [], []
    with torch.no_grad():
        for i in range(0, len(starts), 256):
            xb = torch.from_numpy(np.stack([X[:, :, s:s + W] for s in starts[i:i + 256]])).to(E.dev)
            p = (torch.sigmoid(m(xb)) > 0.5).float().cpu()
            flagged_px.append(p.mean((1, 2, 3))); flagged_win.append((p.amax((1, 2, 3)) > 0).float())
    return dict(windows=len(starts), pixel_rate=float(torch.cat(flagged_px).mean()),
                window_rate=float(torch.cat(flagged_win).mean()))


def run(tag="unet2"):
    import modelio
    m, _ = modelio.load_model(tag)
    # reference: LOBSTER clean test windows, same non-overlapping windowing
    ref = {}
    lob_med = []
    for t in T:
        b = E.base(t); s, e = D.split_range(len(b["t"]), "train"); v = b["vol"][s:e]; lob_med.append(np.median(v[v > 0]))
    lob_med = float(np.mean(lob_med)); sc = np.mean([E.scale(t) for t in T], 0)
    for t in T:
        w, _ = E.build(t, "test", 0, density=0.0)
        ref[t] = stats(m, w.X[D.CH2])
    out = {"lobster_clean_test": ref, "files": {}}
    for f in sorted(glob.glob(f"{RAW}/Test_*.txt")):
        v = fi_tensor(f)
        v = v * (lob_med / np.median(v[v > 0]))
        ask = v.copy(); ask[10:] = 0; bid = v.copy(); bid[:10] = 0
        X = np.stack([ask, bid]); X = np.clip(np.log1p(X) / sc[:2, None, None], 0, 1).astype(np.float32)
        out["files"][os.path.basename(f)] = stats(m, X)
        print(os.path.basename(f), out["files"][os.path.basename(f)], flush=True)
    json.dump(out, open(f"{OUT}/fi2010_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "unet2")
