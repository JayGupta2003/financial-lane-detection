"""Batch-1 latency on an otherwise idle GPU (warm-up, CUDA sync, 1000 timed runs)."""
import json, os, time
import numpy as np, torch
import experiment as E, modelio, models as M
OUT = os.path.join(os.path.dirname(__file__), "..", "results"); res = {}
for tag in ("m3b@0", "m6b@0"):
    m, ch = modelio.load_model(tag); x = torch.randn(1, len(ch), 20, 256, device=E.dev); ts = []
    with torch.no_grad():
        for _ in range(100): m(x)
        for _ in range(1000):
            torch.cuda.synchronize(); t0 = time.perf_counter(); m(x); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    res[tag] = dict(mean=float(np.mean(ts)), p95=float(np.percentile(ts, 95)), median=float(np.median(ts)), params=M.count_params(m))
    print(tag, res[tag], flush=True)
json.dump(res, open(os.path.join(OUT, "latency.json"), "w"), indent=1)
