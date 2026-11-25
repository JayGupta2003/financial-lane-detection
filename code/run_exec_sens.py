"""Execution sensitivity: vary slice size and number of slices (density 3 spoofs/1000 events, 2500 paired orders)."""
import json, os, sys
import execsim as X
OUT = X.OUT
tag = sys.argv[1] if len(sys.argv) > 1 else "unet3_bce"
res = {}
for nslice in (25, 50, 100):
    for frac in (0.1, 0.2, 0.4, 0.8, 1.6):
        X.NSLICE, X.SLICE_FRAC = nslice, frac
        tmp = os.path.join(OUT, "_exec_tmp.json")
        X.run(tag, 2500, densities=(3.0,), out_path=tmp)
        r = json.load(open(tmp))["3.0"]; res[f"nslice{nslice}_frac{frac}"] = r
        print(nslice, frac, round(r["unfiltered"]["IS"], 3), round(r["delta_IS_bps"]["mean"], 4), flush=True)
        json.dump(res, open(os.path.join(OUT, f"execution_sens_{tag}.json"), "w"), indent=1)
