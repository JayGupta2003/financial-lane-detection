"""How do models trained WITHOUT decoys treat decoys (genuine-looking large orders, label 0)?"""
import json, os, sys
import experiment as E, data as D, modelio
OUT = os.path.join(os.path.dirname(__file__), "..", "results"); T = D.TICKERS
tags = sys.argv[1:] or ["unet2", "unet3", "unet3_bce", "unet6"]
te_h = E.build_multi(T, "test", 300, decoys=3.0)
res = {}
for tag in tags:
    m, ch = modelio.load_model(tag)
    res[tag] = E.decoy_report(m, te_h, ch); print(tag, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[tag].items()}, flush=True)
json.dump(res, open(os.path.join(OUT, "decoys_existing.json"), "w"), indent=1)
