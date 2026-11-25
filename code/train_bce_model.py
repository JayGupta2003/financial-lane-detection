"""Train and save the BCE-only 3-channel model (seed 0, same data seeds as run_main.py) for the downstream stages.
The loss was preferred over BCE+Dice on VALIDATION IoU (see results/main.json), not on test."""
import os, json
import torch
import experiment as E, data as D

OUT = os.path.join(os.path.dirname(__file__), "..", "results"); T = D.TICKERS
tr = E.build_multi(T, "train", 100); va = E.build_multi(T, "val", 200); te = E.build_multi(T, "test", 300)
m, info = E.train(D.CH3, tr, va, 0, use_dice=False, log=lambda s: print("  [bce s0] " + s, flush=True))
r, P, Y = E.full_eval(m, te, D.CH3)
torch.save(m.state_dict(), os.path.join(OUT, "unet3_bce_seed0.pt"))
r["component"] = E.component_recall(P, Y)
r.update(info); json.dump(r, open(os.path.join(OUT, "unet3_bce_seed0.json"), "w"), indent=1)
print("unet3_bce seed0 retrain", r, flush=True)
