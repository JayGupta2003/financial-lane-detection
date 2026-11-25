"""Training queue for the extended study. Saves every model (results/models/<name>_s<seed>.pt) and appends metrics to
results/grid.json. Resumable: finished (name, seed) pairs are skipped.
Usage: python train_grid.py [config ...]   (default: all, in priority order)
Each model is evaluated on the standard test set and on a test set that also contains decoys (hard negatives)."""
import json, os, sys, time
import torch
import experiment as E, models as M, data as D

OUT = os.path.join(os.path.dirname(__file__), "..", "results"); os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
T = D.TICKERS
# name: (channels, loss, trained_with_decoys)
CONF = {
    "m3b": (D.CH3, "bce", False), "m6b": (D.CH6, "bce", False), "m2b": (D.CH2, "bce", False),
    "m3b_hn": (D.CH3, "bce", True), "m6b_hn": (D.CH6, "bce", True),
    "m3_dpo": (D.CH3, "bce_dice_posonly", False), "m3_bdb": (D.CH3, "bce_dice_batch", False),
    "m3_pw": (D.CH3, "bce_posw", False), "m3_focal": (D.CH3, "focal", False),
}
DEC = 3.0  # decoys per 1000 events
log = lambda s: print(s, flush=True)

todo = sys.argv[1:] or list(CONF)
gp = os.path.join(OUT, "grid.json")
grid = json.load(open(gp)) if os.path.exists(gp) else {}
tr = E.build_multi(T, "train", 100); va = E.build_multi(T, "val", 200); te = E.build_multi(T, "test", 300)
tr_h = E.build_multi(T, "train", 100, decoys=DEC); va_h = E.build_multi(T, "val", 200, decoys=DEC)
te_h = E.build_multi(T, "test", 300, decoys=DEC)
log("data ready")
for name in todo:
    ch, loss, hn = CONF[name]
    for seed in (0, 1, 2):
        if any(r["seed"] == seed for r in grid.get(name, [])):
            log(f"skip {name} s{seed}"); continue
        t0 = time.time()
        m, info = E.train(ch, tr_h if hn else tr, va_h if hn else va, seed, loss=loss,
                          log=lambda s: log(f"  [{name} s{seed}] {s}"))
        std = E.full_eval(m, te, ch)[0]
        rec = dict(seed=seed, std=std, hn=E.decoy_report(m, te_h, ch), train_s=time.time() - t0, **info)
        torch.save(m.state_dict(), os.path.join(OUT, "models", f"{name}_s{seed}.pt"))
        grid.setdefault(name, []).append(rec)
        json.dump(grid, open(gp, "w"), indent=1)
        log(f"{name} seed{seed} std_iou={std['iou']:.3f} hn={rec['hn']}")
log("GRID DONE")
