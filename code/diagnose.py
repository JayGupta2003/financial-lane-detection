"""Precision-gap diagnosis. Read-only analysis of saved seed-0 models.
1. per-ticker and per-time-of-day (thirds of the test split) validation vs test metrics
2. distribution drift of input channels between train / val / test
3. threshold analysis: IoU vs threshold on val and test; test IoU at the val-optimal threshold
4. where false positives sit: adjacent to a true region (boundary bleed) vs spurious
5. channel zeroing at test time (which channels carry the signal)
Output: results/diagnosis.json"""
import json, os, sys
import numpy as np, torch
from scipy import ndimage
import experiment as E, models as M, data as D

OUT = os.path.join(os.path.dirname(__file__), "..", "results"); T = D.TICKERS
CH = {"unet2": D.CH2, "unet3": D.CH3, "unet3_bce": D.CH3, "unet6": D.CH6}
log = lambda s: print(s, flush=True)


def load(tag):
    import modelio
    return modelio.load_model(tag)


def sm(P, Y, thr=0.5):
    return {k: round(v, 4) for k, v in E.seg_metrics(P, Y, thr).items()}


def main(tags):
    res = {}
    te = [E.build(t, "test", 300 + i)[0] for i, t in enumerate(T)]
    va = [E.build(t, "val", 200 + i)[0] for i, t in enumerate(T)]
    tr = [E.build(t, "train", 100 + i)[0] for i, t in enumerate(T)]
    # (2) drift of inputs: mean normalized channel value per split (all windows)
    drift = {}
    for name, ws in (("train", tr), ("val", va), ("test", te)):
        drift[name] = {f"ch{c}": float(np.mean([w.X[c].mean() for w in ws])) for c in range(6)}
        drift[name]["pos_pixel_rate"] = float(np.mean([w.M.mean() for w in ws]))
    res["input_drift"] = drift
    log(f"drift {json.dumps(drift)}")
    for tag in tags:
        m, ch = load(tag); R = res.setdefault(tag, {})
        # (1) per ticker, per time-of-day third
        R["per_ticker_val"] = {}; R["per_ticker_test"] = {}; R["test_by_time_third"] = {}
        Pt, Yt = [], []
        for i, t in enumerate(T):
            Pv, Yv = E.predict(m, [va[i]], ch); P, Y = E.predict(m, [te[i]], ch)
            R["per_ticker_val"][t] = sm(Pv, Yv); R["per_ticker_test"][t] = sm(P, Y)
            Pt.append(P); Yt.append(Y)
            n = len(P); thirds = np.array_split(np.arange(n), 3)
            for k, idx in enumerate(thirds):
                R["test_by_time_third"].setdefault(f"third{k+1}", []).append(
                    (E.seg_metrics(P[idx], Y[idx])["precision"], float(Y[idx].mean())))
        R["test_by_time_third"] = {k: dict(mean_precision=float(np.mean([a for a, _ in v])), mean_pos_rate=float(np.mean([b for _, b in v])))
                                   for k, v in R["test_by_time_third"].items()}
        Pv_all, Yv_all = E.predict(m, va, ch); P, Y = torch.cat(Pt), torch.cat(Yt)
        # (3) threshold analysis
        thr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        R["threshold"] = {str(t): dict(val=sm(Pv_all, Yv_all, t)["iou"], test=sm(P, Y, t)["iou"],
                                       test_precision=sm(P, Y, t)["precision"], test_recall=sm(P, Y, t)["recall"]) for t in thr}
        best = max(thr, key=lambda t: sm(Pv_all, Yv_all, t)["iou"])
        R["val_optimal_threshold"] = best; R["test_iou_at_val_optimal"] = sm(P, Y, best)["iou"]
        # (4) false positive location
        pr = (P > 0.5).numpy()[:, 0]; y = (Y > 0.5).numpy()[:, 0]
        near = ndimage.binary_dilation(y, structure=np.ones((1, 3, 11), bool))
        fp = pr & ~y
        R["false_positives"] = dict(total=int(fp.sum()), adjacent_to_true_region=float((fp & near).sum() / max(fp.sum(), 1)),
                                    spurious=float((fp & ~near).sum() / max(fp.sum(), 1)),
                                    windows_with_no_true_region_but_flagged=float(((~y.any((1, 2))) & pr.any((1, 2))).mean()))
        # (5) channel zeroing at test
        if len(ch) > 2:
            R["zero_channel_iou"] = {}
            for c in range(2, len(ch)):
                Pz = []
                for x, yy in E.iterate(te, ch, 256, False):
                    x = x.clone(); x[:, c] = 0
                    with torch.no_grad():
                        Pz.append(torch.sigmoid(m(x.to(E.dev))).cpu())
                R["zero_channel_iou"][f"ch{ch[c]}"] = sm(torch.cat(Pz), Y)["iou"]
        log(f"{tag} {json.dumps(R)[:600]}")
    name = "diagnosis.json" if set(tags) == {"unet2", "unet3", "unet6"} else "diagnosis_" + "_".join(tags) + ".json"
    json.dump(res, open(f"{OUT}/{name}", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["unet2", "unet3", "unet6"])
