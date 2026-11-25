"""Main experiment: baselines + U-Net (2/3/6 ch) x 3 seeds + loss ablation. Saves results/main.json and models."""
import json, os, sys, time
import numpy as np, torch
import experiment as E, models as M, data as D

OUT = os.path.join(os.path.dirname(__file__), "..", "results"); os.makedirs(OUT, exist_ok=True)
T = D.TICKERS
log = lambda s: print(s, flush=True)
tr = E.build_multi(T, "train", 100); va = E.build_multi(T, "val", 200); te = E.build_multi(T, "test", 300)
log(f"windows train {sum(map(len,tr))} val {sum(map(len,va))} test {sum(map(len,te))}")
pos = lambda ws: float(np.mean([w.M[:, s:s + 256].any() for w in ws for s in w.starts]))
ppx = lambda ws: float(np.mean([w.M[:, s:s + 256].mean() for w in ws for s in w.starts]))
res = {"data": dict(train_windows=sum(map(len, tr)), val_windows=sum(map(len, va)), test_windows=sum(map(len, te)),
                    pos_windows_test=pos(te), pos_pixels_test=ppx(te))}


def classical(fn, wins, **kw):
    P, Y = [], []
    for x, y in E.iterate(wins, D.CH3, 256, False):
        P.append(fn(x, **kw)); Y.append(y)
    return torch.cat(P), torch.cat(Y)


def tune(fn, grid, key):
    best = max(grid, key=lambda g: E.seg_metrics(*classical(fn, va, **{key: g}))["iou"])
    P, Y = classical(fn, te, **{key: best})
    r = E.seg_metrics(P, Y); r["ap"] = E.average_precision(P, Y); r["param"] = float(best)
    return r


_prev = json.load(open(os.path.join(OUT,"main.json"))) if os.path.exists(os.path.join(OUT,"main.json")) else {}
SKIP = "cnn_classifier" in _prev
if SKIP:
    res.update({k: _prev[k] for k in ("volume_threshold","zscore","cnn_classifier")})
else:
  res["volume_threshold"] = tune(M.volume_threshold_mask, [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995], "thr")
  log(f"vol thr {res['volume_threshold']}")
  res["zscore"] = tune(M.zscore_mask, [1.5, 2, 2.5, 3, 4, 5], "z")
  log(f"zscore {res['zscore']}")

  # CNN classifier (window-level), mask = volume-threshold pixels inside windows flagged anomalous
  torch.manual_seed(0); rng = np.random.default_rng(0)
  clf = M.CNNClassifier(3).to(E.dev); opt = torch.optim.AdamW(clf.parameters(), 1e-3, weight_decay=1e-4)
  best, state = -1, None
  for ep in range(15):
      clf.train()
      for x, y in E.iterate(tr, D.CH3, 64, True, rng):
          lab = (y.flatten(1).max(1).values > 0).float()[:, None]
          loss = torch.nn.functional.binary_cross_entropy_with_logits(clf(x.to(E.dev)), lab.to(E.dev))
          opt.zero_grad(); loss.backward(); opt.step()
      clf.eval(); ok = n = 0
      with torch.no_grad():
          for x, y in E.iterate(va, D.CH3, 256, False):
              lab = (y.flatten(1).max(1).values > 0)
              ok += ((clf(x.to(E.dev)).cpu()[:, 0] > 0) == lab).sum().item(); n += len(lab)
      if ok / n > best: best, state = ok / n, {k: v.clone() for k, v in clf.state_dict().items()}
  clf.load_state_dict(state); clf.eval()
  vthr = res["volume_threshold"]["param"]
  P, Y = [], []
  with torch.no_grad():
      for x, y in E.iterate(te, D.CH3, 256, False):
          gate = (torch.sigmoid(clf(x.to(E.dev))).cpu() > 0.5).float()[:, :, None, None]
          P.append(M.volume_threshold_mask(x, vthr) * gate); Y.append(y)
  P, Y = torch.cat(P), torch.cat(Y)
  res["cnn_classifier"] = {**E.seg_metrics(P, Y), "ap": E.average_precision(P, Y), "val_window_acc": best}
  log(f"cnn clf {res['cnn_classifier']}")
res.update({k: v for k, v in _prev.items() if (k.startswith("unet") and k != "unet6") or k == "component_unet3"})
json.dump(res, open(os.path.join(OUT, "main.json"), "w"), indent=1)

configs ={"unet2": (D.CH2, True, True), "unet3": (D.CH3, True, True), "unet6": (D.CH6, True, True),
           "unet3_bce": (D.CH3, False, True), "unet3_dice": (D.CH3, True, False)}
for name, (ch, dice, bce) in configs.items():
    if name in res and len(res[name]) == 3:      # finished in an earlier run (resume)
        log(f"skip {name} (already done)"); continue
    runs = []
    for seed in (0, 1, 2):
        t0 = time.time()
        m, info = E.train(ch, tr, va, seed, use_dice=dice, use_bce=bce, log=lambda s: log(f'  [{name} s{seed}] {s}'))
        r, P, Y = E.full_eval(m, te, ch)
        r.update(info, train_s=time.time() - t0, params=M.count_params(m))
        runs.append(r); log(f"{name} seed{seed} {r}")
        if seed == 0 and name in ("unet3", "unet6", "unet2"):
            torch.save(m.state_dict(), os.path.join(OUT, f"{name}_seed0.pt"))
            if name == "unet3":
                res["component_unet3"] = E.component_recall(P[:600], Y[:600])
    res[name] = runs
    json.dump(res, open(os.path.join(OUT, "main.json"), "w"), indent=1)
log("DONE")
