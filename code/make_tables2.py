"""Seed-aggregated tables for the extended study -> paper/gen2/*.tex and numbers2.tex (also prints a summary)."""
import json, os, glob
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), ".."); RES = os.path.join(ROOT, "results"); GEN = os.path.join(ROOT, "paper", "gen2")
os.makedirs(GEN, exist_ok=True)
T = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
macros = {}


def J(n):
    p = os.path.join(RES, n); return json.load(open(p)) if os.path.exists(p) else None


def pm(v, d=3):
    v = np.asarray(v, float); return f"{v.mean():.{d}f} $\\pm$ {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def mean(v): return float(np.mean(v))


def table(cols, rows, align=None):
    align = align or "l" + "r" * (len(cols) - 1)
    return f"\\begin{{tabular}}{{{align}}}\n\\toprule\n" + " & ".join(cols) + " \\\\\n\\midrule\n" + "".join(" & ".join(map(str, r)) + " \\\\\n" for r in rows) + "\\bottomrule\n\\end{tabular}\n"


def write(n, s): open(os.path.join(GEN, n), "w").write(s)


W = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
mname = lambda k: "".join(W.get(c, c) for c in k.replace("_", "").replace("@", "").replace(".", "p"))
main, grid = J("main.json"), J("grid.json")
seeds = (0, 1, 2)

# ---------- main + decoy + loss ----------
met = ("iou", "dice", "precision", "recall", "ap")
rows = []
for nm, key in (("Volume threshold", "volume_threshold"), ("Local $z$-score", "zscore"), ("CNN classifier (gated)", "cnn_classifier")):
    r = main[key]; rows.append([nm] + [f"{r[k]:.3f}" for k in met])
for nm, key in (("U-Net 2 ch., BCE", "m2b"), ("U-Net 3 ch., BCE", "m3b"), ("U-Net 6 ch., BCE", "m6b")):
    rows.append([nm] + [pm([r["std"][k] for r in grid[key]]) for k in met])
for nm, key in (("U-Net 2 ch., BCE+Dice", "unet2"), ("U-Net 3 ch., BCE+Dice", "unet3"), ("U-Net 6 ch., BCE+Dice", "unet6")):
    rows.append([nm] + [pm([r[k] for r in main[key]]) for k in met])
write("main.tex", table(["Model", "IoU", "Dice", "Precision", "Recall", "AP"], rows))
for key in ("m2b", "m3b", "m6b"):
    for k in ("iou", "precision", "recall"):
        macros[f"{key}{k}"] = f"{mean([r['std'][k] for r in grid[key]]):.3f}"
    macros[f"{key}iousd"] = f"{np.std([r['std']['iou'] for r in grid[key]], ddof=1):.3f}"
    macros[f"{key}val"] = f"{mean([r['best_val_iou'] for r in grid[key]]):.3f}"

# decoy table
dex = J("decoys_existing.json")
rows = []
for nm, key in (("3 ch., BCE", "m3b"), ("6 ch., BCE", "m6b"), ("3 ch., BCE, trained with decoys", "m3b_hn"), ("6 ch., BCE, trained with decoys", "m6b_hn")):
    rows.append([nm, pm([r["std"]["iou"] for r in grid[key]])] + [pm([r["hn"][k] for r in grid[key]]) for k in ("iou", "precision", "recall")]
                + [pm([100 * r["hn"]["decoy_flag_rate"] for r in grid[key]], 1)])
    macros[f"{key}hniou"] = f"{mean([r['hn']['iou'] for r in grid[key]]):.3f}"
    macros[f"{key}hnprec"] = f"{mean([r['hn']['precision'] for r in grid[key]]):.3f}"
    macros[f"{key}hnrec"] = f"{mean([r['hn']['recall'] for r in grid[key]]):.3f}"
    macros[f"{key}decoyflag"] = f"{100*mean([r['hn']['decoy_flag_rate'] for r in grid[key]]):.0f}"
    macros[f"{key}decoyflagone"] = f"{100*mean([r['hn']['decoy_flag_rate'] for r in grid[key]]):.1f}"
    macros[f"{key}stdiou"] = f"{mean([r['std']['iou'] for r in grid[key]]):.3f}"
for nm, key in (("3 ch., BCE+Dice (seed 0)", "unet3"), ("6 ch., BCE+Dice (seed 0)", "unet6")):
    r = dex[key]; rows.insert(2 if "3" in nm else 3, [nm, "--", f"{r['iou']:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{100*r['decoy_flag_rate']:.1f}"])
write("decoy.tex", table(["Model", "IoU (std. test)", "IoU (decoy test)", "Precision", "Recall", "Decoy cells flagged (\\%)"], rows))

# loss variants
loss_rows = []
names = [("BCE only", "m3b", grid), ("BCE + Dice (per window)", "unet3", main), ("Dice only", "unet3_dice", main),
         ("BCE + Dice, positive windows only", "m3_dpo", grid), ("BCE + Dice (batch-level)", "m3_bdb", grid),
         ("BCE, positive weight 10", "m3_pw", grid), ("Focal loss ($\\gamma{=}2$)", "m3_focal", grid)]
for nm, key, src in names:
    runs = src[key]
    if src is grid:
        iou = [r["std"]["iou"] for r in runs]; pr = [r["std"]["precision"] for r in runs]; rc = [r["std"]["recall"] for r in runs]; vl = [r["best_val_iou"] for r in runs]
    else:
        iou = [r["iou"] for r in runs]; pr = [r["precision"] for r in runs]; rc = [r["recall"] for r in runs]; vl = [r["best_val_iou"] for r in runs]
    loss_rows.append([nm, pm(vl), pm(iou), pm(pr), pm(rc)]); macros[f"loss{mname(key)}iou"] = f"{mean(iou):.3f}"
write("loss.tex", table(["Loss (3 ch.)", "Val. IoU", "Test IoU", "Test precision", "Test recall"], loss_rows))

# per-ticker test precision by loss (seed 0)
rows = []
files = [("BCE + Dice", "diagnosis.json", "unet3"), ("BCE only", "diagnosis_m3b@0.json", "m3b@0"), ("Dice, pos. windows", "diagnosis_m3_dpo@0.json", "m3_dpo@0"),
         ("Batch Dice", "diagnosis_m3_bdb@0.json", "m3_bdb@0"), ("Pos. weight", "diagnosis_m3_pw@0.json", "m3_pw@0"), ("Focal", "diagnosis_m3_focal@0.json", "m3_focal@0")]
for nm, fn, k in files:
    d = J(fn)
    if d and k in d:
        pt = d[k]["per_ticker_test"]; rows.append([nm] + [f"{pt[t]['precision']:.2f}" for t in T])
write("loss_ticker.tex", table(["Loss (seed 0)"] + T, rows))

# ---------- sweeps (seed aggregated) ----------
sw = J("sweeps.json")
def sweep_table(kind, keys, first, name):
    rows = []
    for k in keys:
        row = [k if kind != "size" else f"{k}$\\times$"]
        for m in ("m3b", "m6b"):
            for met_ in ("iou", "recall"):
                row.append(pm([sw[f"{m}@{s}"][kind][str(k)][met_] for s in seeds], 3))
        rows.append(row)
    write(name, table([first, "3 ch.\\ IoU", "3 ch.\\ recall", "6 ch.\\ IoU", "6 ch.\\ recall"], rows))
sweep_table("size", [2, 3, 5, 10, 20], "Size ratio", "size.tex")
sweep_table("duration", [5, 10, 20, 50, 100], "Residence (events)", "duration.tex")
rows = []
for q in sw["m3b@0"]["regime"]:
    rows.append([q] + [pm([sw[f"{m}@{s}"]["regime"][q][met_] for s in seeds]) for m in ("m3b", "m6b") for met_ in ("iou", "precision")])
write("regime.tex", table(["Volatility quartile", "3 ch.\\ IoU", "3 ch.\\ prec.", "6 ch.\\ IoU", "6 ch.\\ prec."], rows))
rows = [[t] + [pm([sw[f"{m}@{s}"]["per_asset"][t][met_] for s in seeds]) for m in ("m3b", "m6b") for met_ in ("iou", "precision")] for t in T]
write("asset.tex", table(["Asset", "3 ch.\\ IoU", "3 ch.\\ prec.", "6 ch.\\ IoU", "6 ch.\\ prec."], rows))
macros["heldoutbceiou"] = f"{sw['heldout_bce']['INTC_unseen']['iou']:.3f}"; macros["heldoutbceprec"] = f"{sw['heldout_bce']['INTC_unseen']['precision']:.3f}"
macros["heldoutdiceiou"] = f"{sw['heldout']['INTC_unseen']['iou']:.3f}"
lat = J("latency.json")
macros["latthreemean"] = f"{lat['m3b@0']['mean']:.2f}"; macros["latsixmean"] = f"{lat['m6b@0']['mean']:.2f}"
macros["latthreep"] = f"{lat['m3b@0']['p95']:.2f}"; macros["latsixp"] = f"{lat['m6b@0']['p95']:.2f}"

# ---------- OOD ----------
fams = list(J("ood_m3b@0.json").keys()); rows = []
vt = {f: J("ood_m3b@0.json")[f]["volume_threshold_iou"] for f in fams}
for f in fams:
    row = [f.replace("_", " ")]
    for m in ("m3b", "m6b", "m3b_hn", "m6b_hn"):
        row.append(pm([J(f"ood_{m}@{s}.json")[f]["iou"] for s in seeds]))
    row.append(f"{vt[f]:.3f}"); rows.append(row)
    for m in ("m3b", "m6b", "m3b_hn", "m6b_hn"):
        macros[f"ood{mname(f)}{mname(m)}"] = f"{mean([J(f'ood_{m}@{s}.json')[f]['iou'] for s in seeds]):.3f}"
write("ood.tex", table(["Family", "3 ch.", "6 ch.", "3 ch. +decoys", "6 ch. +decoys", "Vol.-thr."], rows))

# ---------- real-book flags per hour ----------
rows = []
for m, lab in (("m3b", "3 ch."), ("m6b", "6 ch."), ("m3b_hn", "3 ch. +decoys"), ("m6b_hn", "6 ch. +decoys")):
    rows.append([lab] + [f"{mean([J(f'real_{m}@{s}.json')['per_asset'][t]['per_hour'] for s in seeds]):.0f}" for t in T])
write("real.tex", table(["Model (mean of 3 seeds)"] + T, rows))

# ---------- execution ----------
rows = []
for m, lab in (("m3b", "3 ch."), ("m3b_hn", "3 ch. +decoys")):
    ex = [J(f"execution_{m}@{s}.json") for s in seeds]
    for dens in ex[0]:
        d = [e[dens]["delta_IS_bps"]["mean"] for e in ex]; isu = [e[dens]["unfiltered"]["IS"] for e in ex]
        fu = [e[dens]["unfiltered"]["fill"] for e in ex]; ff = [e[dens]["filtered"]["fill"] for e in ex]
        rows.append([lab, dens, f"{mean(isu):.3f}", f"{mean(d):+.4f}", f"{100*mean(d)/mean(isu):+.2f}", f"{100*mean(fu):.2f} / {100*mean(ff):.2f}"])
        macros[f"exec{mname(m)}d{mname(dens)}"] = f"{mean(d):+.4f}"; macros[f"exec{mname(m)}p{mname(dens)}"] = f"{abs(100*mean(d)/mean(isu)):.2f}"
write("exec.tex", table(["Model", "Spoofs/1000", "IS unfilt.\\ (bps)", "$\\Delta$IS (bps)", "$\\Delta$IS (\\%)", "1st-try fill unf./filt.\\ (\\%)"], rows, "llrrrr"))
sens = J("execution_sens_unet3_bce.json"); rows = []
for ns in (25, 50, 100):
    rows.append([ns] + [f"{sens[f'nslice{ns}_frac{f}']['delta_IS_bps']['mean']:+.4f}" for f in (0.1, 0.2, 0.4, 0.8, 1.6)])
write("exec_sens.tex", table(["Slices", "0.1$\\times$", "0.2$\\times$", "0.4$\\times$", "0.8$\\times$", "1.6$\\times$"], rows))
macros["sensmax"] = f"{max(abs(v['delta_IS_bps']['mean']) for v in sens.values()):.3f}"
macros["sensmaxpct"] = f"{max(100*abs(v['delta_IS_bps']['mean'])/abs(v['unfiltered']['IS']) for v in sens.values() if abs(v['unfiltered']['IS'])>0.5):.2f}"

# ---------- weak labels ----------
wl = J("weak_labels.json"); rows = []
for tag, lab in (("unet3_bce", "3 ch. (BCE)"), ("m3b_hn@0", "3 ch. +decoys"), ("m6b_hn@0", "6 ch. +decoys")):
    r = wl[tag]["3.0"]
    rows.append([lab] + [f"{r[t]['model']['pred_overlapping_weak']:.2f} / {r[t]['model']['weak_overlapping_pred']:.2f}" for t in ("AAPL", "AMZN", "GOOG")])
r = wl["unet3_bce"]["3.0"]
rows.append(["Volume threshold"] + [f"{r[t]['volume_threshold']['pred_overlapping_weak']:.2f} / {r[t]['volume_threshold']['weak_overlapping_pred']:.2f}" for t in ("AAPL", "AMZN", "GOOG")])
rows.append(["Random flags"] + [f"{r[t]['random']['pred_overlapping_weak']:.2f} / {r[t]['random']['weak_overlapping_pred']:.2f}" for t in ("AAPL", "AMZN", "GOOG")])
rows.append(["Candidates in test period"] + [str(r[t]["n_candidates_test"]) for t in ("AAPL", "AMZN", "GOOG")])
write("weak.tex", table(["Detector", "AAPL", "AMZN", "GOOG"], rows))

# ---------- zero-channel (6 ch, BCE, mean over seeds) ----------
zc = [J(f"diagnosis_m6b@{k}.json")[f"m6b@{k}"]["zero_channel_iou"] for k in seeds]
names_ = {"ch2": "Cancellation rate", "ch3": "Arrival rate", "ch4": "Execution rate", "ch5": "Order age"}
write("zero.tex", table(["Channel zeroed at test", "IoU"], [[names_[c], pm([z[c] for z in zc])] for c in zc[0]]))
for c in zc[0]:
    macros["zero" + c] = f"{mean([z[c] for z in zc]):.3f}"

# ---------- FI-2010 (2 ch., BCE, seed 0) ----------
fi = J("fi2010_m2b@0.json")
if fi:
    rows = [["LOBSTER " + t_, f"{100*v['pixel_rate']:.3f}", f"{100*v['window_rate']:.1f}"] for t_, v in fi["lobster_clean_test"].items()]
    rows += [["FI-2010 " + k.replace("Test_Dst_NoAuction_DecPre_", "").replace(".txt", "").replace("_", "\_"), f"{100*v['pixel_rate']:.2f}", f"{100*v['window_rate']:.1f}"] for k, v in fi["files"].items()]
    write("fi2010.tex", table(["Data", "Pixels flagged (\%)", "Windows flagged (\%)"], rows))
    macros["fipix"] = f"{100*min(v['pixel_rate'] for v in fi['files'].values()):.1f}--{100*max(v['pixel_rate'] for v in fi['files'].values()):.1f}"
    macros["lobpix"] = f"{100*min(v['pixel_rate'] for v in fi['lobster_clean_test'].values()):.2f}--{100*max(v['pixel_rate'] for v in fi['lobster_clean_test'].values()):.2f}"

# ---------- per-seed appendix ----------
rows = []
for nm, key, src in (("U-Net 2 ch., BCE+Dice", "unet2", main), ("U-Net 3 ch., BCE+Dice", "unet3", main), ("U-Net 6 ch., BCE+Dice", "unet6", main), ("U-Net 3 ch., Dice only", "unet3_dice", main),
                     ("U-Net 2 ch., BCE", "m2b", grid), ("U-Net 3 ch., BCE", "m3b", grid), ("U-Net 6 ch., BCE", "m6b", grid), ("3 ch., BCE, decoys", "m3b_hn", grid), ("6 ch., BCE, decoys", "m6b_hn", grid)):
    runs = src[key]
    get = (lambda r, k: r["std"][k]) if src is grid else (lambda r, k: r[k])
    rows.append([nm] + [f"{get(r, 'iou'):.3f}" for r in runs])
write("perseed.tex", table(["Configuration", "Seed 0", "Seed 1", "Seed 2"], rows))

write("numbers2.tex", "".join("\\newcommand{\\" + mname(k) + "}{" + str(v) + "}\n" for k, v in macros.items()))
print("macros:", len(macros)); print({k: v for k, v in macros.items() if not k.startswith("ood") and not k.startswith("exec")})
print({k: v for k, v in macros.items() if k.startswith("ood") or k.startswith("exec")})
