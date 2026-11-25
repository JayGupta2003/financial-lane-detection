"""Generate LaTeX tables + number macros for the paper from results/*.json.
Output: paper/gen/*.tex and paper/gen/numbers.tex. Missing inputs produce a visible TBD marker, never invented values."""
import json, os, glob
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results"); GEN = os.path.join(ROOT, "paper", "gen"); os.makedirs(GEN, exist_ok=True)
T = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]


def J(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ms(runs, k, d=3):
    v = np.array([r[k] for r in runs]); return f"{v.mean():.{d}f} $\\pm$ {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def f(x, d=3):
    return "--" if x is None else f"{x:.{d}f}"


def write(name, s):
    open(os.path.join(GEN, name), "w").write(s)


def table(cols, rows, align=None):
    align = align or "l" + "r" * (len(cols) - 1)
    s = f"\\begin{{tabular}}{{{align}}}\n\\toprule\n" + " & ".join(cols) + " \\\\\n\\midrule\n"
    for r in rows:
        s += " & ".join(str(c) for c in r) + " \\\\\n"
    return s + "\\bottomrule\n\\end{tabular}\n"


TBD = "\\textbf{TBD}"
macros = {}
main = J("main.json")
if main:
    mm = lambda r, k: f"{r[k]:.3f}"
    rows = []
    for name, key in (("Volume threshold", "volume_threshold"), ("Local $z$-score", "zscore"), ("CNN classifier (gated mask)", "cnn_classifier")):
        r = main[key]; rows.append([name, mm(r, "iou"), mm(r, "dice"), mm(r, "precision"), mm(r, "recall"), mm(r, "ap")])
    for name, key in (("U-Net, 2 ch.\ (BCE+Dice)", "unet2"), ("U-Net, 3 ch.\ (BCE+Dice)", "unet3"), ("U-Net, 3 ch.\ (BCE only)", "unet3_bce"), ("U-Net, 6 ch.\ (BCE+Dice)", "unet6")):
        if key in main:
            runs = main[key]
            rows.append([name] + [ms(runs, k) for k in ("iou", "dice", "precision", "recall", "ap")])
            macros[f"{key}IoU"] = f"{np.mean([r['iou'] for r in runs]):.3f}"
            macros[f"{key}IoUsd"] = f"{np.std([r['iou'] for r in runs], ddof=1):.3f}"
            macros[f"{key}Prec"] = f"{np.mean([r['precision'] for r in runs]):.3f}"
            macros[f"{key}Rec"] = f"{np.mean([r['recall'] for r in runs]):.3f}"
            macros[f"{key}AP"] = f"{np.mean([r['ap'] for r in runs]):.3f}"
            macros[f"{key}Params"] = f"{runs[0]['params']/1e6:.2f}"
        else:
            rows.append([name] + [TBD] * 5)
    write("mainresults.tex", table(["Model", "IoU", "Dice", "Precision", "Recall", "AP"], rows))
    macros["volIoU"] = f"{main['volume_threshold']['iou']:.3f}"; macros["zIoU"] = f"{main['zscore']['iou']:.3f}"
    macros["cnnIoU"] = f"{main['cnn_classifier']['iou']:.3f}"; macros["volThr"] = f"{main['volume_threshold']['param']:.2f}"
    d = main["data"]
    macros["trainWin"] = f"{d['train_windows']:,}".replace(",", "{,}"); macros["valWin"] = f"{d['val_windows']:,}".replace(",", "{,}")
    macros["testWin"] = f"{d['test_windows']:,}".replace(",", "{,}")
    macros["posWinTest"] = f"{100*d['pos_windows_test']:.0f}"; macros["posPxTest"] = f"{100*d['pos_pixels_test']:.1f}"
    # ablation: loss + channels (+ time columns)
    rows = []
    for name, key in (("BCE + Dice (3 ch)", "unet3"), ("BCE only (3 ch)", "unet3_bce"), ("Dice only (3 ch)", "unet3_dice"),
                      ("2 channels (BCE + Dice)", "unet2"), ("6 channels (BCE + Dice)", "unet6")):
        rows.append([name] + ([ms(main[key], k) for k in ("iou", "precision", "recall", "ap")] if key in main else [TBD] * 4))
    tc = J("timecols.json")
    rows.append(["Time-based columns, 1\\,s bins (3 ch)"] + ([ms(tc, k) for k in ("iou", "precision", "recall", "ap")] if tc else [TBD] * 4))
    write("ablation.tex", table(["Variant", "IoU", "Precision", "Recall", "AP"], rows))
    if "component_unet3" in main:
        c = main["component_unet3"]
        macros["compRecall"] = f"{100*c['recall']:.1f}"; macros["compN"] = f"{c['n_gt']:,}".replace(",", "{,}")
        macros["compErrL"] = f"{c['med_err_levels']:.1f}"; macros["compErrC"] = f"{c['med_err_cols']:.1f}"

# dataset table from message-file line counts
rows = []
for t in T:
    mp = glob.glob(os.path.join(ROOT, "data", "lobster", t, "*message*.csv"))
    n = sum(1 for _ in open(mp[0])) if mp else 0
    rows.append([t, f"{n:,}".replace(",", "{,}"), f"{n/23400:.1f}"])
write("dataset.tex", table(["Ticker", "Events", "Events/s"], rows))

sw = J("sweeps.json")
for tag in ("unet3", "unet3_bce", "unet6"):
    if sw and tag in sw:
        R = sw[tag]
        write(f"magnitude_{tag}.tex", table(["Size ratio", "IoU", "Precision", "Recall", "AP"],
              [[f"{r}$\\times$"] + [f(R["size"][r][k]) for k in ("iou", "precision", "recall", "ap")] for r in R["size"]]))
        write(f"duration_{tag}.tex", table(["Residence (events)", "IoU", "Precision", "Recall", "AP"],
              [[d] + [f(R["duration"][d][k]) for k in ("iou", "precision", "recall", "ap")] for d in R["duration"]]))
        write(f"regime_{tag}.tex", table(["Volatility quartile", "Windows", "IoU", "Precision", "Recall"],
              [[k, v["n_windows"]] + [f(v[x]) for x in ("iou", "precision", "recall")] for k, v in R["regime"].items()]))
        L = R["latency_ms"]
        macros[f"{tag}LatMean"] = f"{L['mean']:.2f}"; macros[f"{tag}LatP"] = f"{L['p95']:.2f}"
for tag, hk in (("unet3", "heldout"), ("unet3_bce", "heldout_bce")):
    if sw and tag in sw:
        rows = [[t] + [f(sw[tag]["per_asset"][t][k]) for k in ("iou", "precision", "recall")] for t in T]
        h = sw.get(hk)
        if h:
            u = h["INTC_unseen"]; rows.append(["INTC (held out of training)"] + [f(u[k]) for k in ("iou", "precision", "recall")])
            macros[f"{tag}heldoutIoU".replace("_", "")] = f"{u['iou']:.3f}"; macros[f"{tag}heldoutPrec".replace("_", "")] = f"{u['precision']:.3f}"
        write(f"crossasset_{tag}.tex", table(["Asset", "IoU", "Precision", "Recall"], rows))
if sw:
    write("latency.tex", table(["Model", "Params (M)", "Mean (ms)", "p95 (ms)"],
          [[tag.replace("_", "\_"), f"{sw[tag]['latency_ms']['params']/1e6:.2f}", f"{sw[tag]['latency_ms']['mean']:.2f}", f"{sw[tag]['latency_ms']['p95']:.2f}"]
           for tag in ("unet3", "unet3_bce", "unet6") if tag in sw]))

# OOD
rows = []
for tag in ("unet3", "unet3_bce", "unet6"):
    o = J(f"ood_{tag}.json")
    if o:
        for fam, v in o.items():
            rows.append([tag.replace("_", "\_"), fam.replace("_", " "), f(v["iou"]), f(v["precision"]), f(v["recall"]), f(v["volume_threshold_iou"])])
write("ood.tex", table(["Model", "Family", "IoU", "Precision", "Recall", "Vol.-thr. IoU"], rows) if rows else TBD)

# execution
for tag in ("unet3", "unet3_bce", "unet6"):
    ex = J(f"execution_{tag}.json")
    if not ex:
        continue
    rows = []
    for dens, r in ex.items():
        for arm in ("unfiltered", "filtered", "oracle", "clean"):
            a = r[arm]; rows.append([dens if arm == "unfiltered" else "", arm, f"{a['IS']:.3f}", f"{100*a['fill']:.2f}", f"{a['sends']:.1f}"])
        d = r["delta_IS_bps"]
        rows.append(["", "$\\Delta$IS (filt.$-$unfilt.)", f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]", "", f"$p$={d['p_two_sided']:.3f}"])
    write(f"execution_{tag}.tex", table(["Spoofs/1000 ev.", "Arm", "IS (bps)", "First-try fill (\\%)", "Sends"], rows, "llrrr"))
    k = "3.0" if "3.0" in ex else list(ex)[0]
    d = ex[k]["delta_IS_bps"]
    macros[f"{tag}dIS"] = f"{d['mean']:+.4f}"; macros[f"{tag}dISlo"] = f"{d['ci95'][0]:+.4f}"; macros[f"{tag}dIShi"] = f"{d['ci95'][1]:+.4f}"
    macros[f"{tag}dISp"] = f"{d['p_two_sided']:.3f}"; macros[f"{tag}ISunf"] = f"{ex[k]['unfiltered']['IS']:.3f}"
    macros[f"{tag}ISfil"] = f"{ex[k]['filtered']['IS']:.3f}"

# diagnosis
dg = J("diagnosis.json")
dg_b = J("diagnosis_unet3_bce.json")
if dg and "unet3" in dg:
    R = dg["unet3"]
    write("diag_ticker.tex", table(["Ticker", "Val. precision", "Test precision", "Val. recall", "Test recall"],
          [[t, f(R["per_ticker_val"][t]["precision"]), f(R["per_ticker_test"][t]["precision"]),
            f(R["per_ticker_val"][t]["recall"]), f(R["per_ticker_test"][t]["recall"])] for t in T]))
    write("diag_threshold.tex", table(["Threshold", "Val. IoU", "Test IoU", "Test precision", "Test recall"],
          [[k, f(v["val"]), f(v["test"]), f(v["test_precision"]), f(v["test_recall"])] for k, v in R["threshold"].items()]))
    fp = R["false_positives"]; macros["fpSpurious"] = f"{100*fp['spurious']:.0f}"
    t3 = R["test_by_time_third"]
    macros["precThirdA"] = f"{t3['third1']['mean_precision']:.2f}"; macros["precThirdB"] = f"{t3['third2']['mean_precision']:.2f}"
    macros["precThirdC"] = f"{t3['third3']['mean_precision']:.2f}"
    if "unet6" in dg and "zero_channel_iou" in dg["unet6"]:
        z = dg["unet6"]["zero_channel_iou"]
        write("diag_zero.tex", table(["Channel zeroed at test", "IoU"], [[k, f(v)] for k, v in z.items()]))

if dg and dg_b and "unet3" in dg and "unet3_bce" in dg_b:
    A, B = dg["unet3"], dg_b["unet3_bce"]
    write("diag_compare.tex", table(["Ticker", "BCE+Dice val.", "BCE+Dice test", "BCE-only val.", "BCE-only test"],
          [[t, f(A["per_ticker_val"][t]["precision"]), f(A["per_ticker_test"][t]["precision"]),
            f(B["per_ticker_val"][t]["precision"]), f(B["per_ticker_test"][t]["precision"])] for t in T]))
    macros["bceFpSpurious"] = f"{100*B['false_positives']['spurious']:.0f}"
    macros["bceThirdC"] = f"{B['test_by_time_third']['third3']['mean_precision']:.2f}"

# real-data check + FI-2010
rows = []
for tag in ("unet2", "unet3", "unet3_bce", "unet6"):
    r = J(f"real_{tag}.json")
    if r:
        for t, v in r["per_asset"].items():
            rows.append([tag.replace("_", "\_"), t, f"{v['per_hour']:.0f}", f"{100*v['window_rate']:.1f}", f"{100*v['pixel_rate']:.3f}",
                         f(v["pulled_share"], 2)])
write("real.tex", table(["Model", "Ticker", "Flags/hour", "Windows flagged (\\%)", "Pixels flagged (\\%)", "Pulled share"], rows) if rows else TBD)
fi = J("fi2010_unet2.json")
if fi:
    ref = fi["lobster_clean_test"]
    rows = [["LOBSTER clean test, " + t, f"{100*v['pixel_rate']:.3f}", f"{100*v['window_rate']:.1f}"] for t, v in ref.items()]
    rows += [["FI-2010 " + k.replace("Test_Dst_NoAuction_DecPre_", "").replace(".txt", "").replace("_", "\\_"), f"{100*v['pixel_rate']:.2f}", f"{100*v['window_rate']:.1f}"]
             for k, v in fi["files"].items()]
    write("fi2010.tex", table(["Data", "Pixels flagged (\\%)", "Windows flagged (\\%)"], rows))

W = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
def mname(k):
    return "".join(W.get(c, c) for c in k.replace("_", ""))
write("numbers.tex", "".join("\\newcommand{\\" + mname(k) + "}{" + str(v) + "}\n" for k, v in macros.items()))
print(f"wrote {len(os.listdir(GEN))} files; macros: {len(macros)}")
