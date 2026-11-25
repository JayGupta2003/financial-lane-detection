"""Extra figures: precision-recall curves, failure cases, decoy flag rates. -> paper/figs/"""
import json, os
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experiment as E, data as D, modelio

ROOT = os.path.join(os.path.dirname(__file__), ".."); RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "paper", "figs")
T = D.TICKERS
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = ["#2a6f97", "#e07a1f", "#6a994e", "#8d5a97"]


def pr_curve(P, Y, n=400):
    p = P.flatten().to(E.dev); y = Y.flatten().to(E.dev)
    o = torch.argsort(p, descending=True); y = y[o]
    tp = torch.cumsum(y, 0); k = torch.arange(1, len(y) + 1, device=E.dev)
    idx = torch.unique(torch.linspace(0, len(y) - 1, n * 20, dtype=torch.float64).long().to(E.dev))
    return (tp[idx] / tp[-1]).cpu().numpy(), (tp[idx] / k[idx]).cpu().numpy()


def figs(tags_labels):
    te = E.build_multi(T, "test", 300)
    fig, ax = plt.subplots(figsize=(3.4, 2.6)); store = {}
    for (tag, lab), c in zip(tags_labels, C):
        m, ch = modelio.load_model(tag); P, Y = E.predict(m, te, ch); store[tag] = (P, Y)
        r, p = pr_curve(P, Y); ax.plot(r, p, color=c, label=lab, lw=1.3)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    plt.savefig(os.path.join(FIG, "fig_pr.pdf"), bbox_inches="tight"); plt.close(); print("wrote fig_pr.pdf")
    # failure cases for the primary model
    tag = tags_labels[1][0]; m, ch = modelio.load_model(tag); P, Y = store[tag]
    wins = [(i, j) for i, w in enumerate(te) for j in range(len(w))]
    pr = (P[:, 0] > 0.5).float(); y = Y[:, 0]
    fp = (pr * (1 - y)).sum((1, 2)); fn = ((1 - pr) * y).sum((1, 2))
    picks = [("False positives", int(fp.argmax())), ("Missed spoof", int(fn.argmax()))]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.4), sharey=True)
    for r_, (lab, k) in enumerate(picks):
        i, j = wins[k]; x, _ = te[i].batch(np.array([j]), D.CH3)
        axs[r_, 0].imshow(np.maximum(x[0, 0], x[0, 1]), aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        axs[r_, 1].imshow(y[k].numpy(), aspect="auto", cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axs[r_, 2].imshow(P[k, 0].numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axs[r_, 0].set_ylabel(f"{lab}\n({T[i]})\nbook row")
    for a, t in zip(axs[0], ["Observed volume", "Injected mask", "Predicted probability"]): a.set_title(t, fontsize=7.5)
    for a in axs[1]: a.set_xlabel("Event index")
    plt.savefig(os.path.join(FIG, "fig_failures.pdf"), bbox_inches="tight"); plt.close(); print("wrote fig_failures.pdf")


def decoy_fig():
    g = json.load(open(os.path.join(RES, "grid.json")))
    names = [("m3b", "3 ch.\nno decoys"), ("m6b", "6 ch.\nno decoys"), ("m3b_hn", "3 ch.\nwith decoys"), ("m6b_hn", "6 ch.\nwith decoys")]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    fl = [[100 * r["hn"]["decoy_flag_rate"] for r in g[k]] for k, _ in names]
    axs[0].bar(range(4), [np.mean(v) for v in fl], yerr=[np.std(v, ddof=1) for v in fl], color=[C[0], C[0], C[1], C[1]], capsize=3)
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels([n for _, n in names], fontsize=6.5); axs[0].set_ylabel("Decoy cells flagged (%)")
    io = [[r["hn"]["iou"] for r in g[k]] for k, _ in names]
    axs[1].bar(range(4), [np.mean(v) for v in io], yerr=[np.std(v, ddof=1) for v in io], color=[C[0], C[0], C[1], C[1]], capsize=3)
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels([n for _, n in names], fontsize=6.5); axs[1].set_ylabel("IoU on decoy-containing test")
    plt.savefig(os.path.join(FIG, "fig_decoy.pdf"), bbox_inches="tight"); plt.close(); print("wrote fig_decoy.pdf")


if __name__ == "__main__":
    figs([("unet3", "3 ch, BCE+Dice"), ("m3b@0", "3 ch, BCE"), ("m6b@0", "6 ch, BCE")])
    decoy_fig()
