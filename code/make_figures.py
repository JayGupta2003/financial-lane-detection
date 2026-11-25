"""Paper figures -> paper/figs/*.pdf. Skips a figure if its inputs are missing."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "paper", "figs"); os.makedirs(FIG, exist_ok=True)
T = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C1, C2, C3 = "#2a6f97", "#e07a1f", "#6a994e"


def J(n):
    p = os.path.join(RES, n); return json.load(open(p)) if os.path.exists(p) else None


def save(name):
    plt.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close(); print("wrote", name)


def pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 1.35)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 20)
    labels = ["Event\nstream", "LOB\nreconstruction", "Level-time\ntensor", "U-Net", "Anomaly\nmask", "Filtered\nLOB", "Execution\nalgorithm"]
    w, gap = 12.2, 2.1
    for i, s in enumerate(labels):
        x = i * (w + gap) + 0.5
        ax.add_patch(FancyBboxPatch((x, 4), w, 12, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#e8f1f8" if i != 3 else "#fbe5cf", ec=C1 if i != 3 else C2, lw=1))
        ax.text(x + w / 2, 10, s, ha="center", va="center", fontsize=7)
        if i < len(labels) - 1:
            ax.annotate("", xy=(x + w + gap - 0.3, 10), xytext=(x + w + 0.35, 10), arrowprops=dict(arrowstyle="->", lw=0.9, color="#444"))
    save("fig_pipeline.pdf")


def window_example():
    import experiment as E, models as M, data as D, torch
    ws, sp = E.build("AAPL", "test", 300)
    pos = [i for i, s in enumerate(ws.starts) if ws.M[:, s:s + 256].sum() > 120]
    k = pos[len(pos) // 2]; s0 = ws.starts[k]
    X = ws.X[:, :, s0:s0 + 256]; Y = ws.M[:, s0:s0 + 256]
    # representation figure (6 channels)
    names = ["Ask volume", "Bid volume", "Cancel rate", "Arrival rate", "Execution rate", "Order age"]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.3), sharex=True, sharey=True)
    for a, c, n in zip(axs.flat, range(6), names):
        im = a.imshow(X[c], aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest"); a.set_title(n, fontsize=8)
        a.axhline(9.5, color="w", lw=0.4, ls=":")
    for a in axs[:, 0]: a.set_ylabel("Book row")
    for a in axs[1]: a.set_xlabel("Event index in window")
    fig.colorbar(im, ax=axs, shrink=0.8, label="normalised value"); save("fig_representation.pdf")
    # segmentation figure
    p = os.path.join(RES, "unet3_bce_seed0.pt")
    if not os.path.exists(p): return
    m = M.UNet(3).to(E.dev); m.load_state_dict(torch.load(p)); m.eval()
    with torch.no_grad():
        P = torch.sigmoid(m(torch.from_numpy(X[D.CH3][None]).to(E.dev)))[0, 0].cpu().numpy()
    vol = np.maximum(X[0], X[1]); filt = vol * (1 - (P > 0.5))
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
    for a, im, t, cm in zip(axs, [vol, Y, P, filt], ["Observed book (volume)", "Injected mask (truth)", "Predicted probability", "Filtered book"], ["viridis", "gray_r", "magma", "viridis"]):
        a.imshow(im, aspect="auto", cmap=cm, vmin=0, vmax=1, interpolation="nearest"); a.set_title(t, fontsize=7.5); a.set_xlabel("Event index")
        a.axhline(9.5, color="w" if cm != "gray_r" else "k", lw=0.4, ls=":")
    axs[0].set_ylabel("Book row"); save("fig_segmentation.pdf")


def execution():
    ex = J("execution_m3b@0.json")
    if not ex: return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    dens = list(ex); w = 0.3
    for j, a in enumerate(["unfiltered", "filtered"]):
        axs[0].bar(np.arange(len(dens)) + (j - 0.5) * w, [ex[d][a]["IS"] - ex[d]["clean"]["IS"] for d in dens], w, label=a, color=[C1, C2][j])
    axs[0].set_xticks(range(len(dens))); axs[0].set_xticklabels(dens); axs[0].set_xlabel("Injected spoofs per 1000 events")
    axs[0].set_ylabel("IS minus clean-book IS (bps)"); axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    mean = [ex[d]["delta_IS_bps"]["mean"] for d in dens]; lo = [ex[d]["delta_IS_bps"]["ci95"][0] for d in dens]; hi = [ex[d]["delta_IS_bps"]["ci95"][1] for d in dens]
    axs[1].errorbar(range(len(dens)), mean, yerr=[np.array(mean) - lo, np.array(hi) - mean], fmt="o", color=C2, capsize=3)
    axs[1].axhline(0, color="#555", lw=0.7); axs[1].set_xticks(range(len(dens))); axs[1].set_xticklabels(dens)
    axs[1].set_xlabel("Injected spoofs per 1000 events"); axs[1].set_ylabel("$\\Delta$IS, filtered $-$ unfiltered (bps)")
    fig.subplots_adjust(wspace=0.4); save("fig_execution.pdf")


def diagnosis():
    dg = J("diagnosis.json")
    if not dg or "unet3" not in dg: return
    R = dg["unet3"]; x = np.arange(len(T))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    axs[0].bar(x - 0.18, [R["per_ticker_val"][t]["precision"] for t in T], 0.36, label="validation", color=C1)
    axs[0].bar(x + 0.18, [R["per_ticker_test"][t]["precision"] for t in T], 0.36, label="test", color=C2)
    axs[0].set_xticks(x); axs[0].set_xticklabels(T); axs[0].set_ylabel("Precision (U-Net, 3 ch)"); axs[0].set_ylim(0, 1.25); axs[0].legend(frameon=False, ncol=2, loc="upper center")
    th = list(R["threshold"]); axs[1].plot([float(t) for t in th], [R["threshold"][t]["val"] for t in th], "-o", ms=3, color=C1, label="validation IoU")
    axs[1].plot([float(t) for t in th], [R["threshold"][t]["test"] for t in th], "-o", ms=3, color=C2, label="test IoU")
    axs[1].set_xlabel("Decision threshold"); axs[1].set_ylabel("IoU"); axs[1].set_ylim(0.5, 1.0); axs[1].legend(frameon=False, loc="center right"); fig.subplots_adjust(wspace=0.3); save("fig_diagnosis.pdf")


def real_examples():
    for tag in ("m3b@0", "unet3_bce", "unet3"):
        p = os.path.join(RES, f"real_examples_{tag}.npz")
        if os.path.exists(p):
            d = np.load(p); n = min(4, len(d["X"]))
            fig, axs = plt.subplots(2, n, figsize=(7.0, 2.8), sharey=True)
            for i in range(n):
                axs[0, i].imshow(np.maximum(d["X"][i][0], d["X"][i][1]), aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
                axs[0, i].set_title(str(d["tick"][i]), fontsize=7.5)
                axs[1, i].imshow(d["P"][i], aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest"); axs[1, i].set_xlabel("Event index")
            axs[0, 0].set_ylabel("Observed volume"); axs[1, 0].set_ylabel("Predicted mask"); save("fig_real.pdf"); return


if __name__ == "__main__":
    pipeline(); window_example(); execution(); diagnosis(); real_examples()
