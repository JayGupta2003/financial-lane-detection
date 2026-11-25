"""Load a saved model by tag. Tags: 'unet3_bce' -> results/unet3_bce_seed0.pt (legacy); 'm3b@1' -> results/models/m3b_s1.pt."""
import os, torch
import models as M, data as D, experiment as E

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
CHS = {2: D.CH2, 3: D.CH3, 6: D.CH6}


def path(tag):
    if "@" in tag:
        name, seed = tag.split("@"); return os.path.join(OUT, "models", f"{name}_s{seed}.pt")
    return os.path.join(OUT, f"{tag}_seed0.pt")


def load_model(tag):
    sd = torch.load(path(tag), map_location=E.dev)
    c = sd["enc.0.0.weight"].shape[1]
    m = M.UNet(c).to(E.dev); m.load_state_dict(sd); m.eval()
    return m, CHS[c]
