"""U-Net, CNN classifier baseline, and classical detectors for LOB anomaly masking."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Encoder widths 32-64-128-256, 512 bottleneck, 1x1 conv head (logits)."""

    def __init__(self, in_ch=3, widths=(32, 64, 128, 256), bottleneck=512):
        super().__init__()
        self.enc = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.enc.append(block(c, w))
            c = w
        self.mid = block(c, bottleneck)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.dec.append(block(2 * w, w))
            c = w
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        d = 2 ** len(self.enc)
        ph, pw = (-h) % d, (-w) % d
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], 1))
        return self.head(x)[..., :h, :w]


class CNNClassifier(nn.Module):
    """Baseline: window-level anomaly classifier (no spatial output).
    Pixel masks for it are produced by a Grad-free fallback: the whole window mask is
    either empty or the volume-above-median region, see evaluate.classifier_mask."""

    def __init__(self, in_ch=3):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc(self.f(x).flatten(1))


def volume_threshold_mask(x, thr):
    """x: (B,C,H,W) normalized tensor, channels 0/1 are ask/bid volume."""
    return (x[:, :2].max(1, keepdim=True).values > thr).float()


def zscore_mask(x, z=3.0, k=15):
    """Local z-score of volume vs a (2k+1)-column trailing/leading neighborhood per row."""
    v = x[:, :2].max(1, keepdim=True).values
    pad = F.pad(v, (k, k, 0, 0), mode="replicate")
    mean = F.avg_pool2d(pad, (1, 2 * k + 1), stride=1)
    sq = F.avg_pool2d(pad ** 2, (1, 2 * k + 1), stride=1)
    std = (sq - mean ** 2).clamp_min(1e-8).sqrt()
    return ((v - mean) / std > z).float()


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def dice_bce_loss(logits, y, w=0.5, eps=1.0, use_dice=True, use_bce=True):
    loss = 0.0
    if use_bce:
        loss = loss + (w if use_dice else 1.0) * F.binary_cross_entropy_with_logits(logits, y)
    if use_dice:
        p = torch.sigmoid(logits)
        inter = (p * y).sum((1, 2, 3))
        dice = (2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)
        loss = loss + (w if use_bce else 1.0) * (1 - dice).mean()
    return loss


def get_loss(name):
    """Named loss variants for the loss study (logits, y: (B,1,H,W))."""
    F_ = F

    def per_window_dice(logits, y, eps=1.0):
        p = torch.sigmoid(logits)
        inter = (p * y).sum((1, 2, 3))
        return (2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)

    if name == "bce":
        return lambda l, y: F_.binary_cross_entropy_with_logits(l, y)
    if name == "bce_dice":
        return lambda l, y: 0.5 * F_.binary_cross_entropy_with_logits(l, y) + 0.5 * (1 - per_window_dice(l, y)).mean()
    if name == "bce_dice_batch":      # Dice over the whole batch instead of per window
        def f(l, y):
            p = torch.sigmoid(l)
            d = (2 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)
            return 0.5 * F_.binary_cross_entropy_with_logits(l, y) + 0.5 * (1 - d)
        return f
    if name == "bce_dice_posonly":    # per-window Dice only on windows that contain a positive cell
        def f(l, y):
            pos = y.sum((1, 2, 3)) > 0
            dl = (1 - per_window_dice(l, y))[pos].mean() if pos.any() else l.sum() * 0.0
            return 0.5 * F_.binary_cross_entropy_with_logits(l, y) + 0.5 * dl
        return f
    if name == "bce_posw":            # class-imbalance handled by a positive weight
        return lambda l, y: F_.binary_cross_entropy_with_logits(l, y, pos_weight=torch.tensor(10.0, device=l.device))
    if name == "focal":               # gamma=2, alpha=0.25 as in Lin et al.
        def f(l, y, g=2.0, a=0.25):
            ce = F_.binary_cross_entropy_with_logits(l, y, reduction="none")
            p = torch.sigmoid(l); pt = p * y + (1 - p) * (1 - y)
            at = a * y + (1 - a) * (1 - y)
            return (at * (1 - pt) ** g * ce).mean()
        return f
    raise ValueError(name)


if __name__ == "__main__":
    for c in (2, 3, 6):
        m = UNet(c)
        print(c, "ch params", count_params(m), m(torch.randn(2, c, 40, 256)).shape)
    print("cnn", count_params(CNNClassifier()))
