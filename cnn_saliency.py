"""
Saliency heatmap: which input channels (body joints + club path) at which point
in the swing drive the CNN's decision. Gradient of the model's OTT logit w.r.t.
the input, |.|, averaged over held-out clips, shown separately for true-OTT and
true-non-OTT clips so you can see how it differentiates the two.

Reading it: bright = the model leans on that channel at that phase. If the signal
sits on the clubhead/wrists during the downswing (right half), it's using swing
mechanics; if it's smeared across static body channels for the whole clip, it's
riding the pro-vs-amateur shortcut.  Run:  python cnn_saliency.py
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split

import cnn_pose as c

SEED = 0


def saliency(model, X):
    model.eval()
    xt = torch.tensor(X, device=c.DEVICE, requires_grad=True)
    logit = model(xt)
    logit.sum().backward()
    return xt.grad.abs().cpu().numpy()          # (N, 33, 64)


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    X, y, _ = c.build_dataset()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25,
                              stratify=y, random_state=SEED)
    itr, iva = train_test_split(tr, test_size=0.15, stratify=y[tr],
                                random_state=SEED)
    pw = (y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1)
    model = c.train_one(X[itr], y[itr], X[iva], y[iva], pw)

    sal = saliency(model, X[te])                # (Nte, 33, 64)
    yte = y[te]
    sal_ott = sal[yte == 1].mean(0)             # (33, 64)
    sal_non = sal[yte == 0].mean(0)
    vmax = max(sal_ott.max(), sal_non.max())

    # ---- figure ----
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888",
                         "figure.facecolor": "white"})
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.12)
    axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1]); cax = fig.add_subplot(gs[2])

    t_pct = [0, 16, 32, 48, 63]                 # tick positions over 64 steps
    for ax, data, title in [(axL, sal_ott, f"true OTT  (n={int((yte==1).sum())})"),
                            (axR, sal_non, f"true non-OTT  (n={int((yte==0).sum())})")]:
        im = ax.imshow(data, aspect="auto", cmap="cividis", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xticks(t_pct); ax.set_xticklabels([f"{int(p/63*100)}%" for p in t_pct])
        ax.set_xlabel("swing time (clip start → end)")
        # divider between pose block (0-23) and club block (24-32)
        ax.axhline(c.N_POSE - 0.5, color="white", lw=1.5)

    axL.set_yticks(range(len(c.CHANNEL_NAMES)))
    axL.set_yticklabels(c.CHANNEL_NAMES, fontsize=7)
    axR.set_yticks([])
    # bracket labels for the two channel groups
    axL.text(-9.5, (c.N_POSE - 1) / 2, "BODY", rotation=90, va="center",
             ha="center", fontsize=9, weight="bold", color="#444")
    axL.text(-9.5, (c.N_POSE + len(c.CHANNEL_NAMES) - 1) / 2, "CLUB", rotation=90,
             va="center", ha="center", fontsize=9, weight="bold", color="#444")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("mean |saliency|  (model reliance)", fontsize=9)
    fig.suptitle("What the CNN looks at to decide over-the-top\n"
                 "(gradient saliency by input channel × swing time)",
                 fontsize=12, y=0.98)
    fig.tight_layout(rect=[0.03, 0, 1, 0.94])
    fig.savefig("cnn_saliency_heatmap.png", dpi=150, bbox_inches="tight")
    print("saved cnn_saliency_heatmap.png")

    # quick text read-out: top channels by total reliance (both classes)
    total = sal.mean(0).sum(1)
    order = np.argsort(total)[::-1]
    print("\ntop channels by reliance:")
    for i in order[:8]:
        print(f"  {c.CHANNEL_NAMES[i]:12s} {total[i]/total.max():.2f}")


if __name__ == "__main__":
    main()
