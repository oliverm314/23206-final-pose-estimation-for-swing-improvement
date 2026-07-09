"""Direction-only CNN with head_above_plane REMOVED: do the raw motion-direction
channels carry independent OTT signal, or was the plane feature doing all the work?"""
import numpy as np, torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import cnn_pose as c, cnn_deshortcut as d

X, y, _ = d.build_dataset()
X = X[:, :33, :].copy()                 # drop channel 33 = head_above_plane
names = d.CHANNEL_NAMES[:33]

means = []
for seed in range(3):
    np.random.seed(seed); torch.manual_seed(seed)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed); a = []
    for tr, te in skf.split(X, y):
        itr, iva = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=seed)
        pw = (y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1)
        m = d.train_one(X[itr], y[itr], X[iva], y[iva], pw)
        with torch.no_grad():
            p = torch.sigmoid(m(torch.tensor(X[te], device=c.DEVICE))).cpu().numpy()
        a.append(roc_auc_score(y[te], p))
    means.append(np.mean(a))
print(f"direction-only, NO plane channel: CV-AUC = {np.mean(means):.3f} +/- {np.std(means):.3f}")
print("(with plane 0.906 ; full-velocity 0.902 ; XGBoost 0.913)")

np.random.seed(0); torch.manual_seed(0)
tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=0)
itr, iva = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=0)
pw = (y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1)
m = d.train_one(X[itr], y[itr], X[iva], y[iva], pw)
xt = torch.tensor(X[te], device=c.DEVICE, requires_grad=True)
m.eval(); m(xt).sum().backward()
sal = xt.grad.abs().cpu().numpy(); yte = y[te]
so, sn = sal[yte == 1].mean(0), sal[yte == 0].mean(0); vmax = max(so.max(), sn.max())
fig = plt.figure(figsize=(11, 8)); gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.12)
axL, axR, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
for ax, data, title in [(axL, so, f"true OTT (n={int((yte==1).sum())})"),
                        (axR, sn, f"true non-OTT (n={int((yte==0).sum())})")]:
    im = ax.imshow(data, aspect="auto", cmap="cividis", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10); ax.set_xlabel("downswing  top → impact")
    ax.set_xticks([0, d.L // 2, d.L - 1]); ax.set_xticklabels(["top", "mid", "impact"])
    for hl in (23.5, 29.5): ax.axhline(hl, color="white", lw=1.2)
axL.set_yticks(range(len(names))); axL.set_yticklabels(names, fontsize=7); axR.set_yticks([])
fig.colorbar(im, cax=cax).set_label("mean |saliency|", fontsize=9)
fig.suptitle("Direction-only CNN — head_above_plane REMOVED", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("cnn_deshortcut_noplane_saliency.png", dpi=150, bbox_inches="tight")
print("saved cnn_deshortcut_noplane_saliency.png")
total = sal.mean(0).sum(1); order = np.argsort(total)[::-1]
print("top channels now:")
for i in order[:8]:
    print(f"  {names[i]:14s} {total[i]/total.max():.2f}")
