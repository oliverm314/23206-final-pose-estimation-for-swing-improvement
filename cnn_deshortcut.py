"""
De-shortcutted CNN: forces the model off the pro-vs-amateur stance cue by
applying all three Lever-1 fixes to the input:

  1. CROP to the downswing (top -> impact) and phase-normalise it to a fixed
     length -> removes the static whole-clip stance signal AND tempo.
  2. VELOCITIES not positions for every joint/club point -> a static knee has
     ~zero velocity, so the stance shortcut disappears.
  3. CLUB-RELATIVE-TO-PLANE: an extra channel = signed clubhead height above the
     ball->trail-elbow plane line per frame (the literal over-the-top geometry),
     which is meaningful position info that is NOT a stance/framing proxy.

Channels (34): 12 joints x (vx,vy)=24, club head/handle/ball x (vx,vy)=6,
3 detection masks, 1 clubhead-above-plane. Run:  python cnn_deshortcut.py
(prints 5-fold CV and writes cnn_deshortcut_saliency.png)
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

import train_xgboost as t
import cnn_pose as c

L = 48
# Unit-normalise each velocity vector -> keep motion DIRECTION (technique), drop
# SPEED (athleticism / pro-vs-amateur confound). Direction-only scored 0.910 vs
# 0.812 speed-only vs 0.890 full velocity, so this is both better and cleaner.
# Flip to False to reproduce the full-velocity model.
DIRECTION_ONLY = True
JOINTS = c.JOINTS
CHANNEL_NAMES = ([f"{j}_v{ax}" for j in c.JOINT_NAMES for ax in ("x", "y")]
                 + ["head_vx", "head_vy", "handle_vx", "handle_vy",
                    "ball_vx", "ball_vy", "head_det", "handle_det", "ball_det",
                    "head_above_plane"])
DEVICE = c.DEVICE


def _interp(y):
    v = np.isfinite(y)
    return np.zeros_like(y) if v.sum() < 2 else np.interp(
        np.arange(len(y)), np.arange(len(y))[v], y[v])


def _resample(a, n):
    m = len(a)
    if m < 2:
        return None
    src, dst = np.linspace(0, 1, m), np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, a[:, k]) for k in range(a.shape[1])], 1)


def build_one(seq):
    lm = seq.landmarks
    if lm.shape[0] == 0:
        return None
    ph = t.detect_swing_phases(seq)
    if ph.downswing_start is None:
        return None
    top = ph.downswing_start
    imp = ph.downswing_end if ph.downswing_end is not None else lm.shape[0] - 1
    if imp - top < 2:
        imp = min(lm.shape[0] - 1, top + 2)
    a, b = top, imp + 1

    xy = lm[:, JOINTS, :2].astype(float)
    for j in range(xy.shape[1]):
        for d in range(2):
            xy[:, j, d] = _interp(xy[:, j, d])
    midhip = (xy[:, 6] + xy[:, 7]) / 2.0
    xy = xy - midhip[:, None, :]
    sh, ank = (xy[:, 0] + xy[:, 1]) / 2.0, (xy[:, 10] + xy[:, 11]) / 2.0
    scale = np.median(np.linalg.norm(sh - ank, axis=1))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    xy = xy / scale
    body = xy[a:b].reshape(b - a, -1)                    # (F, 24) positions

    F = b - a
    club_pos, mask, plane = np.zeros((F, 6)), np.zeros((F, 3)), np.zeros((F, 1))
    if seq.club is not None and seq.club.shape[0] >= b:
        for k, ci in enumerate((t.HEAD, t.HANDLE, t.BALL)):
            cx, cy = seq.club[a:b, ci, 0], seq.club[a:b, ci, 1]
            det = np.isfinite(cx) & np.isfinite(cy)
            club_pos[:, 2 * k] = np.where(det, (np.where(det, cx, 0) - midhip[a:b, 0]) / scale, 0)
            club_pos[:, 2 * k + 1] = np.where(det, (np.where(det, cy, 0) - midhip[a:b, 1]) / scale, 0)
            mask[:, k] = det
        line = t.plane_line_at_top(seq)
        bs = t._body_scale(lm)
        if line is not None and np.isfinite(bs) and bs > 1e-6:
            bx, by, ex, ey = line
            dxp = ex - bx
            if abs(dxp) > 1e-6:
                head = seq.club[:, t.HEAD, :2]
                for i, f in enumerate(range(a, b)):
                    hx, hy = head[f]
                    if np.all(np.isfinite([hx, hy])):
                        line_y = by + (ey - by) / dxp * (hx - bx)
                        plane[i, 0] = (line_y - hy) / bs

    pos = np.concatenate([body, club_pos], axis=1)       # (F, 30) positions
    posr = _resample(pos, L)
    maskr, planer = _resample(mask, L), _resample(plane, L)
    if posr is None:
        return None
    vel = np.diff(posr, axis=0, prepend=posr[:1])        # (L, 30) velocities
    if DIRECTION_ONLY:                                   # keep which-way, drop speed
        V = vel.reshape(L, 15, 2)
        vel = (V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-6)).reshape(L, 30)
    feats = np.concatenate([vel, maskr, planer], axis=1)  # (L, 34)
    return feats.T.astype(np.float32)


def mirror(x):
    x = x.copy()
    body = x[:24].reshape(12, 2, -1)
    body[:, 0, :] *= -1.0
    for aa, bb in c.PAIRS:
        body[[aa, bb]] = body[[bb, aa]]
    x[:24] = body.reshape(24, -1)
    for cx in (24, 26, 28):                              # club vx channels
        x[cx] *= -1.0
    return x                                             # masks + plane invariant


def build_dataset():
    X, y, names = [], [], []
    for path, label in t.find_videos(t.DATASET_DIR):
        seq, _ = t.load_or_extract_pose(path)
        if t.USE_CLUB and os.path.exists(t.CLUB_WEIGHTS):
            raw, _ = t.load_or_extract_club(path)
            club = t._clean_club_track(raw)
            if (seq.landmarks.shape[0] and club.shape[0]
                    and club.shape[0] != seq.landmarks.shape[0]):
                m = min(club.shape[0], seq.landmarks.shape[0])
                seq.landmarks, club = seq.landmarks[:m], club[:m]
            seq.club = club
        ten = build_one(seq)
        if ten is not None:
            X.append(ten); y.append(label); names.append(os.path.basename(path))
    return np.stack(X), np.array(y), names


def train_one(Xtr, ytr, Xva, yva, pos_weight, epochs=80, patience=15):
    model = c.PoseCNN(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
    Xva_t = torch.tensor(Xva, device=DEVICE)
    best_auc, best_state, bad = -1.0, None, 0
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, 16):
            idx = perm[i:i + 16]
            xb = Xtr[idx].copy()
            for k in range(len(xb)):
                if np.random.rand() < 0.5:
                    xb[k] = mirror(xb[k])
            xb += np.random.randn(*xb.shape).astype(np.float32) * 0.02
            yb = torch.tensor(ytr[idx], dtype=torch.float32, device=DEVICE)
            opt.zero_grad()
            lossf(model(torch.tensor(xb, device=DEVICE)), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva_t)).cpu().numpy()
        auc = roc_auc_score(yva, p) if len(np.unique(yva)) > 1 else 0.5
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def saliency_plot(X, y):
    np.random.seed(0); torch.manual_seed(0)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=0)
    itr, iva = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=0)
    pw = (y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1)
    model = train_one(X[itr], y[itr], X[iva], y[iva], pw)
    xt = torch.tensor(X[te], device=DEVICE, requires_grad=True)
    model.eval(); model(xt).sum().backward()
    sal = xt.grad.abs().cpu().numpy()
    yte = y[te]
    so, sn = sal[yte == 1].mean(0), sal[yte == 0].mean(0)
    vmax = max(so.max(), sn.max())
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.12)
    axL, axR, cax = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]))
    for ax, data, title in [(axL, so, f"true OTT (n={int((yte==1).sum())})"),
                            (axR, sn, f"true non-OTT (n={int((yte==0).sum())})")]:
        im = ax.imshow(data, aspect="auto", cmap="cividis", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10); ax.set_xlabel("downswing  top → impact")
        ax.set_xticks([0, L // 2, L - 1]); ax.set_xticklabels(["top", "mid", "impact"])
        for hl in (23.5, 29.5, 32.5):
            ax.axhline(hl, color="white", lw=1.2)
    axL.set_yticks(range(len(CHANNEL_NAMES))); axL.set_yticklabels(CHANNEL_NAMES, fontsize=7)
    axR.set_yticks([])
    fig.colorbar(im, cax=cax).set_label("mean |saliency|", fontsize=9)
    tag = "direction" if DIRECTION_ONLY else "velocity"
    sub = ("direction only: WHICH-WAY joints/club move, speed removed"
           if DIRECTION_ONLY else "velocities + club-above-plane; stance removed")
    fig.suptitle(f"De-shortcutted CNN: reliance by channel × downswing time\n({sub})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"cnn_deshortcut_{tag}_saliency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    total = sal.mean(0).sum(1); order = np.argsort(total)[::-1]
    print("top channels by reliance:")
    for i in order[:8]:
        print(f"  {CHANNEL_NAMES[i]:16s} {total[i]/total.max():.2f}")


def main():
    X, y, _ = build_dataset()
    print(f"usable (downswing found)={len(y)}  OTT={int(y.sum())}  non={int((1-y).sum())}  X={X.shape}")
    seed_means = []
    for seed in range(3):
        np.random.seed(seed); torch.manual_seed(seed)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        aucs = []
        for tr, te in skf.split(X, y):
            itr, iva = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=seed)
            pw = (y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1)
            model = train_one(X[itr], y[itr], X[iva], y[iva], pw)
            with torch.no_grad():
                p = torch.sigmoid(model(torch.tensor(X[te], device=DEVICE))).cpu().numpy()
            aucs.append(roc_auc_score(y[te], p))
        seed_means.append(float(np.mean(aucs)))
        print(f"  seed {seed}: 5-fold CV-AUC = {seed_means[-1]:.3f}")
    mode = "direction-only" if DIRECTION_ONLY else "full-velocity"
    print(f"\nDe-shortcut CNN ({mode}) 5-fold CV-AUC = "
          f"{np.mean(seed_means):.3f} +/- {np.std(seed_means):.3f}")
    print("(full-velocity de-shortcut 0.902 ; pose+club 0.896 ; XGBoost 0.913)")
    saliency_plot(X, y)


if __name__ == "__main__":
    main()
