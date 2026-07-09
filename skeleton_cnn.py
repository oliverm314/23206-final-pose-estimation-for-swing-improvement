"""
Supervisor's idea: render the pose as skeleton IMAGES on a black background and
train a CNN on them. A swing is a video, so we STACK the frames into a volume
(time x height x width) and use a 3D CNN (Conv3D convolves over space AND time) —
the standard way video CNNs (C3D/I3D) handle frames.

Rendering on black + normalising the skeleton removes the background / framing /
resolution leaks we caught earlier; it does NOT remove the build/stance shape, so
the pro-vs-amateur confound can still leak through posture. Main upside vs the
coordinate CNN: this image format is what lets you fine-tune PRETRAINED 2D/3D nets
later. Kept tiny + regularised because a 3D CNN on ~230 clips overfits easily.

    python skeleton_cnn.py   ->  CV-AUC + skeleton_example.png (a swing's frames)
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

import train_xgboost as t
import cnn_pose as c

JOINTS = c.JOINTS                       # 12 body joints, order = c.JOINT_NAMES
# skeleton edges (indices into the 12-joint array)
EDGES = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7),
         (6, 7), (6, 8), (8, 10), (7, 9), (9, 11)]
T, H, W = 16, 48, 48                    # frames, height, width
DEVICE = c.DEVICE


def _interp(y):
    v = np.isfinite(y)
    return np.zeros_like(y) if v.sum() < 2 else np.interp(
        np.arange(len(y)), np.arange(len(y))[v], y[v])


def render(J):
    """(12,2) normalised joints -> (H,W) uint8 skeleton on black."""
    img = np.zeros((H, W), np.uint8)
    s = H * 0.32
    pt = [(int(W / 2 + J[i, 0] * s), int(H / 2 + J[i, 1] * s)) for i in range(12)]
    for a, b in EDGES:
        cv2.line(img, pt[a], pt[b], 255, 1, cv2.LINE_AA)
    for i in range(12):
        cv2.circle(img, pt[i], 1, 255, -1)
    return img


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

    xy = lm[:, JOINTS, :2].astype(float)
    for j in range(xy.shape[1]):
        for d in range(2):
            xy[:, j, d] = _interp(xy[:, j, d])
    midhip = (xy[:, 6] + xy[:, 7]) / 2.0
    xy = xy - midhip[:, None, :]                     # centre each frame
    sh, ank = (xy[:, 0] + xy[:, 1]) / 2.0, (xy[:, 10] + xy[:, 11]) / 2.0
    scale = np.median(np.linalg.norm(sh - ank, axis=1))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    xy = xy / scale

    crop = xy[top:imp + 1].reshape(imp + 1 - top, -1)   # (F, 24)
    src, dst = np.linspace(0, 1, len(crop)), np.linspace(0, 1, T)
    res = np.stack([np.interp(dst, src, crop[:, k]) for k in range(24)], 1)
    res = res.reshape(T, 12, 2)
    vol = np.stack([render(res[f]) for f in range(T)], 0)   # (T,H,W)
    return (vol.astype(np.float32) / 255.0)


def build_dataset():
    X, y, names = [], [], []
    for path, label in t.find_videos(t.DATASET_DIR):
        seq, _ = t.load_or_extract_pose(path)
        v = build_one(seq)
        if v is not None:
            X.append(v); y.append(label); names.append(os.path.basename(path))
    return np.stack(X), np.array(y), names


class Skel3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.BatchNorm3d(8), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(16, 1))

    def forward(self, x):
        return self.head(self.net(x).flatten(1)).squeeze(-1)


def train_one(Xtr, ytr, Xva, yva, pw, epochs=60, patience=12):
    m = Skel3D().to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=DEVICE))
    Xva_t = torch.tensor(Xva[:, None], device=DEVICE)
    best, state, bad = -1, None, 0
    n = len(Xtr)
    for _ in range(epochs):
        m.train(); perm = np.random.permutation(n)
        for i in range(0, n, 8):
            idx = perm[i:i + 8]
            xb = Xtr[idx].copy()
            for k in range(len(xb)):
                if np.random.rand() < 0.5:
                    xb[k] = xb[k][:, :, ::-1]           # horizontal mirror
            xb = torch.tensor(np.ascontiguousarray(xb[:, None]), device=DEVICE)
            yb = torch.tensor(ytr[idx], dtype=torch.float32, device=DEVICE)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = torch.sigmoid(m(Xva_t)).cpu().numpy()
        auc = roc_auc_score(yva, p) if len(np.unique(yva)) > 1 else 0.5
        if auc > best:
            best, bad = auc, 0
            state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    return m


def save_example(X, y, names):
    i = int(np.where(y == 1)[0][0])
    grid = np.concatenate([np.pad(X[i, f], 1, constant_values=0.3)
                           for f in range(T)], axis=1)
    cv2.imwrite("skeleton_example.png", (grid * 255).astype(np.uint8))
    print(f"saved skeleton_example.png  ({names[i]}, {T} downswing frames)")


def main():
    X, y, names = build_dataset()
    print(f"usable={len(y)}  OTT={int(y.sum())}  non={int((1-y).sum())}  "
          f"vol={X.shape[1:]}  device={DEVICE}")
    save_example(X, y, names)
    seed_means = []
    for seed in range(3):
        np.random.seed(seed); torch.manual_seed(seed)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        aucs = []
        for tr, te in skf.split(X, y):
            itr, iva = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=seed)
            pw = (y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1)
            m = train_one(X[itr], y[itr], X[iva], y[iva], pw)
            with torch.no_grad():
                p = torch.sigmoid(m(torch.tensor(X[te][:, None], device=DEVICE))).cpu().numpy()
            aucs.append(roc_auc_score(y[te], p))
        seed_means.append(float(np.mean(aucs)))
        print(f"  seed {seed}: 5-fold CV-AUC = {seed_means[-1]:.3f}")
    print(f"\nSkeleton 3D-CNN 5-fold CV-AUC = {np.mean(seed_means):.3f} +/- {np.std(seed_means):.3f}")
    print("(direction 1D-CNN 0.906 ; XGBoost 0.913)")


if __name__ == "__main__":
    main()
