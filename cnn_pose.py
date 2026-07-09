"""
Prototype: small 1D temporal CNN on MediaPipe pose + YOLO club-track sequences,
as a head-to-head contrast against the feature + XGBoost model.

Input per clip is a (33, 64) array = 33 channels x 64 normalised-time steps:
    - 12 body joints x (x, y)                 -> 24 channels
    - clubhead / handle / ball x (x, y)       ->  6 channels
    - clubhead / handle / ball detection flag ->  3 channels
Everything is centred on mid-hip and scaled by body size, so absolute position
and camera distance are normalised out. Club points share that frame so the
club path sits in the same coordinate space as the body.

Deliberately small + heavily regularised (dataset ~230 clips). Reuses
train_xgboost's cached pose+club extraction. Run from the project root:
    python cnn_pose.py
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

import train_xgboost as t

# body joints, ordered as L,R pairs so a mirror flip = swap each pair + negate x
JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]   # sh/el/wr/hip/kn/ank
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
JOINT_NAMES = ["L_sh", "R_sh", "L_el", "R_el", "L_wr", "R_wr",
               "L_hip", "R_hip", "L_kn", "R_kn", "L_an", "R_an"]
# human-readable channel names (for the saliency heatmap)
CHANNEL_NAMES = [f"{j}_{ax}" for j in JOINT_NAMES for ax in ("x", "y")] + [
    "head_x", "head_y", "handle_x", "handle_y", "ball_x", "ball_y",
    "head_det", "handle_det", "ball_det"]
N_POSE = 2 * len(JOINTS)            # 24
CLUB_X_CHANNELS = (24, 26, 28)      # head_x, handle_x, ball_x (negated on mirror)
SEQ_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _interp_nan_1d(y):
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return np.zeros_like(y)
    idx = np.arange(len(y))
    return np.interp(idx, idx[valid], y[valid])


def _resample(a, n=SEQ_LEN):
    """Resample (frames, C) -> (n, C) over normalised time."""
    m = len(a)
    if m == n:
        return a
    src, dst = np.linspace(0, 1, m), np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, a[:, c]) for c in range(a.shape[1])], 1)


def video_to_tensor(landmarks, club=None):
    """(frames,33,4) pose [+ (frames,3,3) club] -> (33, SEQ_LEN). None if unusable."""
    if landmarks.shape[0] == 0:
        return None
    F = landmarks.shape[0]
    xy = landmarks[:, JOINTS, :2].astype(float)             # (F, J, 2)
    for j in range(xy.shape[1]):
        for d in range(2):
            xy[:, j, d] = _interp_nan_1d(xy[:, j, d])
    midhip = (xy[:, 6] + xy[:, 7]) / 2.0                    # L/R hip
    xy = xy - midhip[:, None, :]
    sh = (xy[:, 0] + xy[:, 1]) / 2.0
    ank = (xy[:, 10] + xy[:, 11]) / 2.0
    scale = np.median(np.linalg.norm(sh - ank, axis=1))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    xy = xy / scale
    pose = xy.reshape(F, -1)                                # (F, 24)

    # club: clubhead / handle / ball, centred+scaled in the SAME frame, + masks
    club_ch = np.zeros((F, 9), dtype=float)
    if club is not None and club.shape[0]:
        Fc = min(F, club.shape[0])
        for k, ci in enumerate((t.HEAD, t.HANDLE, t.BALL)):
            cx, cy = club[:Fc, ci, 0], club[:Fc, ci, 1]
            det = np.isfinite(cx) & np.isfinite(cy)
            px = (np.where(det, cx, 0.0) - midhip[:Fc, 0]) / scale
            py = (np.where(det, cy, 0.0) - midhip[:Fc, 1]) / scale
            club_ch[:Fc, 2 * k] = np.where(det, px, 0.0)
            club_ch[:Fc, 2 * k + 1] = np.where(det, py, 0.0)
            club_ch[:Fc, 6 + k] = det.astype(float)

    feats = np.concatenate([pose, club_ch], axis=1)         # (F, 33)
    return _resample(feats).T.astype(np.float32)


def mirror(x):
    """Mirror a (33, L) sample: negate x of every joint + club point, swap L/R
    joint pairs. Detection-flag channels are left as-is."""
    x = x.copy()
    pose = x[:N_POSE].reshape(len(JOINTS), 2, -1)
    pose[:, 0, :] *= -1.0
    for a, b in PAIRS:
        pose[[a, b]] = pose[[b, a]]
    x[:N_POSE] = pose.reshape(N_POSE, -1)
    for cx in CLUB_X_CHANNELS:
        x[cx] *= -1.0
    return x


class PoseCNN(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.net(x).squeeze(-1)).squeeze(-1)


def train_one(Xtr, ytr, Xva, yva, pos_weight, epochs=80, patience=15):
    model = PoseCNN(Xtr.shape[1]).to(DEVICE)
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
            xb = torch.tensor(xb, device=DEVICE)
            yb = torch.tensor(ytr[idx], dtype=torch.float32, device=DEVICE)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
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


def build_dataset():
    """Return (X, y, names): the (N,33,64) tensors, labels, and video names."""
    X, y, names = [], [], []
    for path, label in t.find_videos(t.DATASET_DIR):
        seq, _ = t.load_or_extract_pose(path)
        club = None
        if t.USE_CLUB and os.path.exists(t.CLUB_WEIGHTS):
            raw, _ = t.load_or_extract_club(path)
            club = t._clean_club_track(raw)
        ten = video_to_tensor(seq.landmarks, club)
        if ten is not None:
            X.append(ten); y.append(label); names.append(os.path.basename(path))
    return np.stack(X), np.array(y), names


def main():
    X, y, _ = build_dataset()
    print(f"usable videos={len(y)}  OTT={int(y.sum())}  non={int((1-y).sum())}  "
          f"X={X.shape}  device={DEVICE}")

    seed_means = []
    for seed in range(3):
        np.random.seed(seed); torch.manual_seed(seed)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        aucs = []
        for tr, te in skf.split(X, y):
            itr, iva = train_test_split(tr, test_size=0.15, stratify=y[tr],
                                        random_state=seed)
            pw = (y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1)
            model = train_one(X[itr], y[itr], X[iva], y[iva], pw)
            with torch.no_grad():
                p = torch.sigmoid(model(torch.tensor(X[te], device=DEVICE))).cpu().numpy()
            aucs.append(roc_auc_score(y[te], p))
        seed_means.append(float(np.mean(aucs)))
        print(f"  seed {seed}: 5-fold CV-AUC = {seed_means[-1]:.3f}  "
              f"(folds {np.round(aucs,3)})")
    print(f"\nPoseCNN+club 5-fold CV-AUC (avg of 3 seeds) = "
          f"{np.mean(seed_means):.3f} +/- {np.std(seed_means):.3f}")
    print("pose-only CNN was 0.960 ; XGBoost+features 0.913")


if __name__ == "__main__":
    main()
