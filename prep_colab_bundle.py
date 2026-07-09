"""
Make a compact bundle for the Colab notebook: for every swing, the DOWNSWING is
cropped, the pose is centred per-frame on the mid-hip and scaled by body size,
and everything is resampled to a fixed length. That normalisation already removes
the biggest shortcuts we found (camera zoom / framing / absolute position), so the
notebook only has to render + train. Club points are carried (normalised in the
same frame) with a detection mask so the notebook can draw the shaft/head.

Output: golf_pose_bundle.npz  (a few MB) -> upload this to Colab.
Run locally:  python prep_colab_bundle.py
"""
import os
import numpy as np
import train_xgboost as t
import cnn_pose as c

T = 16
JOINTS = c.JOINTS


def _interp(y):
    v = np.isfinite(y)
    return np.zeros_like(y) if v.sum() < 2 else np.interp(
        np.arange(len(y)), np.arange(len(y))[v], y[v])


def _resample(a, n=T):
    src, dst = np.linspace(0, 1, len(a)), np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, a[:, k]) for k in range(a.shape[1])], 1)


def one(seq):
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
    xy = xy - midhip[:, None, :]
    sh, ank = (xy[:, 0] + xy[:, 1]) / 2.0, (xy[:, 10] + xy[:, 11]) / 2.0
    scale = np.median(np.linalg.norm(sh - ank, axis=1))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    xy = xy / scale

    F = lm.shape[0]
    club = np.zeros((F, 3, 2))
    cmask = np.zeros((F, 3))
    if seq.club is not None and seq.club.shape[0] >= F:
        for k, ci in enumerate((t.HEAD, t.HANDLE, t.BALL)):
            cx, cy = seq.club[:F, ci, 0], seq.club[:F, ci, 1]
            det = np.isfinite(cx) & np.isfinite(cy)
            club[:, k, 0] = np.where(det, (np.where(det, cx, 0) - midhip[:, 0]) / scale, 0)
            club[:, k, 1] = np.where(det, (np.where(det, cy, 0) - midhip[:, 1]) / scale, 0)
            cmask[:, k] = det

    a, b = top, imp + 1
    joints = _resample(xy[a:b].reshape(b - a, -1)).reshape(T, 12, 2)
    club_r = _resample(club[a:b].reshape(b - a, -1)).reshape(T, 3, 2)
    mask_r = _resample(cmask[a:b])                       # (T,3) fractions
    return joints, club_r, mask_r


def main():
    J, C, M, y, names = [], [], [], [], []
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
        r = one(seq)
        if r is None:
            continue
        J.append(r[0]); C.append(r[1]); M.append(r[2])
        y.append(label); names.append(os.path.basename(path))

    J, C, M, y = np.array(J, np.float32), np.array(C, np.float32), \
        np.array(M, np.float32), np.array(y, np.int64)
    np.savez_compressed("golf_pose_bundle.npz", joints=J, club=C, club_mask=M,
                        y=y, names=np.array(names), T=T)
    sz = os.path.getsize("golf_pose_bundle.npz") / 1e6
    print(f"saved golf_pose_bundle.npz  N={len(y)}  OTT={int(y.sum())} "
          f"non={int((1-y).sum())}  joints={J.shape}  ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
