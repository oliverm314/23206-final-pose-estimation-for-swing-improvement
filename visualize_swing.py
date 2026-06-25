"""
Render one annotated swing MP4 for demonstration.

Overlays on the source video:
  * the MediaPipe pose skeleton (key joints),
  * the YOLO ball / club-handle / club-head detections,
  * the SWING-PLANE LINE (ball -> trail elbow) used by clubhead_above_plane,
    with the clubhead drawn GREEN when above the line / RED when below,
  * the lead-wrist path, coloured by phase (backswing vs downswing),
  * a banner naming the current swing phase, the detected phase frames, and the
    four feature values for the whole swing.

Usage:
    python visualize_swing.py                 # auto-pick a clean OTT example
    python visualize_swing.py "dataset/OTT/somefile.mp4"
    python visualize_swing.py "dataset/OTT/somefile.mp4" out.mp4
"""
from __future__ import annotations

import os
import sys
import glob

import cv2
import numpy as np

import train_xgboost as t


# Skeleton connections (MediaPipe landmark indices) we draw.
CONNECTIONS = [
    (11, 12),                       # shoulders
    (11, 13), (13, 15),             # left arm
    (12, 14), (14, 16),             # right arm
    (11, 23), (12, 24), (23, 24),   # torso
    (23, 25), (25, 27),             # left leg
    (24, 26), (26, 28),             # right leg
]
JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def _px(x, y, W, H):
    return int(round(x * W)), int(round(y * H))


def load_swing(video_path: str) -> t.SwingSequence:
    """Pose + cleaned club track for one video (uses the on-disk caches)."""
    seq, _ = t.load_or_extract_pose(video_path)
    club, _ = t.load_or_extract_club(video_path)
    if (seq.landmarks.shape[0] and club.shape[0]
            and club.shape[0] != seq.landmarks.shape[0]):
        L = min(club.shape[0], seq.landmarks.shape[0])
        seq.landmarks, club = seq.landmarks[:L], club[:L]
    seq.club = t._clean_club_track(club)
    return seq


def trail_elbow_idx(seq: t.SwingSequence, top: int) -> int:
    """Trail elbow = same side as the higher shoulder around the top."""
    lm = seq.landmarks
    a, b = max(0, top - 3), min(lm.shape[0], top + 4)
    lsy = np.nanmean(t._xy(lm, t.L_SHOULDER)[a:b, 1])
    rsy = np.nanmean(t._xy(lm, t.R_SHOULDER)[a:b, 1])
    return t.R_ELBOW if rsy < lsy else t.L_ELBOW


def auto_pick() -> str:
    """First OTT video with a full downswing and real ball+head detections."""
    for path in sorted(glob.glob(os.path.join(t.DATASET_DIR, "OTT", "*"))):
        if not path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue
        try:
            seq = load_swing(path)
        except Exception:
            continue
        ph = t.detect_swing_phases(seq)
        if ph.downswing_start is None or ph.downswing_end is None:
            continue
        if seq.club is None:
            continue
        ball_ok = np.isfinite(seq.club[:, t.BALL, 0]).sum() >= 5
        head_ok = np.isfinite(seq.club[:, t.HEAD, 0]).sum() >= 5
        if ball_ok and head_ok:
            return path
    raise RuntimeError("No suitable OTT video found; pass one explicitly.")


def phase_name(f: int, ph: t.SwingPhases) -> str:
    if ph.backswing_start is None or f < ph.backswing_start:
        return "Address"
    if ph.downswing_start is None or f < ph.downswing_start:
        return "Backswing"
    if ph.downswing_end is None or f <= ph.downswing_end:
        return "Downswing"
    return "Follow-through"


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else auto_pick()
    out_path = sys.argv[2] if len(sys.argv) > 2 else "swing_annotated.mp4"
    print(f"[info] annotating: {video_path}")

    seq = load_swing(video_path)
    lm = seq.landmarks
    n = lm.shape[0]
    ph = t.detect_swing_phases(seq)
    top = ph.downswing_start if ph.downswing_start is not None else 0
    imp = (ph.downswing_end if ph.downswing_end is not None else n - 1)
    elbow_idx = trail_elbow_idx(seq, top)

    # FIXED swing-plane line, determined once at the top and held constant — the
    # same line feat_clubhead_above_plane now measures against.
    line = t.plane_line_at_top(seq)
    bx = by = ex0 = ey0 = None
    if line is not None:
        bx, by, ex0, ey0 = line

    # Whole-swing feature values for the banner.
    feats = {name: fn(seq) for name, fn in t.FEATURE_REGISTRY.items()}

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))

    wrist = elbow_idx  # placeholder to avoid confusion below
    lead = seq.lead_wrist_idx
    trail_label = "R" if elbow_idx == t.R_ELBOW else "L"

    f = 0
    while f < n:
        ok, frame = cap.read()
        if not ok:
            break

        # --- skeleton ---
        for a, b in CONNECTIONS:
            pa, pb = t._xy(lm, a)[f], t._xy(lm, b)[f]
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                cv2.line(frame, _px(*pa, W, H), _px(*pb, W, H), (200, 200, 200), 2)
        for j in JOINTS:
            p = t._xy(lm, j)[f]
            if np.all(np.isfinite(p)):
                cv2.circle(frame, _px(*p, W, H), 4, (255, 255, 255), -1)

        # --- lead-wrist path so far, coloured by phase ---
        for g in range(1, f + 1):
            p0, p1 = t._xy(lm, lead)[g - 1], t._xy(lm, lead)[g]
            if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
                continue
            in_down = ph.downswing_start is not None and g >= ph.downswing_start \
                and g <= imp
            col = (0, 0, 255) if in_down else (255, 160, 0)   # down=red, back=blue
            cv2.line(frame, _px(*p0, W, H), _px(*p1, W, H), col, 2)

        # --- fixed swing-plane line (ball -> trail elbow @ top) + head above/below
        if bx is not None:
            p_ball = np.array(_px(bx, by, W, H), dtype=float)
            p_elb = np.array(_px(ex0, ey0, W, H), dtype=float)
            d = p_elb - p_ball
            nrm = np.linalg.norm(d)
            if nrm > 1e-6:
                d = d / nrm
                a_pt = (p_ball - d * 2000).astype(int)
                b_pt = (p_elb + d * 2000).astype(int)
                cv2.line(frame, tuple(a_pt), tuple(b_pt), (0, 255, 255), 2)

            # clubhead colour vs the fixed plane during the downswing
            hx, hy = (seq.club[f, t.HEAD, :2] if seq.club is not None
                      else (np.nan, np.nan))
            if np.all(np.isfinite([hx, hy])):
                hp = _px(hx, hy, W, H)
                col = (0, 255, 255)
                if (top <= f <= imp) and abs(ex0 - bx) > 1e-6:
                    line_y = by + (ey0 - by) / (ex0 - bx) * (hx - bx)
                    col = (0, 255, 0) if (line_y - hy) > 0 else (0, 0, 255)
                cv2.circle(frame, hp, 7, col, -1)
                cv2.putText(frame, "head", (hp[0] + 8, hp[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # --- ball + handle markers ---
        if bx is not None:
            cv2.circle(frame, _px(bx, by, W, H), 6, (0, 215, 255), 2)
        if seq.club is not None:
            hxh, hyh = seq.club[f, t.HANDLE, :2]
            if np.all(np.isfinite([hxh, hyh])):
                cv2.circle(frame, _px(hxh, hyh, W, H), 6, (255, 0, 255), -1)

        # --- banner: phase + phase frames + feature values ---
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (W, 96), (0, 0, 0), -1)
        frame = cv2.addWeighted(bar, 0.45, frame, 0.55, 0)
        cv2.putText(frame, f"Phase: {phase_name(f, ph)}  (frame {f}/{n - 1})",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame,
                    f"backswing={ph.backswing_start}  top={ph.downswing_start}"
                    f"  impact={ph.downswing_end}  trail-elbow={trail_label}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)
        fx = 10
        for name in t.FEATURE_REGISTRY:
            v = feats[name]
            txt = f"{name}={'n/a' if v != v else f'{v:.3f}'}"
            cv2.putText(frame, txt, (fx, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (200, 255, 200), 1)
            fx += 12 + int(8.2 * len(txt))

        writer.write(frame)
        f += 1

    cap.release()
    writer.release()
    print(f"[info] wrote {out_path}  ({f} frames, {W}x{H} @ {fps:.0f} fps)")
    print("[legend] yellow line = swing plane (ball->trail elbow); "
          "clubhead green=above / red=below; wrist path blue=backswing "
          "red=downswing; magenta=handle, orange ring=ball.")


if __name__ == "__main__":
    main()
