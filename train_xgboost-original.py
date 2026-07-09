"""
OTT (Over-The-Top) Golf Swing Detection — experimental pipeline.

Starts with a SINGLE pose-derived feature and trains an XGBoost classifier
to test whether that feature carries predictive signal.

The architecture deliberately separates three stages:
    1. Pose extraction      (video  -> per-frame landmarks)
    2. Feature computation   (landmarks -> scalar features)   <-- extend here
    3. Modelling            (features -> classifier + metrics)

Adding a new feature later means writing ONE function and registering it in
FEATURE_REGISTRY. Nothing else in the pipeline changes. See notes at bottom.

Expected layout:
    dataset/
    ├── OTT/        (videos labelled 1)
    └── NON_OTT/    (videos labelled 0)
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)
import xgboost as xgb
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATASET_DIR = "dataset"
CLASS_DIRS = {"OTT": 1, "NON_OTT": 0}
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV")

# Pose extraction (MediaPipe) is the slow stage. Raw landmarks are cached to
# disk so re-runs (e.g. after adding a feature) skip it entirely. Set USE_CACHE
# to False, or delete CACHE_DIR, to force re-extraction.
CACHE_DIR = "pose_cache"
USE_CACHE = True

# Club tracking via a YOLO11 detect model (classes: ball, club handle, club head).
# Detections are cached per video just like the pose landmarks. Set USE_CLUB to
# False to skip club extraction/features entirely.
CLUB_WEIGHTS = "best (1).pt"
CLUB_CACHE_DIR = "club_cache"
USE_CLUB = True
CLUB_IMGSZ = 320          # 320 matches 640 detection on these clips, ~3x faster
CLUB_CONF = 0.25          # detection confidence threshold

# Handedness of the golfers. Lead wrist = left wrist for a right-handed golfer.
# If your dataset is mixed, see the notes at the bottom on auto-detecting this.
LEAD_HAND = "right"   # 'right' -> lead wrist is LEFT; 'left' -> lead wrist is RIGHT

RANDOM_STATE = 42
TEST_SIZE = 0.30


# --------------------------------------------------------------------------- #
# Stage 1: Pose extraction
# --------------------------------------------------------------------------- #
# MediaPipe is imported lazily inside extract_pose_sequence so the rest of the
# pipeline (feature math, modelling) can run/test without it loaded eagerly.
def _get_mp_pose():
    import mediapipe as mp
    return mp.solutions.pose

# MediaPipe Pose landmark indices we care about
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28


@dataclass
class SwingSequence:
    """Per-frame pose landmarks for one swing video.

    landmarks: array of shape (n_frames, 33, 4) -> (x, y, z, visibility),
    in normalised image coordinates (0..1). NaN rows mean no pose was
    detected in that frame.
    """
    video_name: str
    landmarks: np.ndarray
    club: Optional[np.ndarray] = None      # (n_frames, 3, 3): [x, y, conf] per
    #                                        class [ball, handle, head], or None

    @property
    def lead_wrist_idx(self) -> int:
        return L_WRIST if LEAD_HAND == "right" else R_WRIST


def extract_pose_sequence(video_path: str,
                          min_detection_confidence: float = 0.5,
                          min_tracking_confidence: float = 0.5) -> SwingSequence:
    """Run MediaPipe Pose over a video and return all per-frame landmarks."""
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []

    mp_pose = _get_mp_pose()
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                lm = np.array(
                    [[p.x, p.y, p.z, p.visibility]
                     for p in result.pose_landmarks.landmark],
                    dtype=np.float32,
                )
            else:
                lm = np.full((33, 4), np.nan, dtype=np.float32)
            frames.append(lm)

    cap.release()

    if not frames:
        # Unreadable video -> empty sequence, handled downstream
        landmarks = np.full((0, 33, 4), np.nan, dtype=np.float32)
    else:
        landmarks = np.stack(frames, axis=0)

    return SwingSequence(video_name=os.path.basename(video_path),
                         landmarks=landmarks)


# --------------------------------------------------------------------------- #
# Landmark cache
# --------------------------------------------------------------------------- #
# One .npy file per video holds its raw (n_frames, 33, 4) landmark array. On a
# re-run we reload that instead of calling MediaPipe again, so iterating on
# FEATURES only redoes the cheap numpy stage. A cache entry is reused only if it
# is at least as new as its source video (so re-encoding a clip invalidates it).
def _cache_path(video_path: str) -> str:
    parent = os.path.basename(os.path.dirname(video_path))   # OTT / NON_OTT
    base = os.path.basename(video_path)
    return os.path.join(CACHE_DIR, f"{parent}__{base}.npy")


def load_or_extract_pose(video_path: str) -> tuple[SwingSequence, bool]:
    """Return (sequence, cache_hit). Runs MediaPipe only on a cache miss."""
    name = os.path.basename(video_path)
    cpath = _cache_path(video_path)

    if (USE_CACHE and os.path.exists(cpath)
            and os.path.getmtime(cpath) >= os.path.getmtime(video_path)):
        landmarks = np.load(cpath)
        return SwingSequence(video_name=name, landmarks=landmarks), True

    seq = extract_pose_sequence(video_path)

    # Only cache real results, so a transient read failure can be retried.
    if USE_CACHE and seq.landmarks.shape[0] > 0:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cpath, seq.landmarks)

    return seq, False


# --------------------------------------------------------------------------- #
# Club tracking (YOLO detect)
# --------------------------------------------------------------------------- #
# A YOLO11 detect model finds the ball, club handle (grip) and club head per
# frame. From handle+head we get the shaft line; the head gives the true
# clubhead path. Detections are cached to disk (one .npy per video) like the
# pose landmarks, so the model runs once. It loads lazily on a cache miss and
# uses the GPU when available.
BALL, HANDLE, HEAD = 0, 1, 2            # YOLO class indices

_CLUB_MODEL = None


def _get_club_model():
    global _CLUB_MODEL
    if _CLUB_MODEL is None:
        from ultralytics import YOLO
        _CLUB_MODEL = YOLO(CLUB_WEIGHTS)
    return _CLUB_MODEL


def _club_device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def extract_club_track(video_path: str) -> np.ndarray:
    """Run the YOLO detector over a video, one frame at a time.

    Returns shape (n_frames, 3, 3): for each class [ball, handle, head] the
    highest-confidence detection's normalised centre (x, y) and confidence, or
    NaN when that class isn't detected. Frame-by-frame inference keeps GPU memory
    bounded and avoids a pathological batched-NMS slowdown on blurry frames.
    """
    cap = cv2.VideoCapture(video_path)
    model = _get_club_model()
    dev = _club_device()
    rows: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row = np.full((3, 3), np.nan, dtype=np.float32)
        r = model.predict(frame, verbose=False, device=dev,
                          imgsz=CLUB_IMGSZ, conf=CLUB_CONF, max_det=10)[0]
        b = r.boxes
        if b is not None and len(b):
            cls = b.cls.cpu().numpy().astype(int)
            conf = b.conf.cpu().numpy()
            xywhn = b.xywhn.cpu().numpy()       # (k, 4): xc, yc, w, h (norm)
            for c in (BALL, HANDLE, HEAD):
                mk = cls == c
                if mk.any():
                    ci = int(np.argmax(np.where(mk, conf, -1.0)))   # best of c
                    row[c, :2] = xywhn[ci, :2]
                    row[c, 2] = conf[ci]
        rows.append(row)
    cap.release()
    if not rows:
        return np.full((0, 3, 3), np.nan, dtype=np.float32)
    return np.stack(rows, axis=0)


def _club_cache_path(video_path: str) -> str:
    parent = os.path.basename(os.path.dirname(video_path))
    base = os.path.basename(video_path)
    return os.path.join(CLUB_CACHE_DIR, f"{parent}__{base}.npy")


def load_or_extract_club(video_path: str) -> tuple[np.ndarray, bool]:
    """Return (club_track, cache_hit). Runs YOLO only on a cache miss."""
    cpath = _club_cache_path(video_path)
    if (USE_CLUB and os.path.exists(cpath)
            and os.path.getmtime(cpath) >= os.path.getmtime(video_path)):
        return np.load(cpath), True

    track = extract_club_track(video_path)
    if USE_CLUB and track.shape[0] > 0:
        os.makedirs(CLUB_CACHE_DIR, exist_ok=True)
        np.save(cpath, track)
    return track, False


def _interp_nan_capped(y: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN gaps up to `max_gap` frames long; leave longer gaps
    (and the un-tracked head/tail) as NaN so we never fabricate a long path."""
    n = len(y)
    valid = np.isfinite(y)
    if int(valid.sum()) < 2:
        return y.astype(float).copy()
    idx = np.arange(n)
    out = np.interp(idx, idx[valid], y[valid])
    first, last = idx[valid][0], idx[valid][-1]
    out[:first] = np.nan
    out[last + 1:] = np.nan
    invalid = ~valid
    i = first
    while i <= last:
        if invalid[i]:
            j = i
            while j <= last and invalid[j]:
                j += 1
            if (j - i) > max_gap:           # gap too long -> don't invent a path
                out[i:j] = np.nan
            i = j
        else:
            i += 1
    return out


def _smooth_nan(y: np.ndarray, window: int) -> np.ndarray:
    """Moving average that ignores NaNs and preserves the input NaN pattern."""
    if window <= 1:
        return y
    finite = np.isfinite(y)
    yf = np.where(finite, y, 0.0)
    kernel = np.ones(window)
    num = np.convolve(yf, kernel, mode="same")
    den = np.convolve(finite.astype(float), kernel, mode="same")
    out = np.where(den > 0, num / den, np.nan)
    out[~finite] = np.nan
    return out


def _clean_club_track(track: np.ndarray, max_gap: int = 5,
                      window: int = 3) -> np.ndarray:
    """Gap-fill (short gaps only) and lightly smooth the per-class x, y of a club
    track, leaving confidences untouched. Cuts detection jitter/dropout so the
    clubhead path is cleaner; long dropouts stay NaN and are simply skipped."""
    if track.shape[0] == 0:
        return track
    out = track.copy()
    for c in (BALL, HANDLE, HEAD):
        for d in (0, 1):                    # x, y channels
            filled = _interp_nan_capped(out[:, c, d], max_gap)
            out[:, c, d] = _smooth_nan(filled, window)
    return out


# --------------------------------------------------------------------------- #
# Stage 2: Feature computation  <-- THIS IS WHERE YOU EXTEND
# --------------------------------------------------------------------------- #
# Each feature is a function: SwingSequence -> float.
# Register it in FEATURE_REGISTRY and it automatically flows through the rest
# of the pipeline (dataset building, training, importance plots).

FeatureFn = Callable[[SwingSequence], float]


def _xy(landmarks: np.ndarray, idx: int) -> np.ndarray:
    """Return (n_frames, 2) array of (x, y) for one landmark, NaNs preserved."""
    return landmarks[:, idx, :2]


def _interp_nan(y: np.ndarray) -> Optional[np.ndarray]:
    """Linearly interpolate over NaN gaps so the signal has no holes.

    Returns None if fewer than two valid samples exist (nothing to build on).
    """
    n = len(y)
    valid = ~np.isnan(y)
    if int(valid.sum()) < 2:
        return None
    idx = np.arange(n)
    return np.interp(idx, idx[valid], y[valid])


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving-average smoother that preserves the input length."""
    if window <= 1:
        return x
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(xp, kernel, mode="valid")[:len(x)]


@dataclass
class SwingPhases:
    """Frame indices marking the swing's vertical-motion turning points.

    Derived purely from lead-wrist height over time. Any index may be None if
    the rise -> drop -> rise pattern cannot be found.

        backswing_start  first frame where the hands begin RISING   (address)
        downswing_start  first frame where the hands begin DROPPING  (top)
        downswing_end    first frame where the hands RISE AGAIN      (impact)

    The downswing is the dropping segment: frames [downswing_start, downswing_end].
    """
    backswing_start: Optional[int]
    downswing_start: Optional[int]
    downswing_end: Optional[int]

    @property
    def has_downswing(self) -> bool:
        return self.downswing_start is not None


def _first_sustained(sign: np.ndarray, target: int,
                     start: int, min_run: int) -> Optional[int]:
    """Index of the first run of `min_run` consecutive `target` values."""
    n = len(sign)
    i = max(start, 0)
    while i <= n - min_run:
        if np.all(sign[i:i + min_run] == target):
            return i
        i += 1
    return None


def detect_swing_phases(seq: SwingSequence,
                        smooth_window: int = 5,
                        deadband_frac: float = 0.02,
                        min_run: int = 2) -> SwingPhases:
    """Simple phase detection from lead-wrist vertical motion.

    Through a swing the hands trace a rise (backswing) -> drop (downswing) ->
    rise (follow-through) pattern. We smooth the wrist-height signal, take the
    sign of its frame-to-frame change (with a small deadband to ignore jitter),
    and record the first frame of each monotonic segment.

    Image y grows downward, so wrist height is taken as -y (up = larger).
    """
    lm = seq.landmarks
    if lm.shape[0] == 0:
        return SwingPhases(None, None, None)

    wrist_y = _xy(lm, seq.lead_wrist_idx)[:, 1]
    y = _interp_nan(wrist_y)
    if y is None:
        return SwingPhases(None, None, None)

    height = _moving_average(-y, smooth_window)        # up = larger
    dv = np.diff(height)
    span = float(np.max(height) - np.min(height))
    thresh = deadband_frac * span if span > 0 else 0.0
    sign = np.where(dv > thresh, 1, np.where(dv < -thresh, -1, 0))

    backswing_start = _first_sustained(sign, 1, 0, min_run)
    if backswing_start is None:
        return SwingPhases(None, None, None)
    downswing_start = _first_sustained(sign, -1, backswing_start + 1, min_run)
    if downswing_start is None:
        return SwingPhases(backswing_start, None, None)
    downswing_end = _first_sustained(sign, 1, downswing_start + 1, min_run)
    return SwingPhases(backswing_start, downswing_start, downswing_end)


def _body_scale(lm: np.ndarray) -> float:
    """Robust body-height scale (normalised image units) for one swing.

    Uses the shoulder_mid -> ankle_mid distance (~full standing height), which
    — unlike shoulder WIDTH — does not collapse in down-the-line camera views.
    Where the ankles aren't tracked it approximates from torso length
    (shoulder_mid -> hip_mid, ~2.7x shorter). The MEDIAN over frames is returned
    so a few bad frames can't distort the scale.
    """
    sh = (_xy(lm, L_SHOULDER) + _xy(lm, R_SHOULDER)) / 2.0
    hip = (_xy(lm, L_HIP) + _xy(lm, R_HIP)) / 2.0
    ank = (_xy(lm, L_ANKLE) + _xy(lm, R_ANKLE)) / 2.0

    full = np.linalg.norm(sh - ank, axis=1)            # shoulder -> ankle
    torso = np.linalg.norm(sh - hip, axis=1)           # shoulder -> hip
    scale = np.where(np.isnan(full), 2.7 * torso, full)

    scale = scale[np.isfinite(scale) & (scale > 1e-6)]
    if scale.size == 0:
        return np.nan
    return float(np.median(scale))


def feat_out_to_in_hand_path(seq: SwingSequence) -> float:
    """
    Out-to-in ("over the top") hand-path measure: how far OUTSIDE its own
    backswing the lead hand travels on the way down.

    The hand traces a loop — up on the backswing, down on the downswing. We
    compare the two paths at matched HEIGHTS and average the horizontal gap
    (downswing minus backswing), normalised by body height:

        gap(h)  = x_down(h) - x_up(h)        # at the same height h
        feature = mean over the overlapping height range / body_scale

    The horizontal axis is oriented by the backswing direction, which makes the
    sign independent of camera side and handedness: a mirror flips both the
    backswing direction and the hand x, so the gap is preserved. An over-the-top
    swing has the downswing path displaced to one consistent side of the
    backswing path.

    LIMITATION: this is a 2D projection, meaningful for DOWN-THE-LINE footage;
    face-on clips can't show out-to-in and contribute noise. Returns NaN if a
    clean up-then-down loop can't be found.
    """
    lm = seq.landmarks
    if lm.shape[0] == 0:
        return np.nan
    phases = detect_swing_phases(seq)
    if phases.backswing_start is None or phases.downswing_start is None:
        return np.nan
    bs, top = phases.backswing_start, phases.downswing_start
    imp = (phases.downswing_end if phases.downswing_end is not None
           else lm.shape[0] - 1)
    body_scale = _body_scale(lm)
    if not np.isfinite(body_scale) or body_scale <= 1e-6:
        return np.nan
    return _out_to_in_gap(_xy(lm, seq.lead_wrist_idx), bs, top, imp, body_scale)


def _out_to_in_gap(pts: np.ndarray, bs: int, top: int, imp: int,
                   body_scale: float, exp_k: float = 15.0) -> float:
    """Matched-height horizontal gap (down-path minus up-path) of a tracked
    point through the swing loop, oriented by the backswing direction and
    normalised by body height. Mirror-invariant. NaN if no clean loop.

    The per-height gaps are combined with an EXPONENTIAL weighting (exp_k): the
    most extreme out-to-in moments count far more than a swing that is only
    mildly off throughout. The emphasis has to happen here, across heights —
    a tree model is invariant to a monotonic transform of the final scalar, so
    weighting only changes the ranking between swings when done before the
    reduction. exp_k=0 recovers a plain mean; ~15 matched the best CV.

    Shared by the lead-hand and clubhead out-to-in features.
    """
    up = pts[bs:top + 1]                    # rising  (backswing)
    down = pts[top:imp + 1]                # dropping (downswing)
    up = up[~np.isnan(up).any(axis=1)]
    down = down[~np.isnan(down).any(axis=1)]
    if len(up) < 3 or len(down) < 3:
        return np.nan

    # Orient the horizontal axis by the backswing direction (mirror-invariant).
    k = max(1, len(up) // 5)
    s = np.sign(np.mean(up[-k:, 0]) - np.mean(up[:k, 0]))
    if s == 0:
        return np.nan

    xu, hu = s * up[:, 0], -up[:, 1]        # height = -y (up is larger)
    xd, hd = s * down[:, 0], -down[:, 1]

    # Sort by height so np.interp gets nondecreasing sample points.
    ou, od = np.argsort(hu), np.argsort(hd)
    hu, xu = hu[ou], xu[ou]
    hd, xd = hd[od], xd[od]

    lo = max(hu.min(), hd.min())
    hi = min(hu.max(), hd.max())
    if hi - lo <= 1e-9:                     # paths don't overlap in height
        return np.nan

    heights = np.linspace(lo, hi, 25)
    gap = (np.interp(heights, hd, xd) - np.interp(heights, hu, xu)) / body_scale
    w = np.exp(np.abs(gap) * exp_k)
    return float(np.sum(gap * w) / np.sum(w))


def feat_early_downswing_out_vs_down(seq: SwingSequence) -> float:
    """
    Out-vs-down hand move at the START of the downswing: how far the lead hand
    travels OUTWARD (away from the body, toward the ball side) relative to how
    far it drops, over the first half of the downswing.

        ratio = outward_displacement / |downward_displacement|

    From the coaching cue: a good move drops the hands almost straight DOWN into
    the slot (small outward, large drop -> low ratio), while an over-the-top move
    throws them OUT and over the top (large outward -> high ratio).

    The horizontal axis is oriented by the backswing direction, so the sign is
    mirror/handedness-invariant, and the ratio is dimensionless (no body scale
    needed). NaN if the downswing or a clean backswing direction can't be found.
    """
    lm = seq.landmarks
    if lm.shape[0] == 0:
        return np.nan
    phases = detect_swing_phases(seq)
    if phases.backswing_start is None or phases.downswing_start is None:
        return np.nan
    bs, top = phases.backswing_start, phases.downswing_start
    imp = (phases.downswing_end if phases.downswing_end is not None
           else lm.shape[0] - 1)
    if imp <= top:
        return np.nan

    pts = _xy(lm, seq.lead_wrist_idx)
    x = _interp_nan(pts[:, 0])
    y = _interp_nan(pts[:, 1])
    if x is None or y is None:
        return np.nan

    # Orient the horizontal axis by the backswing direction (mirror-invariant).
    up = pts[bs:top + 1]
    up = up[~np.isnan(up).any(axis=1)]
    if len(up) < 3:
        return np.nan
    k = max(1, len(up) // 5)
    s = np.sign(np.mean(up[-k:, 0]) - np.mean(up[:k, 0]))
    if s == 0:
        return np.nan

    half = top + max(1, (imp - top) // 2)
    outward = s * (x[half] - x[top])
    down = y[half] - y[top]                 # image y grows down, so +ve = a drop
    return float(outward / (abs(down) + 1e-9))


def feat_clubhead_out_to_in(seq: SwingSequence) -> float:
    """
    Out-to-in path of the actual CLUBHEAD (from YOLO) — the most direct
    over-the-top signal. Same matched-height loop as the lead-hand version, but
    on the clubhead, which is what literally comes "over the top". NaN if there
    is no club track or no clean loop.
    """
    lm = seq.landmarks
    if seq.club is None or lm.shape[0] == 0:
        return np.nan
    phases = detect_swing_phases(seq)
    if phases.backswing_start is None or phases.downswing_start is None:
        return np.nan
    bs, top = phases.backswing_start, phases.downswing_start
    imp = (phases.downswing_end if phases.downswing_end is not None
           else lm.shape[0] - 1)
    body_scale = _body_scale(lm)
    if not np.isfinite(body_scale) or body_scale <= 1e-6:
        return np.nan
    return _out_to_in_gap(seq.club[:, HEAD, :2], bs, top, imp, body_scale)


def feat_shaft_pitch_early_downswing(seq: SwingSequence) -> float:
    """
    Shaft pitch in the first half of the downswing: how high the CLUBHEAD (YOLO)
    sits above the LEAD WRIST (MediaPipe), normalised by body height — i.e. the
    shaft treated as the hand -> clubhead line. An over-the-top move keeps the
    clubhead UP / shaft steep early; a shallowing move drops it behind and down.

        per-frame = (wrist_y - head_y) / body_scale   # +ve: head above the hand
        feature   = mean over the first half of the downswing

    The lead wrist replaces the YOLO club-handle here: the handle is detected
    less reliably (often missed), whereas the pose wrist is almost always
    tracked, which made this feature markedly stronger and more stable in CV.
    """
    lm = seq.landmarks
    if seq.club is None or lm.shape[0] == 0:
        return np.nan
    phases = detect_swing_phases(seq)
    if phases.downswing_start is None:
        return np.nan
    top = phases.downswing_start
    imp = (phases.downswing_end if phases.downswing_end is not None
           else lm.shape[0] - 1)
    if imp <= top:
        return np.nan
    body_scale = _body_scale(lm)
    if not np.isfinite(body_scale) or body_scale <= 1e-6:
        return np.nan

    half = top + max(1, (imp - top) // 2)
    head_y = seq.club[top:half, HEAD, 1]
    wrist_y = _xy(lm, seq.lead_wrist_idx)[top:half, 1]
    vals = (wrist_y - head_y) / body_scale
    if np.all(np.isnan(vals)):
        return np.nan
    return float(np.nanmean(vals))


def plane_line_at_top(seq: SwingSequence):
    """The swing-plane line (ball -> trail elbow) FIXED at the top of the swing.

    Returns (bx, by, ex, ey): the stationary ball reference and the trail-elbow
    position averaged around the top of the backswing. The line is fixed once,
    at the start of the downswing, and held constant — using the per-frame elbow
    instead makes the line jitter as the arm moves, which added noise in CV.
    Returns None if the ball, elbow or downswing can't be located.

    Shared by feat_clubhead_above_plane and the demo visualiser so both draw and
    measure against exactly the same line.
    """
    lm = seq.landmarks
    if seq.club is None or lm.shape[0] == 0:
        return None
    phases = detect_swing_phases(seq)
    if phases.downswing_start is None:
        return None
    top = phases.downswing_start

    ball = seq.club[:, BALL, :2]
    # Stationary reference = median detected position up to impact.
    imp = (phases.downswing_end if phases.downswing_end is not None
           else lm.shape[0] - 1)
    ball = ball[:imp + 1]
    ball = ball[~np.isnan(ball).any(axis=1)]
    if len(ball) == 0:
        return None
    bx, by = float(np.median(ball[:, 0])), float(np.median(ball[:, 1]))

    # Trail elbow = same side as the higher shoulder at the top, position taken
    # as the mean over a small window around the top (then held fixed).
    a, b = max(0, top - 3), min(lm.shape[0], top + 4)
    lsy = np.nanmean(_xy(lm, L_SHOULDER)[a:b, 1])
    rsy = np.nanmean(_xy(lm, R_SHOULDER)[a:b, 1])
    if not (np.isfinite(lsy) and np.isfinite(rsy)):
        return None
    elbow = _xy(lm, R_ELBOW if rsy < lsy else L_ELBOW)
    ex = np.nanmean(elbow[a:b, 0])
    ey = np.nanmean(elbow[a:b, 1])
    if not (np.isfinite(ex) and np.isfinite(ey)):
        return None
    return bx, by, float(ex), float(ey)


def feat_clubhead_above_plane(seq: SwingSequence, exp_k: float = 8.0) -> float:
    """
    How far ABOVE the swing-plane line the CLUBHEAD travels through the downswing
    (signed, normalised by body height). The plane line (ball -> trail elbow) is
    FIXED at the top of the swing; for each downswing frame we take the signed
    height of the clubhead relative to that line at the head's x (+ve = above).
    A good, shallowing move drops the clubhead BELOW into impact; an over-the-top
    move keeps it ABOVE — so a higher value means more over-the-top.

    The per-frame distances are combined with an EXPONENTIAL weighting (exp_k),
    so the most extreme above/below moments dominate over a swing that is only
    mildly off. Mirror-invariant (compares heights at the clubhead's own x, so it
    works for left- and right-handers). NaN if the ball, clubhead or elbow can't
    be located.
    """
    lm = seq.landmarks
    line = plane_line_at_top(seq)
    if line is None:
        return np.nan
    bx, by, ex, ey = line
    dx = ex - bx
    if abs(dx) < 1e-6:                      # near-vertical line: undefined
        return np.nan

    phases = detect_swing_phases(seq)
    top = phases.downswing_start
    imp = (phases.downswing_end if phases.downswing_end is not None
           else lm.shape[0] - 1)
    body_scale = _body_scale(lm)
    if not np.isfinite(body_scale) or body_scale <= 1e-6:
        return np.nan
    head = seq.club[:, HEAD, :2]

    dists = []
    for f in range(top, imp + 1):
        hx, hy = head[f]
        if not np.all(np.isfinite([hx, hy])):
            continue
        line_y = by + (ey - by) / dx * (hx - bx)   # plane height at clubhead x
        dists.append((line_y - hy) / body_scale)   # +ve = clubhead above line
    if not dists:
        return np.nan
    d = np.array(dists)
    # Exponential weighting, computed in a numerically stable (softmax) way.
    z = np.abs(d) * exp_k
    w = np.exp(z - z.max())
    return float(np.sum(d * w) / np.sum(w))


def feat_trail_shoulder_above_lead(seq: SwingSequence) -> float:
    """
    How HIGH the trail shoulder sits relative to the lead shoulder through the
    downswing, normalised by body height. From the coaching cue: in an
    over-the-top move the trail (back) shoulder STAYS UP and goes out, whereas a
    good move drops it DOWN and under the chin into impact.

        per-frame = (lead_shoulder_y - trail_shoulder_y) / body_scale
        feature   = mean over the downswing

    Image y grows downward, so a HIGHER trail shoulder gives a POSITIVE value
    (over-the-top); a trail shoulder that drops below the lead gives NEGATIVE.

    The trail shoulder is identified per swing as the higher shoulder at the top
    of the backswing, so the feature is handedness-invariant (works for lefties).
    Unlike shoulder ROTATION (which happens in depth and foreshortens), this is a
    VERTICAL measure that projects cleanly in down-the-line video. NaN if the
    downswing can't be found.
    """
    lm = seq.landmarks
    n = lm.shape[0]
    if n == 0:
        return np.nan

    phases = detect_swing_phases(seq)
    if phases.downswing_start is None:
        return np.nan
    top = phases.downswing_start
    imp = phases.downswing_end if phases.downswing_end is not None else n - 1
    if imp <= top:
        return np.nan

    body_scale = _body_scale(lm)
    if not np.isfinite(body_scale) or body_scale <= 1e-6:
        return np.nan

    ls_y = _xy(lm, L_SHOULDER)[:, 1]
    rs_y = _xy(lm, R_SHOULDER)[:, 1]

    # Identify the trail shoulder = the higher one (smaller y) around the top.
    a, b = max(0, top - 3), min(n, top + 4)
    left_top, right_top = np.nanmean(ls_y[a:b]), np.nanmean(rs_y[a:b])
    if not (np.isfinite(left_top) and np.isfinite(right_top)):
        return np.nan
    if right_top < left_top:                # right shoulder higher -> it's trail
        trail_y, lead_y = rs_y, ls_y
    else:
        trail_y, lead_y = ls_y, rs_y

    vals = (lead_y[top:imp + 1] - trail_y[top:imp + 1]) / body_scale
    if np.all(np.isnan(vals)):
        return np.nan
    return float(np.nanmean(vals))


# The registry. To add a feature: write a function above, add it here. Done.
#
# History: a leave-one-feature-out ablation trimmed the original 5 features to
# 3 (dropping trail_shoulder_above_lead and clubhead_out_to_in, which didn't
# stack — removing either *improved* CV). Their feat_* functions are kept above
# so they can be re-registered if club/ball detection improves.
#
# Then early_downswing_out_vs_down was added (4th) and out_to_in's averaging
# switched to exponential weighting (see _out_to_in_gap). In experiments this
# lifted 5-fold CV from ~0.84 to ~0.91 with about half the std — both changes
# came from the coaching cue that an over-the-top move throws the hands OUT,
# while a good move drops them straight DOWN into the slot.
#
# Three further refinements took CV to ~0.93 +/- 0.02: clubhead_above_plane now
# fixes the plane line at the top (instead of a per-frame elbow) and exp-weights
# the signed above/below distance; shaft_pitch_early_downswing measures the
# clubhead against the LEAD WRIST (reliable) instead of the YOLO handle (noisy).
FEATURE_REGISTRY: Dict[str, FeatureFn] = {
    "out_to_in_hand_path": feat_out_to_in_hand_path,
    "early_downswing_out_vs_down": feat_early_downswing_out_vs_down,
    "shaft_pitch_early_downswing": feat_shaft_pitch_early_downswing,
    "clubhead_above_plane": feat_clubhead_above_plane,
}


def compute_features(seq: SwingSequence) -> Dict[str, float]:
    """Run every registered feature function over one swing."""
    return {name: fn(seq) for name, fn in FEATURE_REGISTRY.items()}


# --------------------------------------------------------------------------- #
# Dataset building
# --------------------------------------------------------------------------- #
def find_videos(dataset_dir: str) -> List[tuple[str, int]]:
    """Return list of (video_path, label) across all class folders.

    Paths are de-duplicated by normalised absolute path so a file is never
    counted twice. This matters on Windows, where globbing is case-insensitive
    and patterns like '*.mp4' and '*.MP4' otherwise both match the same files.
    """
    items: List[tuple[str, int]] = []
    seen: set[str] = set()
    for class_name, label in CLASS_DIRS.items():
        folder = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(folder):
            print(f"[warn] missing class folder: {folder}")
            continue
        for ext in VIDEO_EXTS:
            for path in glob.glob(os.path.join(folder, f"*{ext}")):
                key = os.path.normcase(os.path.abspath(path))
                if key in seen:
                    continue
                seen.add(key)
                items.append((path, label))
    return items


def build_dataset(dataset_dir: str = DATASET_DIR) -> pd.DataFrame:
    """Process every video into one row of features + label."""
    videos = find_videos(dataset_dir)
    if not videos:
        raise RuntimeError(f"No videos found under {dataset_dir}/")

    rows: List[dict] = []
    for i, (path, label) in enumerate(videos, 1):
        name = os.path.basename(path)
        print(f"[{i}/{len(videos)}] processing {name} ...")
        seq, cached = load_or_extract_pose(path)
        if cached:
            print(f"    [cache] loaded landmarks from {CACHE_DIR}/")

        if USE_CLUB and os.path.exists(CLUB_WEIGHTS):
            club, club_cached = load_or_extract_club(path)
            if club_cached:
                print(f"    [cache] loaded club track from {CLUB_CACHE_DIR}/")
            # Keep pose & club frame-aligned (both read every frame, but guard).
            if (seq.landmarks.shape[0] and club.shape[0]
                    and club.shape[0] != seq.landmarks.shape[0]):
                L = min(club.shape[0], seq.landmarks.shape[0])
                seq.landmarks, club = seq.landmarks[:L], club[:L]
            seq.club = _clean_club_track(club)

        feats = compute_features(seq)

        n_detected = int(np.sum(~np.isnan(seq.landmarks[:, 0, 0]))) \
            if seq.landmarks.shape[0] else 0
        if n_detected == 0:
            print(f"    [warn] no pose detected in {name}")

        rows.append({"video_name": name, **feats, "label": label})

    df = pd.DataFrame(rows)
    feature_cols = list(FEATURE_REGISTRY.keys())

    # Drop rows where every feature is NaN (unreadable / no-pose videos)
    before = len(df)
    df = df.dropna(subset=feature_cols, how="all").reset_index(drop=True)
    if len(df) < before:
        print(f"[info] dropped {before - len(df)} video(s) with no usable pose")

    # Impute any remaining per-feature NaNs with that feature's median
    for col in feature_cols:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            print(f"[info] imputed NaNs in '{col}' with median={med:.4f}")

    return df


# --------------------------------------------------------------------------- #
# Stage 3: Modelling
# --------------------------------------------------------------------------- #
def _make_model() -> xgb.XGBClassifier:
    """A fresh classifier with the project's standard hyper-parameters."""
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )


def cross_validate(df: pd.DataFrame, n_splits: int = 5):
    """Stratified k-fold CV ROC-AUC — the trustworthy metric at this data size.

    Reports each feature ALONE and the full combined set, so you can see whether
    combining features actually beats the best single feature, rather than
    chasing the noise of a single train/test split.
    """
    feature_cols = list(FEATURE_REGISTRY.keys())
    y = df["label"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)

    def cv_auc(cols: List[str]) -> np.ndarray:
        X = df[cols].values
        aucs: List[float] = []
        for tr, te in skf.split(X, y):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = _make_model()
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        return np.array(aucs)

    print(f"\n=========== {n_splits}-fold CV ROC-AUC (mean +/- std) ===========")
    if len(feature_cols) > 1:
        for col in feature_cols:
            a = cv_auc([col])
            print(f"  {col:28s}: {a.mean():.3f} +/- {a.std():.3f}")
    combined = cv_auc(feature_cols)
    label = "ALL FEATURES" if len(feature_cols) > 1 else feature_cols[0]
    print(f"  {label:28s}: {combined.mean():.3f} +/- {combined.std():.3f}")
    print("==================================================\n")
    return combined


def train_and_evaluate(df: pd.DataFrame):
    feature_cols = list(FEATURE_REGISTRY.keys())
    X = df[feature_cols].values
    y = df["label"].values

    # stratify keeps class balance in both splits — important for small sets
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    model = _make_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }
    # ROC AUC needs both classes present in y_test
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    print("\n================ Test metrics ================")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.3f}")
    print("==============================================\n")

    return model, metrics, (X_test, y_test, y_proba, idx_test)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(df: pd.DataFrame, model, eval_bundle, out_prefix="ott"):
    feature_cols = list(FEATURE_REGISTRY.keys())
    X_test, y_test, y_proba, _ = eval_bundle

    # 1. Feature distribution(s): OTT vs Non-OTT
    for col in feature_cols:
        plt.figure(figsize=(7, 4))
        for label, name, color in [(1, "OTT", "tab:red"),
                                    (0, "Non-OTT", "tab:blue")]:
            subset = df.loc[df["label"] == label, col].dropna()
            if len(subset):
                plt.hist(subset, bins=15, alpha=0.55, label=name, color=color)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"Feature distribution: {col}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_dist_{col}.png", dpi=120)
        plt.close()

    # 2. Feature importance
    plt.figure(figsize=(7, 4))
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.bar([feature_cols[i] for i in order], importances[order],
            color="tab:green")
    plt.ylabel("importance")
    plt.title("XGBoost feature importance")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_importance.png", dpi=120)
    plt.close()

    # 3. ROC curve
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="tab:purple", label=f"ROC (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_roc.png", dpi=120)
        plt.close()
    else:
        print("[warn] only one class in test set — skipping ROC plot")

    # 4. Confusion matrix (at the 0.5 decision threshold)
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    class_names = ["Non-OTT", "OTT"]
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(label="count")
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_confusion_matrix.png", dpi=120)
    plt.close()

    print(f"[info] plots saved with prefix '{out_prefix}_'")


def print_example_predictions(df, model, eval_bundle, n=8):
    feature_cols = list(FEATURE_REGISTRY.keys())
    _, y_test, y_proba, idx_test = eval_bundle

    print("\n---------------- Example predictions ----------------")
    print(f"{'video':35s} {'true':>5s} {'pred':>5s} {'p(OTT)':>8s}")
    for j in range(min(n, len(idx_test))):
        row = df.iloc[idx_test[j]]
        pred = int(y_proba[j] >= 0.5)
        true = int(y_test[j])
        flag = "" if pred == true else "  <-- miss"
        print(f"{row['video_name'][:35]:35s} {true:>5d} {pred:>5d} "
              f"{y_proba[j]:>8.3f}{flag}")
    print("-----------------------------------------------------\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("Building dataset from videos...")
    df = build_dataset(DATASET_DIR)

    df.to_csv("ott_features.csv", index=False)
    print(f"\nSaved feature table -> ott_features.csv  ({len(df)} videos)")
    print(df.to_string(index=False))

    if df["label"].nunique() < 2:
        print("\n[error] need both OTT and Non-OTT videos to train. Stopping.")
        return
    if len(df) < 6:
        print(f"\n[warn] only {len(df)} videos — metrics will be very noisy. "
              "Treat results as directional only.")

    cross_validate(df)
    model, metrics, eval_bundle = train_and_evaluate(df)
    make_plots(df, model, eval_bundle)
    print_example_predictions(df, model, eval_bundle)


if __name__ == "__main__":
    main()