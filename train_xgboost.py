"""Train an XGBoost classifier to distinguish professional vs. amateur golf swings.

This script does NOT reimplement pose estimation. It reuses the existing repo
pipeline:

    MediaPipe_class.MediaPipe_PoseEstimation  -> per-frame CSV of angles/keypoints
    process_swing.DataProcessor               -> reduces a swing to 3 key frames
                                                 (address, top, contact)

For every video it produces a single fixed-length feature vector built from those
3 key frames (joint angles, hip/shoulder rotation, spine angle, weight shift /
centre-of-mass movement, and phase timing), then trains and evaluates an
``xgboost.XGBClassifier``.

Expected dataset layout::

    data/
      pro/
        video1.mp4
        ...
      amateur/
        video1.mp4
        ...

Run end-to-end with::

    python train_xgboost.py

Optional arguments::

    python train_xgboost.py --data-dir data --model models/xgboost_golf_model.json
    python train_xgboost.py --predict path/to/swing.mp4
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = "data"
MODEL_PATH = os.path.join("models", "xgboost_golf_model.json")
CACHE_DIR = ".pose_cache"          # per-video CSVs are cached here to avoid re-running MediaPipe
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
POSE_TIMEOUT_SEC = 300             # per-video cap on the pose-extraction subprocess

# Maps the dataset sub-folder name to a label. 1 = pro, 0 = amateur.
LABEL_MAP = {"pro": 1, "amateur": 0}

# The three key swing phases produced by DataProcessor.split_swing(), in order.
PHASES = ["address", "top", "contact"]


# --------------------------------------------------------------------------- #
# Pose CSV generation (delegates to existing repo code)
# --------------------------------------------------------------------------- #
def _cache_paths(video_path):
    """Return (cache_folder, csv_path, annotated_path) for a video, with a stable key.

    The key is a hash of the absolute path. It must be deterministic across processes
    -- Python's built-in hash() is randomised per run, so it cannot be used here or the
    cache would never be reused.
    """
    base = os.path.splitext(os.path.basename(video_path))[0]
    key = hashlib.sha1(os.path.abspath(video_path).encode("utf-8")).hexdigest()[:12]
    cache_folder = os.path.join(CACHE_DIR, f"{base}_{key}")
    return cache_folder, os.path.join(cache_folder, base + ".csv"), \
        os.path.join(cache_folder, base + "_annotated.mp4")


def _extract_one(video_path, csv_path, annotated_video):
    """Run the existing MediaPipe pipeline for a single video (used by the subprocess worker)."""
    from MediaPipe_class import MediaPipe_PoseEstimation

    estimator = MediaPipe_PoseEstimation(video_path, csv_path, annotated_video)
    estimator.process_video()


def _ensure_pose_csv(video_path, use_cache=True):
    """Return path to a folder containing exactly one per-frame pose CSV for ``video_path``.

    Runs the existing MediaPipe pipeline once and caches the CSV under CACHE_DIR so
    repeated runs (or prediction) don't re-process the same video. Each video gets
    its own sub-folder so DataProcessor only ever sees a single CSV.

    Extraction runs in an isolated subprocess: the existing pipeline does not release
    its MediaPipe Pose / OpenCV VideoWriter objects, so processing many videos in one
    process leaks resources until it crashes. A fresh process per video frees
    everything on exit and contains any native crash, so one bad video can't kill the
    whole batch -- and because the cache key is stable, a re-run resumes where it left
    off.
    """
    cache_folder, csv_path, annotated_video = _cache_paths(video_path)

    if use_cache and os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        return cache_folder

    os.makedirs(cache_folder, exist_ok=True)

    cmd = [sys.executable, os.path.abspath(__file__), "--extract-one",
           video_path, csv_path, annotated_video]
    try:
        subprocess.run(cmd, check=True, timeout=POSE_TIMEOUT_SEC,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"pose extraction timed out after {POSE_TIMEOUT_SEC}s")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pose extraction subprocess crashed (exit {exc.returncode})")

    if not (os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0):
        raise RuntimeError("pose extraction produced no CSV")

    return cache_folder


def _split_key_frames(data):
    """Reduce a per-frame swing DataFrame to its 3 key frames: address, top, contact.

    This mirrors process_swing.DataProcessor.split_swing() but uses int() instead of
    .item() on the index labels. The repo version assumes idxmax()/idxmin() return a
    numpy scalar (with .item()); on current pandas they return plain Python ints, which
    breaks that call -- so we keep the repo file untouched and use a version-safe copy
    here. Logic and frame selection are otherwise identical.
    """
    halfway_back_ind = data['right_wrist_x'].idxmin()
    halfway_front_ind = data.right_wrist_x[data.index > halfway_back_ind].idxmax()
    middle_data = data[(data.index > halfway_back_ind) & (data.index < halfway_front_ind)]

    if middle_data.empty:
        halfway_back_ind = data.right_wrist_x[data.index < halfway_back_ind].idxmin()
        halfway_front_ind = data.right_wrist_x[data.index > halfway_back_ind].idxmax()
        middle_data = data[(data.index > halfway_back_ind) & (data.index < halfway_front_ind)]

    # Ball contact: lowest wrist point (max y) during the through-swing.
    contact_frame = int(middle_data['right_wrist_y'].idxmax())
    # Top of backswing: highest wrist point (min y) before contact.
    back_data = data[data.index < contact_frame]
    top_backswing_frame = int(back_data['right_wrist_y'].idxmin())
    # Address: lowest wrist point before the swing starts.
    halfway_back_data = data[data.index < halfway_back_ind]
    address_frame = int(halfway_back_data['right_wrist_y'].idxmax())

    data = data.iloc[[max(address_frame - 4, 0), top_backswing_frame, contact_frame]]
    data['index'] = data.index
    return data.reset_index(drop=True)


def _load_key_frames(video_path, use_cache=True):
    """Return the 3-row (address/top/contact) DataFrame for a swing.

    Our MediaPipe step writes exactly one CSV per video, so we read that known CSV path
    directly rather than going through DataProcessor.load_data() -- its filename handling
    (``filename.split('.')[0]``) breaks on names containing dots (e.g. 'rapidsave.com_*'),
    which silently dropped an entire class. We still apply the same column normalisation
    and key-frame split the repo pipeline uses.
    """
    _ensure_pose_csv(video_path, use_cache=use_cache)
    _, csv_path, _ = _cache_paths(video_path)

    data = pd.read_csv(csv_path)
    data = data.reset_index()
    data.columns = [c.replace(' X', '_x').replace(' Y', '_y').lower() for c in data.columns]
    data = data.drop(['video_timestamp'], axis=1)
    return _split_key_frames(data)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _safe(value, default=0.0):
    """Coerce a possibly-missing/NaN value to a finite float."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _hip_center(row):
    return (
        (_safe(row.get("left_hip_x")) + _safe(row.get("right_hip_x"))) / 2.0,
        (_safe(row.get("left_hip_y")) + _safe(row.get("right_hip_y"))) / 2.0,
    )


def _shoulder_center(row):
    return (
        (_safe(row.get("left_shoulder_x")) + _safe(row.get("right_shoulder_x"))) / 2.0,
        (_safe(row.get("left_shoulder_y")) + _safe(row.get("right_shoulder_y"))) / 2.0,
    )


def _body_scale(address_row):
    """A resolution-independent body scale: shoulder-centre to ankle-midpoint distance.

    Pose CSVs store pixel coordinates, so positional features must be normalised by a
    per-swing body size to be comparable across videos of different resolutions.
    """
    sx, sy = _shoulder_center(address_row)
    ax = _safe(address_row.get("midpoint_x"))
    ay = _safe(address_row.get("midpoint_y"))
    scale = np.hypot(sx - ax, sy - ay)
    return scale if scale > 1e-6 else 1.0


def _features_from_key_frames(data):
    """Build a single ordered fixed-length feature dict from the 3 key-frame rows."""
    rows = {phase: data.iloc[i] for i, phase in enumerate(PHASES)}
    scale = _body_scale(rows["address"])

    feats = {}

    # ---- Per-phase joint angles (address / top / contact) -----------------
    for phase in PHASES:
        r = rows[phase]
        feats[f"shoulders_inclination_{phase}"] = _safe(r.get("shoulders_inclination"))
        feats[f"hips_inclination_{phase}"] = _safe(r.get("hips_inclination"))
        feats[f"knee_angle_{phase}"] = _safe(r.get("knee_angle"))
        feats[f"pelvis_angle_{phase}"] = _safe(r.get("pelvis_angle"))   # spine/posture proxy
        feats[f"arm_angle_{phase}"] = _safe(r.get("arm_angle"))

    # ---- Rotation: change in shoulder/hip line from address ----------------
    feats["shoulder_rotation_top"] = (
        feats["shoulders_inclination_top"] - feats["shoulders_inclination_address"]
    )
    feats["shoulder_rotation_contact"] = (
        feats["shoulders_inclination_contact"] - feats["shoulders_inclination_address"]
    )
    feats["hip_rotation_top"] = (
        feats["hips_inclination_top"] - feats["hips_inclination_address"]
    )
    feats["hip_rotation_contact"] = (
        feats["hips_inclination_contact"] - feats["hips_inclination_address"]
    )
    # X-factor: shoulder-hip separation at the top of the backswing.
    feats["x_factor_top"] = feats["shoulder_rotation_top"] - feats["hip_rotation_top"]

    # ---- Spine angle change (pelvis_angle proxy) --------------------------
    feats["spine_angle_change_top"] = (
        feats["pelvis_angle_top"] - feats["pelvis_angle_address"]
    )
    feats["spine_angle_change_contact"] = (
        feats["pelvis_angle_contact"] - feats["pelvis_angle_address"]
    )

    # ---- Weight shift / centre-of-mass movement (normalised) --------------
    addr_hip = _hip_center(rows["address"])
    top_hip = _hip_center(rows["top"])
    contact_hip = _hip_center(rows["contact"])

    feats["com_shift_x_top"] = (top_hip[0] - addr_hip[0]) / scale
    feats["com_shift_x_contact"] = (contact_hip[0] - addr_hip[0]) / scale
    feats["com_shift_y_top"] = (top_hip[1] - addr_hip[1]) / scale
    feats["com_shift_y_contact"] = (contact_hip[1] - addr_hip[1]) / scale
    feats["com_total_path"] = (
        np.hypot(top_hip[0] - addr_hip[0], top_hip[1] - addr_hip[1])
        + np.hypot(contact_hip[0] - top_hip[0], contact_hip[1] - top_hip[1])
    ) / scale

    # Stance-relative weight shift via the feet midpoint.
    addr_mid = (_safe(rows["address"].get("midpoint_x")), _safe(rows["address"].get("midpoint_y")))
    contact_mid = (_safe(rows["contact"].get("midpoint_x")), _safe(rows["contact"].get("midpoint_y")))
    feats["midpoint_shift_contact"] = (
        np.hypot(contact_mid[0] - addr_mid[0], contact_mid[1] - addr_mid[1]) / scale
    )

    # ---- Head stability (nose displacement from address) ------------------
    addr_nose = (_safe(rows["address"].get("nose_x")), _safe(rows["address"].get("nose_y")))
    for phase in ("top", "contact"):
        nx = _safe(rows[phase].get("nose_x"))
        ny = _safe(rows[phase].get("nose_y"))
        feats[f"head_movement_{phase}"] = (
            np.hypot(nx - addr_nose[0], ny - addr_nose[1]) / scale
        )

    # ---- Timing between phases (frame counts) -----------------------------
    f_addr = _safe(rows["address"].get("index"))
    f_top = _safe(rows["top"].get("index"))
    f_contact = _safe(rows["contact"].get("index"))
    backswing = max(f_top - f_addr, 0.0)
    downswing = max(f_contact - f_top, 0.0)
    feats["backswing_frames"] = backswing
    feats["downswing_frames"] = downswing
    feats["total_swing_frames"] = backswing + downswing
    # Tempo ratio (classic golf metric, backswing:downswing).
    feats["tempo_ratio"] = backswing / downswing if downswing > 1e-6 else 0.0

    return feats


def extract_features(video_path, use_cache=True):
    """Extract a single fixed-length feature vector for one swing video.

    Returns an ordered ``pandas.Series`` (feature name -> value), or ``None`` if the
    swing could not be processed (e.g. pose estimation found no clear swing).
    """
    try:
        key_frames = _load_key_frames(video_path, use_cache=use_cache)
        if key_frames is None or len(key_frames) < len(PHASES):
            print(f"  [skip] {os.path.basename(video_path)}: could not isolate 3 swing phases")
            return None
        feats = _features_from_key_frames(key_frames)
        return pd.Series(feats)
    except Exception as exc:  # noqa: BLE001 - one bad video shouldn't kill the dataset build
        print(f"  [skip] {os.path.basename(video_path)}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
def load_dataset(data_dir=DATA_DIR, use_cache=True):
    """Walk ``data_dir/{pro,amateur}`` and build (X, y).

    Returns
    -------
    X : pandas.DataFrame   feature matrix (one row per successfully processed swing)
    y : numpy.ndarray      labels (1 = pro, 0 = amateur)
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found. Expected '{data_dir}/pro' and "
            f"'{data_dir}/amateur' sub-folders containing swing videos."
        )

    rows, labels = [], []
    for class_name, label in LABEL_MAP.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[warn] missing class folder: {class_dir} (skipping)")
            continue

        videos = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith(VIDEO_EXTS)
        )
        print(f"[{class_name}] found {len(videos)} videos")

        for fname in videos:
            video_path = os.path.join(class_dir, fname)
            print(f"  processing {fname} ...")
            feats = extract_features(video_path, use_cache=use_cache)
            if feats is not None:
                rows.append(feats)
                labels.append(label)

    if not rows:
        raise RuntimeError(
            "No swings were successfully processed. Check that videos exist and that "
            "MediaPipe pose estimation is working."
        )

    X = pd.DataFrame(rows).reset_index(drop=True)
    # Guard against any residual NaNs/infs before training.
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=int)

    print(f"\nBuilt feature matrix: {X.shape[0]} swings x {X.shape[1]} features")
    print(f"  pro={int((y == 1).sum())}  amateur={int((y == 0).sum())}")
    return X, y


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def _build_classifier():
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )


def evaluate_model(model, X_test, y_test):
    """Print accuracy, precision and recall on the held-out test set."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # zero_division=0 keeps metrics sane on tiny/degenerate test splits.
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print("\n=== Evaluation (test set) ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["amateur", "pro"], zero_division=0))

    return {"accuracy": accuracy, "precision": precision, "recall": recall}


def train_model(data_dir=DATA_DIR, model_path=MODEL_PATH, use_cache=True):
    """End-to-end: load dataset, train XGBoost with an 80/20 split, evaluate, and save."""
    X, y = load_dataset(data_dir, use_cache=use_cache)

    # 80/20 train/test split. Stratify only when both classes can be represented.
    stratify = y if (np.bincount(y).min() >= 2 and len(np.unique(y)) > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
    )

    model = _build_classifier()

    # Correct for class imbalance (e.g. far more pro than amateur swings) so the
    # model doesn't just learn to always predict the majority class.
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos > 0 and n_neg > 0:
        model.set_params(scale_pos_weight=n_neg / n_pos)
        print(f"\nClass balance (train): pro={n_pos} amateur={n_neg} "
              f"-> scale_pos_weight={n_neg / n_pos:.3f}")

    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save_model(model_path)
    print(f"\nSaved trained model to: {os.path.abspath(model_path)}")

    return model


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_video(video_path, model_path=MODEL_PATH, use_cache=True):
    """Extract features for one swing video and return the probability it is a PRO swing.

    Returns a float in [0, 1] (probability of class 1 = pro), or ``None`` if the swing
    could not be processed.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Train it first with: python train_xgboost.py"
        )

    feats = extract_features(video_path, use_cache=use_cache)
    if feats is None:
        print(f"Could not extract features from: {video_path}")
        return None

    model = _build_classifier()
    model.load_model(model_path)

    # Align feature order with what the model was trained on.
    booster_features = model.get_booster().feature_names
    row = feats.to_frame().T
    if booster_features is not None:
        row = row.reindex(columns=booster_features, fill_value=0.0)
    row = row.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    proba = float(model.predict_proba(row)[0, 1])
    print(f"\n{os.path.basename(video_path)}: probability of PRO swing = {proba:.4f}")
    return proba


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train / use an XGBoost golf-swing pro-vs-amateur classifier."
    )
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help=f"Dataset root with pro/ and amateur/ sub-folders (default: {DATA_DIR})")
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"Path to save/load the model (default: {MODEL_PATH})")
    parser.add_argument("--predict", metavar="VIDEO",
                        help="Predict pro-probability for a single video instead of training.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-run pose estimation even if a cached CSV exists.")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete the pose CSV cache before running.")
    return parser.parse_args()


def main():
    # Internal subprocess worker: extract pose for one video, then exit. This runs in
    # its own process (see _ensure_pose_csv) so resources are freed per video.
    if len(sys.argv) >= 2 and sys.argv[1] == "--extract-one":
        _extract_one(sys.argv[2], sys.argv[3], sys.argv[4])
        return

    args = _parse_args()

    if args.clear_cache and os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache: {CACHE_DIR}")

    use_cache = not args.no_cache

    if args.predict:
        predict_video(args.predict, model_path=args.model, use_cache=use_cache)
    else:
        train_model(data_dir=args.data_dir, model_path=args.model, use_cache=use_cache)


if __name__ == "__main__":
    main()
