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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
)
import xgboost as xgb
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATASET_DIR = "dataset"
CLASS_DIRS = {"OTT": 1, "NON_OTT": 0}
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV")

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
L_WRIST, R_WRIST = 15, 16


@dataclass
class SwingSequence:
    """Per-frame pose landmarks for one swing video.

    landmarks: array of shape (n_frames, 33, 4) -> (x, y, z, visibility),
    in normalised image coordinates (0..1). NaN rows mean no pose was
    detected in that frame.
    """
    video_name: str
    landmarks: np.ndarray

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
# Stage 2: Feature computation  <-- THIS IS WHERE YOU EXTEND
# --------------------------------------------------------------------------- #
# Each feature is a function: SwingSequence -> float.
# Register it in FEATURE_REGISTRY and it automatically flows through the rest
# of the pipeline (dataset building, training, importance plots).

FeatureFn = Callable[[SwingSequence], float]


def _xy(landmarks: np.ndarray, idx: int) -> np.ndarray:
    """Return (n_frames, 2) array of (x, y) for one landmark, NaNs preserved."""
    return landmarks[:, idx, :2]


def feat_max_lead_wrist_height(seq: SwingSequence) -> float:
    """
    Maximum lead-wrist height above shoulder line, normalised by shoulder width.

        shoulder_mid_y = (left_shoulder_y + right_shoulder_y) / 2
        shoulder_width = distance(left_shoulder, right_shoulder)
        per-frame value = |shoulder_mid_y - lead_wrist_y| / shoulder_width
        feature        = max over all frames

    Note: image y grows downward, so a wrist ABOVE the shoulders has a SMALLER
    y than shoulder_mid_y. The absolute value captures magnitude of separation,
    per your specification.
    """
    lm = seq.landmarks
    if lm.shape[0] == 0:
        return np.nan

    ls = _xy(lm, L_SHOULDER)            # (n, 2)
    rs = _xy(lm, R_SHOULDER)
    lw = _xy(lm, seq.lead_wrist_idx)

    shoulder_mid_y = (ls[:, 1] + rs[:, 1]) / 2.0
    shoulder_width = np.linalg.norm(ls - rs, axis=1)   # (n,)
    lead_wrist_y = lw[:, 1]

    # Avoid divide-by-zero / degenerate frames
    valid = shoulder_width > 1e-6
    vals = np.full(lm.shape[0], np.nan, dtype=np.float64)
    vals[valid] = (
        np.abs(shoulder_mid_y[valid] - lead_wrist_y[valid])
        / shoulder_width[valid]
    )

    if np.all(np.isnan(vals)):
        return np.nan
    return float(np.nanmax(vals))


# The registry. To add a feature: write a function above, add it here. Done.
FEATURE_REGISTRY: Dict[str, FeatureFn] = {
    "max_lead_wrist_height": feat_max_lead_wrist_height,
    # "transition_steepness": feat_transition_steepness,   # <- future
    # "hip_shoulder_separation": feat_hip_shoulder_sep,    # <- future
}


def compute_features(seq: SwingSequence) -> Dict[str, float]:
    """Run every registered feature function over one swing."""
    return {name: fn(seq) for name, fn in FEATURE_REGISTRY.items()}


# --------------------------------------------------------------------------- #
# Dataset building
# --------------------------------------------------------------------------- #
def find_videos(dataset_dir: str) -> List[tuple[str, int]]:
    """Return list of (video_path, label) across all class folders."""
    items: List[tuple[str, int]] = []
    for class_name, label in CLASS_DIRS.items():
        folder = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(folder):
            print(f"[warn] missing class folder: {folder}")
            continue
        for ext in VIDEO_EXTS:
            for path in glob.glob(os.path.join(folder, f"*{ext}")):
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
        seq = extract_pose_sequence(path)
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

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
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

    model, metrics, eval_bundle = train_and_evaluate(df)
    make_plots(df, model, eval_bundle)
    print_example_predictions(df, model, eval_bundle)


if __name__ == "__main__":
    main()