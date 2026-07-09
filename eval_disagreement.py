"""
Confound-breaking evaluation: does the model track OVER-THE-TOP, or just skill/source?

In dataset/OTT vs dataset/NON_OTT, over-the-top is perfectly confounded with
"amateur Reddit clip vs pro/clean database clip". So a high CV score can't tell
the two apart. The only way to separate them is to test on cases where OTT and
skill/source DISAGREE:

    dataset/EVAL_OTT     <- GOOD / pro-looking swings that ARE over the top  (true label 1)
    dataset/EVAL_CLEAN   <- ROUGH / amateur swings that are NOT over the top (true label 0)

We train the normal XGBoost model on dataset/OTT + dataset/NON_OTT, then score
these held-out disagreement clips. Interpretation:

  * EVAL_OTT predicted OTT (p high)   -> model used real over-the-top signal.
    EVAL_OTT predicted clean (p low)  -> model was fooled by the "looks pro" cue.
  * EVAL_CLEAN predicted clean (p low)-> real signal.
    EVAL_CLEAN predicted OTT (p high) -> model just flags "looks amateur".

These clips are NOT in training (train only scans OTT/NON_OTT), so there's no leak.
Even ~10-15 per folder is enough to read the model's behaviour. Run:
    python eval_disagreement.py
"""
import os
import glob
import numpy as np
import pandas as pd

import train_xgboost as t

# Disagreement eval folders and their TRUE labels.
EVAL_DIRS = {"EVAL_OTT": 1, "EVAL_CLEAN": 0}


def _find(folder: str):
    """De-duplicated video paths in one folder (mirrors find_videos)."""
    paths, seen = [], set()
    for ext in t.VIDEO_EXTS:
        for p in glob.glob(os.path.join(folder, f"*{ext}")):
            key = os.path.normcase(os.path.abspath(p))
            if key not in seen:
                seen.add(key)
                paths.append(p)
    return paths


def _features_for(videos):
    """Run the exact pose+club feature pipeline on a list of (path, label)."""
    rows = []
    for path, label in videos:
        name = os.path.basename(path)
        seq, _ = t.load_or_extract_pose(path)
        if t.USE_CLUB and os.path.exists(t.CLUB_WEIGHTS):
            club, _ = t.load_or_extract_club(path)
            if (seq.landmarks.shape[0] and club.shape[0]
                    and club.shape[0] != seq.landmarks.shape[0]):
                L = min(club.shape[0], seq.landmarks.shape[0])
                seq.landmarks, club = seq.landmarks[:L], club[:L]
            seq.club = t._clean_club_track(club)
        rows.append({"video": name, "label": label, **t.compute_features(seq)})
    return pd.DataFrame(rows)


def main():
    feature_cols = list(t.FEATURE_REGISTRY.keys())

    # Gather the eval clips first, so we can bail early with instructions.
    eval_videos = []
    for d, lab in EVAL_DIRS.items():
        folder = os.path.join(t.DATASET_DIR, d)
        os.makedirs(folder, exist_ok=True)
        found = _find(folder)
        eval_videos += [(p, lab) for p in found]
        print(f"  {d:11s} (label {lab}): {len(found)} clip(s)")
    if not eval_videos:
        print("\n[!] No eval clips yet. Drop disagreement cases into:")
        print("      dataset/EVAL_OTT/    -> good/pro swings that ARE over the top")
        print("      dataset/EVAL_CLEAN/  -> rough/amateur swings that are NOT")
        print("    then re-run. (Keep them out of dataset/OTT and dataset/NON_OTT.)")
        return

    # Train the standard model on the full training set (native NaN kept).
    print("\nTraining model on dataset/OTT + dataset/NON_OTT ...")
    train_df = t.build_dataset(t.DATASET_DIR)
    model = t._make_model()
    model.fit(train_df[feature_cols].values, train_df["label"].values)

    # Score the disagreement clips.
    print("\nScoring disagreement clips ...")
    ev = _features_for(eval_videos)
    proba = model.predict_proba(ev[feature_cols].values)[:, 1]
    ev["p_OTT"] = proba
    ev["pred"] = (proba >= 0.5).astype(int)
    ev["correct"] = ev["pred"] == ev["label"]

    print("\n===================== Disagreement results =====================")
    print(f"{'folder':11s} {'video':40s} {'true':>4s} {'p(OTT)':>7s} {'ok':>3s}")
    for d, lab in EVAL_DIRS.items():
        for _, r in ev[ev["label"] == lab].sort_values("p_OTT").iterrows():
            print(f"{d:11s} {r['video'][:40]:40s} {int(r['label']):>4d} "
                  f"{r['p_OTT']:>7.3f} {'Y' if r['correct'] else '.':>3s}")

    print("\n-- read-out --")
    for d, lab in EVAL_DIRS.items():
        sub = ev[ev["label"] == lab]
        if len(sub):
            acc = sub["correct"].mean()
            print(f"  {d:11s}: {int(sub['correct'].sum())}/{len(sub)} correct "
                  f"(mean p(OTT)={sub['p_OTT'].mean():.3f})")
    both = ev["correct"].mean()
    print(f"\n  If the model tracks REAL over-the-top, both folders score high.")
    print(f"  If it just reads skill/source, EVAL_OTT scores LOW and EVAL_CLEAN")
    print(f"  scores 'OTT' -> the confound, exposed. Overall acc = {both:.2f}")
    print("================================================================")


if __name__ == "__main__":
    main()
