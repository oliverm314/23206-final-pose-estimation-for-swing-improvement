import os, re, datetime
import numpy as np, pandas as pd
import train_xgboost as t
from sklearn.model_selection import StratifiedKFold, cross_val_predict

feat_cols = list(t.FEATURE_REGISTRY.keys())
rows = []
for path, label in t.find_videos(t.DATASET_DIR):
    name = os.path.basename(path)
    seq, _ = t.load_or_extract_pose(path)
    if t.USE_CLUB and os.path.exists(t.CLUB_WEIGHTS):
        club, _ = t.load_or_extract_club(path)
        if (seq.landmarks.shape[0] and club.shape[0]
                and club.shape[0] != seq.landmarks.shape[0]):
            L = min(club.shape[0], seq.landmarks.shape[0])
            seq.landmarks, club = seq.landmarks[:L], club[:L]
        seq.club = t._clean_club_track(club)
    rows.append({"video": name, "label": label, "mtime": os.path.getmtime(path),
                 **t.compute_features(seq)})

df = pd.DataFrame(rows).dropna(subset=feat_cols, how="all").reset_index(drop=True)
nan_mask = df[feat_cols].isna()
X = df[feat_cols].copy()
for c in feat_cols:
    X[c] = X[c].fillna(X[c].median())
y = df["label"].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=t.RANDOM_STATE)
df["oof_p"] = cross_val_predict(t._make_model(), X.values, y, cv=skf,
                                method="predict_proba")[:, 1]
df["club_missing"] = nan_mask["clubhead_above_plane"].values

ott = df[df.label == 1]
non = df[df.label == 0]
print("=== Does missing club detection explain the misses? (OTT only) ===")
for grp, sub in [("club DETECTED", ott[~ott.club_missing]),
                 ("club MISSING ", ott[ott.club_missing])]:
    print(f"  {grp}: n={len(sub):2d}  mean oof_p={sub.oof_p.mean():.3f}  "
          f"recall@0.5={np.mean(sub.oof_p>=0.5):.2%}")

print("\n=== Is club detection worse on the NEWER OTT clips? ===")
ott_sorted = ott.sort_values("mtime")
half = len(ott_sorted) // 2
older, newer = ott_sorted.iloc[:half], ott_sorted.iloc[half:]
for nm, sub in [("older half", older), ("newer half", newer)]:
    span = (datetime.datetime.fromtimestamp(sub.mtime.min()).strftime('%m-%d') +
            ".." + datetime.datetime.fromtimestamp(sub.mtime.max()).strftime('%m-%d'))
    print(f"  {nm} ({span}): club_missing={sub.club_missing.mean():.2%}  "
          f"mean oof_p={sub.oof_p.mean():.3f}")

print("\n=== Class separability of clubhead_above_plane when present ===")
print(f"  OTT (detected) median = {ott[~ott.club_missing]['clubhead_above_plane'].median():.3f}")
print(f"  NON (detected) median = {non[~non.club_missing.values]['clubhead_above_plane'].median():.3f}"
      if False else "")
nond = non[~nan_mask['clubhead_above_plane'][df.label==0].values]
print(f"  NON (detected) median = {nond['clubhead_above_plane'].median():.3f}  n={len(nond)}")

print("\n=== Possible duplicate clips (same stem, ' - '/'(1)'/Trim copies) ===")
def stem(n):
    n = re.sub(r'\.(mp4|mov|avi|mkv)$', '', n, flags=re.I)
    n = re.sub(r'\s*\(1\)$|\s*-\s*Trim.*$|\s*-\s*T$', '', n)
    return n[:48]
df["stem"] = df.video.map(stem)
for s, g in df.groupby("stem"):
    if len(g) > 1:
        print(f"  {s!r}: " + ", ".join(f"{r.video[:40]}(lbl{r.label})" for _,r in g.iterrows()))
