import os
import numpy as np, pandas as pd
import train_xgboost as t
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

feat_cols = list(t.FEATURE_REGISTRY.keys())
rows = []
for path, label in t.find_videos(t.DATASET_DIR):
    seq, _ = t.load_or_extract_pose(path)
    if t.USE_CLUB and os.path.exists(t.CLUB_WEIGHTS):
        club, _ = t.load_or_extract_club(path)
        if (seq.landmarks.shape[0] and club.shape[0]
                and club.shape[0] != seq.landmarks.shape[0]):
            L = min(club.shape[0], seq.landmarks.shape[0])
            seq.landmarks, club = seq.landmarks[:L], club[:L]
        seq.club = t._clean_club_track(club)
    rows.append({"video": os.path.basename(path), "label": label,
                 **t.compute_features(seq)})

df = pd.DataFrame(rows).dropna(subset=feat_cols, how="all").reset_index(drop=True)
y = df["label"].values
raw = df[feat_cols].copy()

# A) current: median-impute
imp = raw.copy()
for c in feat_cols:
    imp[c] = imp[c].fillna(imp[c].median())

# B) native NaN (XGBoost learns the missing direction)
nat = raw.copy()

# C) native NaN + explicit club_detected indicator
ind = raw.copy()
ind["club_detected"] = (~raw["clubhead_above_plane"].isna()).astype(int)

def cv(X, n=5, reps=5):
    means = []
    for r in range(reps):
        skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=r)
        a = []
        for tr, te in skf.split(X.values, y):
            m = t._make_model()
            m.fit(X.values[tr], y[tr])
            a.append(roc_auc_score(y[te], m.predict_proba(X.values[te])[:, 1]))
        means.append(np.mean(a))
    return np.mean(means), np.std(means)

for name, X in [("A median-impute (current)", imp),
                ("B native NaN          ", nat),
                ("C native NaN + club_flag", ind)]:
    mu, sd = cv(X)
    print(f"{name}: 5-fold AUC (avg of 5 seeds) = {mu:.3f} +/- {sd:.3f}")
