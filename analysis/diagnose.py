import os
import numpy as np, pandas as pd
import train_xgboost as t
from sklearn.model_selection import StratifiedKFold, cross_val_predict

feat_cols = list(t.FEATURE_REGISTRY.keys())
videos = t.find_videos(t.DATASET_DIR)

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
    feats = t.compute_features(seq)          # raw, may contain NaN
    rows.append({"video": name, "label": label, "mtime": os.path.getmtime(path),
                 **feats})

df = pd.DataFrame(rows)
df = df.dropna(subset=feat_cols, how="all").reset_index(drop=True)
nan_mask = df[feat_cols].isna()
df["n_nan"] = nan_mask.sum(axis=1)

# impute exactly like build_dataset
X = df[feat_cols].copy()
for c in feat_cols:
    if X[c].isna().any():
        X[c] = X[c].fillna(X[c].median())
y = df["label"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=t.RANDOM_STATE)
oof = cross_val_predict(t._make_model(), X.values, y, cv=skf,
                        method="predict_proba")[:, 1]
df["oof_p"] = oof

# rank OTT videos: which the model misses (low p), and are they newly added?
ott = df[df["label"] == 1].copy().sort_values("mtime")
cut = ott["mtime"].quantile(0.0)
newest = ott.nlargest(20, "mtime")["video"].tolist()
print(f"OTT videos: {len(ott)}   mean oof_p(all OTT)={ott['oof_p'].mean():.3f}")
print(f"  newest 20 OTT mean oof_p = {df[df['video'].isin(newest)]['oof_p'].mean():.3f}")
print(f"  older OTT   mean oof_p = {ott[~ott['video'].isin(newest)]['oof_p'].mean():.3f}")

print("\n-- OTT videos the model misses worst (low p(OTT)), newest first by mtime --")
worst = ott[ott["oof_p"] < 0.5].sort_values("mtime", ascending=False)
print(f"{'video':52s} {'oof_p':>6s} {'n_nan':>5s}  imputed_feats")
import datetime
for _, r in worst.iterrows():
    bad = [c.replace('_','',0) for c in feat_cols if nan_mask.loc[r.name, c]]
    dt = datetime.datetime.fromtimestamp(r['mtime']).strftime('%m-%d')
    print(f"{r['video'][:52]:52s} {r['oof_p']:>6.3f} {int(r['n_nan']):>5d}  {dt} {bad}")

print(f"\nOTT missed (p<0.5): {len(worst)} / {len(ott)}")
print("imputation rate by feature (OTT only):")
for c in feat_cols:
    print(f"  {c:30s} {nan_mask[df['label']==1][c].mean():.2%}")
