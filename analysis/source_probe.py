"""How separable are the classes using ONLY source/quality junk that has nothing
to do with the swing? (video length, detection rates, framing, body scale,
landmark confidence). If these alone hit ~0.9, XGBoost's 0.913 is partly riding
the source confound; if they sit low, its swing features carry real signal."""
import os, numpy as np, pandas as pd
import train_xgboost as t
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

rows = []
for path, label in t.find_videos(t.DATASET_DIR):
    seq, _ = t.load_or_extract_pose(path)
    lm = seq.landmarks
    if lm.shape[0] == 0:
        continue
    club, _ = t.load_or_extract_club(path) if (t.USE_CLUB and os.path.exists(t.CLUB_WEIGHTS)) else (np.zeros((0,3,3)), False)
    valid_pose = np.isfinite(lm[:, 0, 0])
    midhip = (t._xy(lm, t.L_HIP) + t._xy(lm, t.R_HIP)) / 2.0
    def drate(ci):
        if club.shape[0] == 0: return 0.0
        cc = club[:, ci, 2]
        return float(np.mean(np.isfinite(cc) & (cc > 0)))
    rows.append({
        "n_frames":        lm.shape[0],
        "pose_detect_rate": float(np.mean(valid_pose)),
        "head_detect_rate": drate(t.HEAD),
        "ball_detect_rate": drate(t.BALL),
        "handle_detect_rate": drate(t.HANDLE),
        "body_scale":      t._body_scale(lm),
        "framing_x":       float(np.nanmean(midhip[:, 0])),
        "framing_y":       float(np.nanmean(midhip[:, 1])),
        "mean_visibility": float(np.nanmean(lm[:, :, 3])),
        "label": label,
    })

df = pd.DataFrame(rows)
cols = [c for c in df.columns if c != "label"]
y = df["label"].values
print(f"videos={len(df)}  source-only features={cols}\n")

skf_means = []
for seed in range(10):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed); a = []
    for tr, te in skf.split(df[cols].values, y):
        m = t._make_model(); m.fit(df[cols].values[tr], y[tr])
        a.append(roc_auc_score(y[te], m.predict_proba(df[cols].values[te])[:, 1]))
    skf_means.append(np.mean(a))
print(f"SOURCE-ONLY junk features: 5-fold CV-AUC = {np.mean(skf_means):.3f} +/- {np.std(skf_means):.3f}")
print("(XGBoost swing features baseline = 0.913)")

print("\n|corr with label| for each junk feature:")
for c, v in df[cols].corrwith(df["label"]).abs().sort_values(ascending=False).items():
    print(f"  {c:20s} {v:.3f}")
