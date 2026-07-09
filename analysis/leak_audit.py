import os
import numpy as np
import train_xgboost as t
import cnn_pose as c

names, X, y = [], [], []
for path, label in t.find_videos(t.DATASET_DIR):
    seq, _ = t.load_or_extract_pose(path)
    ten = c.video_to_tensor(seq.landmarks)
    if ten is None:
        continue
    names.append(os.path.basename(path)); X.append(ten); y.append(label)
X = np.stack(X).reshape(len(names), -1); y = np.array(y)

# pairwise euclidean distance on the (centred/scaled/resampled) trajectories
d2 = np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0)
np.fill_diagonal(d2, np.inf)
dist = np.sqrt(d2)
nn = dist.argmin(1)
nnd = dist[np.arange(len(names)), nn]

# scale: typical distance, to judge what "close" means
med = np.median(dist[np.isfinite(dist)])
print(f"videos={len(names)}  median pairwise dist={med:.2f}")
print(f"nearest-neighbour dist: min={nnd.min():.2f}  p10={np.percentile(nnd,10):.2f}  "
      f"median={np.median(nnd):.2f}")

print("\n-- 25 closest cross-video pairs (candidate near-duplicates) --")
order = np.argsort(nnd)[:25]
seen = set()
for i in order:
    j = nn[i]
    key = tuple(sorted((i, j)))
    if key in seen:
        continue
    seen.add(key)
    same = "SAME" if y[i] == y[j] else "DIFF"
    print(f"  d={nnd[i]:6.2f} [{same} lbl {y[i]}/{y[j]}]  {names[i][:42]:42s} <-> {names[j][:42]}")

# how many videos have a 'twin' much closer than typical?
thr = 0.25 * med
ntwin = int((nnd < thr).sum())
print(f"\nvideos with a neighbour < 25%% of median dist ({thr:.2f}): {ntwin} / {len(names)}")
