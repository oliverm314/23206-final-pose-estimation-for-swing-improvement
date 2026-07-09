"""Test the user's idea: normalise the pose HARDER (remove position, size AND
in-plane rotation/lean -> canonical spine-vertical pose) and see if the static
source-leak dies. Ruler = the single-averaged-pose probe: if canon kills the
cheat, static AUC should fall toward 0.5."""
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import train_xgboost as t
import cnn_pose as c

J = c.JOINTS

def _interp(y):
    v = np.isfinite(y)
    return np.zeros_like(y) if v.sum() < 2 else np.interp(np.arange(len(y)), np.arange(len(y))[v], y[v])

def canon_tensor(landmarks):
    if landmarks.shape[0] == 0:
        return None
    xy = landmarks[:, J, :2].astype(float)
    for j in range(xy.shape[1]):
        for d in range(2):
            xy[:, j, d] = _interp(xy[:, j, d])
    midhip = (xy[:, 6] + xy[:, 7]) / 2.0
    midsh = (xy[:, 0] + xy[:, 1]) / 2.0
    out = np.zeros_like(xy)
    for f in range(len(xy)):
        p = xy[f] - midhip[f]                      # centre on mid-hip
        v = midsh[f] - midhip[f]                   # spine vector
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        # rotate so spine points up (target (0,-1) in image coords)
        ang = np.arctan2(v[0], -v[1])              # current angle from vertical
        ca, sa = np.cos(-ang), np.sin(-ang)
        R = np.array([[ca, -sa], [sa, ca]])
        out[f] = (p @ R.T) / n                     # scale by spine length
    flat = out.reshape(len(out), -1)
    return c._resample(flat).T.astype(np.float32)

X, y = [], []
for path, label in t.find_videos(t.DATASET_DIR):
    seq, _ = t.load_or_extract_pose(path)
    ten = canon_tensor(seq.landmarks)
    if ten is not None:
        X.append(ten); y.append(label)
X = np.stack(X); y = np.array(y)
print(f"canon tensors: {X.shape}  OTT={int(y.sum())} non={int((1-y).sum())}")

def static_mean(A):
    return np.repeat(A.mean(2, keepdims=True), A.shape[2], axis=2)

def cv(Xv, seeds=3):
    sm = []
    for seed in range(seeds):
        np.random.seed(seed); torch.manual_seed(seed)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed); a = []
        for tr, te in skf.split(Xv, y):
            itr, iva = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=seed)
            pw = (y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1)
            m = c.train_one(Xv[itr], y[itr], Xv[iva], y[iva], pw)
            with torch.no_grad():
                p = torch.sigmoid(m(torch.tensor(Xv[te], device=c.DEVICE))).cpu().numpy()
            a.append(roc_auc_score(y[te], p))
        sm.append(np.mean(a))
    return np.mean(sm), np.std(sm)

for name, Xv in [("canon real (motion)   ", X),
                 ("canon AVERAGED pose   ", static_mean(X))]:
    mu, sd = cv(Xv)
    print(f"{name}: CV-AUC = {mu:.3f} +/- {sd:.3f}")
