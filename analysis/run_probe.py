"""Does the CNN actually need the swing MOTION, or just static posture/framing?
Compare CV-AUC on: real clips, time-shuffled (motion destroyed, poses kept),
and a single averaged pose (all static). If shuffled/static stay high, the model
is NOT using the over-the-top motion -> it's reading identity/source cues."""
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import train_xgboost as t
import cnn_pose as c

# build the base tensors once
X, y = [], []
for path, label in t.find_videos(t.DATASET_DIR):
    seq, _ = t.load_or_extract_pose(path)
    ten = c.video_to_tensor(seq.landmarks)
    if ten is not None:
        X.append(ten); y.append(label)
X = np.stack(X); y = np.array(y)

def time_shuffle(A):
    out = A.copy()
    for i in range(len(out)):
        out[i] = out[i][:, np.random.permutation(out.shape[2])]
    return out

def static_mean(A):
    m = A.mean(axis=2, keepdims=True)
    return np.repeat(m, A.shape[2], axis=2)

def cv(Xv, seeds=3):
    sm = []
    for seed in range(seeds):
        np.random.seed(seed); torch.manual_seed(seed)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        a = []
        for tr, te in skf.split(Xv, y):
            itr, iva = train_test_split(tr, test_size=0.15, stratify=y[tr],
                                        random_state=seed)
            pw = (y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1)
            model = c.train_one(Xv[itr], y[itr], Xv[iva], y[iva], pw)
            with torch.no_grad():
                p = torch.sigmoid(model(torch.tensor(Xv[te], device=c.DEVICE))).cpu().numpy()
            a.append(roc_auc_score(y[te], p))
        sm.append(np.mean(a))
    return np.mean(sm), np.std(sm)

for name, Xv in [("real clip (motion)        ", X),
                 ("time-SHUFFLED (no motion) ", time_shuffle(X)),
                 ("single AVERAGED pose      ", static_mean(X))]:
    mu, sd = cv(Xv)
    print(f"{name}: CV-AUC = {mu:.3f} +/- {sd:.3f}")
