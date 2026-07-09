"""Full RGB video (what r3d_18 actually expects), each clip cropped to a
consistent square box around the golfer (kills the zoom confound). We also run
the static-frame probe + a motion (frame-diff) variant to see if it's reading the
swing or just the appearance (background/clothes/physique remain in the pixels)."""
import os, numpy as np, cv2, torch
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import train_xgboost as t

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T, S = 16, 112

def rgb_volume(path):
    seq, _ = t.load_or_extract_pose(path)
    lm = seq.landmarks
    if lm.shape[0] == 0:
        return None
    ph = t.detect_swing_phases(seq)
    if ph.downswing_start is None:
        return None
    top = ph.downswing_start
    imp = ph.downswing_end if ph.downswing_end is not None else lm.shape[0] - 1
    if imp - top < 2:
        imp = min(lm.shape[0] - 1, top + 2)
    cap = cv2.VideoCapture(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W == 0 or H == 0:
        cap.release(); return None
    # consistent square crop box (pixels) from the body bbox over the downswing
    sub = lm[top:imp+1, :, :2]
    xs = (sub[..., 0] * W); ys = (sub[..., 1] * H)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 10:
        cap.release(); return None
    xs, ys = xs[m], ys[m]
    cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
    side = max(xs.max()-xs.min(), ys.max()-ys.min()) * 1.6
    x0, y0 = int(cx-side/2), int(cy-side/2); side = int(side)
    # read the downswing frames, crop (with black pad if box off-edge), resize
    cap.set(cv2.CAP_PROP_POS_FRAMES, top)
    frames = []
    for _ in range(imp - top + 1):
        ok, fr = cap.read()
        if not ok:
            break
        canvas = np.zeros((side, side, 3), np.uint8)
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0+side, W), min(y0+side, H)
        if sx1 > sx0 and sy1 > sy0:
            canvas[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = fr[sy0:sy1, sx0:sx1]
        rgb = cv2.cvtColor(cv2.resize(canvas, (S, S)), cv2.COLOR_BGR2RGB)
        frames.append(rgb.astype(np.float32)/255.0)
    cap.release()
    if len(frames) < 2:
        return None
    arr = np.stack(frames)                       # (f,S,S,3)
    idx = np.linspace(0, len(arr)-1, T).round().astype(int)
    return arr[idx].transpose(3, 0, 1, 2)         # (3,T,S,S)

X, y = [], []
for path, label in t.find_videos(t.DATASET_DIR):
    v = rgb_volume(path)
    if v is not None:
        X.append(v); y.append(label)
X = np.stack(X).astype(np.float32); y = np.array(y)
print(f"N={len(y)} OTT={int(y.sum())} non={int((1-y).sum())} vol={X.shape[1:]}")

from torchvision.models.video import r3d_18, R3D_18_Weights
net = r3d_18(weights=R3D_18_Weights.KINETICS400_V1); net.fc = torch.nn.Identity()
net.eval().to(DEVICE)
MEAN = torch.tensor([0.43216,0.394666,0.37645], device=DEVICE).view(1,3,1,1,1)
STD = torch.tensor([0.22803,0.22145,0.216989], device=DEVICE).view(1,3,1,1,1)

@torch.no_grad()
def feats(v):
    o = []
    for i in range(0, len(v), 8):
        xb = torch.tensor(v[i:i+8], device=DEVICE)
        o.append(net((xb-MEAN)/STD).cpu().numpy())
    return np.concatenate(o)

def cv(F, Fm):
    means = []
    for s in range(5):
        skf = StratifiedKFold(5, shuffle=True, random_state=s); a = []
        for tr, te in skf.split(F, y):
            sc = StandardScaler().fit(F[tr])
            clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(
                np.concatenate([sc.transform(F[tr]), sc.transform(Fm[tr])]),
                np.concatenate([y[tr], y[tr]]))
            a.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(F[te]))[:,1]))
        means.append(np.mean(a))
    return float(np.mean(means)), float(np.std(means))

F = feats(X); Fm = feats(X[:, :, :, :, ::-1].copy())
mu, sd = cv(F, Fm)
print(f"\nREAL VIDEO (cropped) r3d_18: 5-fold CV-AUC = {mu:.3f} +/- {sd:.3f}")

mid = T//2
Fs = feats(np.repeat(X[:, :, mid:mid+1], T, axis=2))
p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"),
                      StandardScaler().fit_transform(Fs), y,
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:,1]
print(f"STATIC single-frame AUC = {roc_auc_score(y, p):.3f}  (high => reading appearance, not swing)")

Xm = np.zeros_like(X); Xm[:, :, 1:] = np.abs(X[:, :, 1:] - X[:, :, :-1])
Fmo = feats(Xm); Fmo_m = feats(Xm[:, :, :, :, ::-1].copy())
mu2, sd2 = cv(Fmo, Fmo_m)
print(f"REAL VIDEO MOTION (frame-diff) r3d_18: 5-fold CV-AUC = {mu2:.3f} +/- {sd2:.3f}")
print("compare -> skeleton-frames 0.950/static0.924 | skeleton-motion 0.920/static0.473 | XGBoost 0.913")
