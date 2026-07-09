"""Local version of the Colab notebook: Kinetics-pretrained r3d_18 as a frozen
feature extractor on the shortcut-reduced skeleton renders, 5-fold CV + static probe."""
import numpy as np, cv2, torch
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load("golf_pose_bundle.npz", allow_pickle=True)
joints, club, cmask = d["joints"], d["club"], d["club_mask"]
y, names, T = d["y"].astype(int), d["names"], int(d["T"])
print(f"N={len(y)} OTT={int(y.sum())} non={int((1-y).sum())} device={DEVICE}")

H = W = 112
DRAW_LEGS, DRAW_CLUB = False, True
UPPER = [(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7)]
LEGS = [(6,8),(8,10),(7,9),(9,11)]

def render(J, C, M):
    img = np.zeros((3, H, W), np.float32); s = H*0.30
    pt = [(int(W/2+J[i,0]*s), int(H/2+J[i,1]*s)) for i in range(12)]
    body = np.zeros((H, W), np.uint8)
    for a,b in UPPER + (LEGS if DRAW_LEGS else []):
        cv2.line(body, pt[a], pt[b], 255, 2, cv2.LINE_AA)
    for i in (range(12) if DRAW_LEGS else range(8)):
        cv2.circle(body, pt[i], 2, 255, -1)
    img[0] = body/255.0
    if DRAW_CLUB:
        clu = np.zeros((H, W), np.uint8)
        cp = [(int(W/2+C[k,0]*s), int(H/2+C[k,1]*s)) for k in range(3)]
        if M[0] > .5 and M[1] > .5: cv2.line(clu, cp[1], cp[0], 255, 2, cv2.LINE_AA)
        if M[0] > .5: cv2.circle(clu, cp[0], 3, 255, -1)
        img[1] = clu/255.0
    return img

X = np.stack([np.stack([render(joints[i,f], club[i,f], cmask[i,f]) for f in range(T)], 1)
              for i in range(len(y))]).astype(np.float32)
print("volumes", X.shape)

from torchvision.models.video import r3d_18
try:
    from torchvision.models.video import R3D_18_Weights
    net = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
except Exception:
    net = r3d_18(pretrained=True)
net.fc = torch.nn.Identity(); net.eval().to(DEVICE)
MEAN = torch.tensor([0.43216,0.394666,0.37645], device=DEVICE).view(1,3,1,1,1)
STD = torch.tensor([0.22803,0.22145,0.216989], device=DEVICE).view(1,3,1,1,1)

@torch.no_grad()
def feats(vols):
    out = []
    for i in range(0, len(vols), 8):
        xb = torch.tensor(vols[i:i+8], device=DEVICE)
        out.append(net((xb-MEAN)/STD).cpu().numpy())
    return np.concatenate(out)

F = feats(X); Fm = feats(X[:, :, :, :, ::-1].copy())
print("features", F.shape)

def cv(seeds=5):
    means = []
    for s in range(seeds):
        skf = StratifiedKFold(5, shuffle=True, random_state=s); a = []
        for tr, te in skf.split(F, y):
            sc = StandardScaler().fit(F[tr])
            Xtr = np.concatenate([sc.transform(F[tr]), sc.transform(Fm[tr])])
            ytr = np.concatenate([y[tr], y[tr]])
            clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(Xtr, ytr)
            a.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(F[te]))[:,1]))
        means.append(np.mean(a))
    return float(np.mean(means)), float(np.std(means))

mu, sd = cv()
print(f"\nPretrained r3d_18 + logistic head: 5-fold CV-AUC = {mu:.3f} +/- {sd:.3f}")
print("compare -> XGBoost 0.913 | direction 1D-CNN 0.906 | skeleton 3D-CNN scratch 0.859")

mid = T//2
Fs = feats(np.repeat(X[:, :, mid:mid+1], T, axis=2))
sc = StandardScaler().fit(Fs)
p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"),
                      sc.transform(Fs), y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                      method="predict_proba")[:, 1]
print(f"STATIC single-frame AUC = {roc_auc_score(y, p):.3f}  (want LOW vs the motion score)")
