"""Force the pretrained model to use MOTION only: feed frame-to-frame DIFFERENCES
of the rendered skeleton instead of the frames. A static pose -> zero input, so
posture cannot be used. Does the Kinetics model still separate OTT from motion alone?"""
import numpy as np, cv2, torch
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load("golf_pose_bundle.npz", allow_pickle=True)
joints, club, cmask = d["joints"], d["club"], d["club_mask"]
y, T = d["y"].astype(int), int(d["T"])
H = W = 112
UPPER = [(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7)]

def render(J, C, M):
    img = np.zeros((3, H, W), np.float32); s = H*0.30
    pt = [(int(W/2+J[i,0]*s), int(H/2+J[i,1]*s)) for i in range(12)]
    body = np.zeros((H, W), np.uint8)
    for a,b in UPPER: cv2.line(body, pt[a], pt[b], 255, 2, cv2.LINE_AA)
    for i in range(8): cv2.circle(body, pt[i], 2, 255, -1)
    img[0] = body/255.0
    clu = np.zeros((H, W), np.uint8)
    cp = [(int(W/2+C[k,0]*s), int(H/2+C[k,1]*s)) for k in range(3)]
    if M[0] > .5 and M[1] > .5: cv2.line(clu, cp[1], cp[0], 255, 2, cv2.LINE_AA)
    if M[0] > .5: cv2.circle(clu, cp[0], 3, 255, -1)
    img[1] = clu/255.0
    return img

X = np.stack([np.stack([render(joints[i,f], club[i,f], cmask[i,f]) for f in range(T)], 1)
              for i in range(len(y))]).astype(np.float32)
# MOTION input: absolute frame-to-frame difference (static pose -> 0)
Xm = np.zeros_like(X)
Xm[:, :, 1:] = np.abs(X[:, :, 1:] - X[:, :, :-1])
print("motion volumes", Xm.shape, " mean nonzero frac:", float((Xm > 0).mean()))

from torchvision.models.video import r3d_18, R3D_18_Weights
net = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
net.fc = torch.nn.Identity(); net.eval().to(DEVICE)
MEAN = torch.tensor([0.43216,0.394666,0.37645], device=DEVICE).view(1,3,1,1,1)
STD = torch.tensor([0.22803,0.22145,0.216989], device=DEVICE).view(1,3,1,1,1)

@torch.no_grad()
def feats(v):
    o = []
    for i in range(0, len(v), 8):
        xb = torch.tensor(v[i:i+8], device=DEVICE)
        o.append(net((xb-MEAN)/STD).cpu().numpy())
    return np.concatenate(o)

F = feats(Xm); Fm = feats(Xm[:, :, :, :, ::-1].copy())

def cv():
    means = []
    for s in range(5):
        skf = StratifiedKFold(5, shuffle=True, random_state=s); a = []
        for tr, te in skf.split(F, y):
            sc = StandardScaler().fit(F[tr])
            Xtr = np.concatenate([sc.transform(F[tr]), sc.transform(Fm[tr])])
            clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(
                Xtr, np.concatenate([y[tr], y[tr]]))
            a.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(F[te]))[:,1]))
        means.append(np.mean(a))
    return float(np.mean(means)), float(np.std(means))

mu, sd = cv()
print(f"\nPretrained r3d_18 on MOTION (frame-diff): 5-fold CV-AUC = {mu:.3f} +/- {sd:.3f}")
print("compare -> pretrained on FRAMES 0.950 (static-probe 0.924) | direction 1D-CNN 0.906 | XGBoost 0.913")

# static probe: a frozen pose -> zero motion -> should collapse to chance
Fs = feats(np.zeros_like(Xm))
p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"),
                      StandardScaler().fit_transform(Fs + 1e-6*np.random.randn(*Fs.shape)), y,
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"STATIC (zero-motion) AUC = {roc_auc_score(y, p):.3f}  (~0.5 by construction = no posture leak)")
