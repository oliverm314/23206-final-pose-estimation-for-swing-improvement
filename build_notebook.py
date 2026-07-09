"""Generates golf_pretrained_colab.ipynb (run locally: python build_notebook.py)."""
import json


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s.splitlines(keepends=True)}


cells = [
md("""# Over-the-top swing — pretrained (Kinetics) video CNN

Transfer learning: a **`r3d_18`** 3D-CNN **pretrained on Kinetics-400** (~650k
human-action clips) is used as a frozen feature extractor; we train only a small
head on the golf data (~255 swings — too few to fine-tune a big net without
overfitting).

**Shortcuts removed up front** (everything we found leaking pro-vs-amateur):
1. **Rendered skeleton on black** → no background / camera / resolution cues.
2. **Per-frame mid-hip centring + body-scale normalisation** → no camera-zoom
   (`body_scale`) or absolute-position cue. *(done in the bundle)*
3. **Downswing crop** → no static setup / follow-through. *(done in the bundle)*
4. **Legs/knees dropped** (`DRAW_LEGS=False`) → the knees were the single biggest
   *static stance* shortcut; over-the-top is an arms/club fault, so we omit them.
5. **Club drawn** (head + shaft) → hands the model the real over-the-top signal.
6. Mirror augmentation + frozen backbone + regularised head.

**Still not fixed by any of this:** the *build/stance* confound can leak through
posture, so a high score is **not** proof of OTT detection — the disagreement
eval (pro-OTT + amateur-clean clips) remains the only validator. There's a
static-frame probe at the end to sanity-check.

**Setup:** Runtime → Change runtime type → **GPU**. Then run each cell. You'll be
asked to upload **`golf_pose_bundle.npz`** (made locally by `prep_colab_bundle.py`)."""),

code("""import torch, torchvision, numpy as np
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NONE  ->  Runtime > Change runtime type > GPU")"""),

md("""### 1. Upload the bundle
Run the cell, then pick `golf_pose_bundle.npz`."""),

code("""from google.colab import files
files.upload()
d = np.load("golf_pose_bundle.npz", allow_pickle=True)
joints, club, cmask = d["joints"], d["club"], d["club_mask"]
y, names, T = d["y"].astype(int), d["names"], int(d["T"])
print("joints", joints.shape, "| OTT", int(y.sum()), "| non", int((1 - y).sum()))"""),

md("""### 2. Render the skeletons (shortcuts removed here)
Black background, legs dropped, club drawn. Channel 0 = body, channel 1 = club."""),

code("""import cv2
H = W = 112                 # r3d_18 input size
DRAW_LEGS = False           # knees were the big STATIC stance shortcut -> drop
DRAW_CLUB = True            # clubhead + shaft = the real over-the-top signal

UPPER = [(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7)]
LEGS  = [(6,8),(8,10),(7,9),(9,11)]

def render(J, C, M):
    img = np.zeros((3, H, W), np.float32)
    s = H * 0.30
    pt = [(int(W/2 + J[i,0]*s), int(H/2 + J[i,1]*s)) for i in range(12)]
    body = np.zeros((H, W), np.uint8)
    for a,b in UPPER + (LEGS if DRAW_LEGS else []):
        cv2.line(body, pt[a], pt[b], 255, 2, cv2.LINE_AA)
    for i in (range(12) if DRAW_LEGS else range(8)):
        cv2.circle(body, pt[i], 2, 255, -1)
    img[0] = body / 255.0
    if DRAW_CLUB:
        clu = np.zeros((H, W), np.uint8)
        cp = [(int(W/2 + C[k,0]*s), int(H/2 + C[k,1]*s)) for k in range(3)]
        if M[0] > .5 and M[1] > .5:                 # shaft: handle -> head
            cv2.line(clu, cp[1], cp[0], 255, 2, cv2.LINE_AA)
        if M[0] > .5:                               # clubhead
            cv2.circle(clu, cp[0], 3, 255, -1)
        img[1] = clu / 255.0
    return img

def volume(i):
    return np.stack([render(joints[i,f], club[i,f], cmask[i,f]) for f in range(T)], 1)

X = np.stack([volume(i) for i in range(len(y))]).astype(np.float32)  # (N,3,T,H,W)
print("volumes", X.shape)"""),

code("""import matplotlib.pyplot as plt
i = int(np.where(y == 1)[0][0])
fig, ax = plt.subplots(1, 8, figsize=(16, 2.4))
for j in range(8):
    f = j * (T // 8)
    ax[j].imshow(X[i, :, f].transpose(1, 2, 0)); ax[j].axis("off"); ax[j].set_title(f"f{f}")
plt.suptitle(f"rendered downswing  (red=body, green=club)  —  {names[i]}"); plt.show()"""),

md("""### 3. Pretrained r3d_18 as a frozen feature extractor
Strip its classifier, forward each volume once → a 512-d feature vector. Frozen
because 255 clips is far too few to fine-tune 33M params without overfitting."""),

code("""from torchvision.models.video import r3d_18
try:
    from torchvision.models.video import R3D_18_Weights
    net = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
except Exception:
    net = r3d_18(pretrained=True)
net.fc = torch.nn.Identity()
net.eval().to(DEVICE)

MEAN = torch.tensor([0.43216,0.394666,0.37645], device=DEVICE).view(1,3,1,1,1)
STD  = torch.tensor([0.22803,0.22145,0.216989], device=DEVICE).view(1,3,1,1,1)

@torch.no_grad()
def feats(vols):
    out = []
    for i in range(0, len(vols), 8):
        xb = torch.tensor(vols[i:i+8], device=DEVICE)
        out.append(net((xb - MEAN) / STD).cpu().numpy())
    return np.concatenate(out)

F  = feats(X)                              # (N,512)
Fm = feats(X[:, :, :, :, ::-1].copy())     # horizontal mirror (augmentation)
print("features", F.shape)"""),

md("""### 4. Train a regularised head, 5-fold CV
Logistic head on the frozen features (mirror-augmented). This is the honest
transfer-learning number for ~255 clips."""),

code("""from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def cv(seeds=5):
    means = []
    for s in range(seeds):
        skf = StratifiedKFold(5, shuffle=True, random_state=s); a = []
        for tr, te in skf.split(F, y):
            sc = StandardScaler().fit(F[tr])
            Xtr = np.concatenate([sc.transform(F[tr]), sc.transform(Fm[tr])])
            ytr = np.concatenate([y[tr], y[tr]])
            clf = LogisticRegression(max_iter=2000, C=0.1,
                                     class_weight="balanced").fit(Xtr, ytr)
            a.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(F[te]))[:,1]))
        means.append(np.mean(a))
    return float(np.mean(means)), float(np.std(means))

mu, sd = cv()
print(f"Pretrained r3d_18 + logistic head: 5-fold CV-AUC = {mu:.3f} +/- {sd:.3f}")
print("compare -> XGBoost 0.913 | direction 1D-CNN 0.906 | skeleton 3D-CNN (scratch) 0.859")"""),

md("""### 5. Shortcut probe — is it using motion or static pose?
Freeze one mid-downswing frame (repeat it for all T) so there is **no motion**.
If that still scores high, the model is reading static posture (stance/build),
not the swing. Want this **low** relative to the real score above."""),

code("""from sklearn.model_selection import cross_val_predict
mid = T // 2
Fs = feats(np.repeat(X[:, :, mid:mid+1], T, axis=2))
sc = StandardScaler().fit(Fs)
p = cross_val_predict(
        LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"),
        sc.transform(Fs), y,
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
        method="predict_proba")[:, 1]
print(f"STATIC single-frame AUC = {roc_auc_score(y, p):.3f}   (want LOW vs the motion score)")"""),

md("""### Reading the results
- **Beats ~0.91 by a clear margin** → the borrowed Kinetics motion features are
  adding something over from-scratch. Nice.
- **Around 0.85–0.91** → it's competitive but not obviously better; the from-
  scratch coordinate model is simpler and just as good.
- **Static-frame AUC ≈ motion AUC** → even here it's leaning on posture, so the
  build/stance confound is still in play (expected — pretraining doesn't fix it).

**Whatever the number, it does not prove OTT detection** while the classes stay
pro-vs-amateur. The only thing that settles it is the disagreement eval
(`EVAL_OTT` / `EVAL_CLEAN`) back in the main repo.

**To fine-tune instead of freeze** (only if you gather a lot more data): unfreeze
`net.layer4`, train with a tiny LR (~1e-4), heavy augmentation."""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

with open("golf_pretrained_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote golf_pretrained_colab.ipynb")
