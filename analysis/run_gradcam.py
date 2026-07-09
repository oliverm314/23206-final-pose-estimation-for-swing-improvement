"""Grad-CAM on the pretrained r3d_18 pipeline: WHERE (body/club) and WHEN
(downswing time) does it look? Heat on club/arms in transition = real signal;
heat smeared on the torso outline = posture/confound."""
import numpy as np, cv2, torch
import torch.nn.functional as Fn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

F = feats(X)
sc = StandardScaler().fit(F)
lr = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(sc.transform(F), y)
w_eff = torch.tensor((lr.coef_[0]/sc.scale_), dtype=torch.float32, device=DEVICE)
b_eff = float(lr.intercept_[0] - np.sum(lr.coef_[0]*sc.mean_/sc.scale_))

acts = {}
def _hook(m, i, o):
    o.retain_grad()
    acts['a'] = o
h = net.layer3.register_forward_hook(_hook)

def cam_of(i):
    x = torch.tensor(X[i:i+1], device=DEVICE, requires_grad=True)
    feat = net((x-MEAN)/STD)
    logit = (feat[0]*w_eff).sum() + b_eff
    net.zero_grad()
    logit.backward()
    A, G = acts['a'][0], acts['a'].grad[0]
    alpha = G.mean(dim=(1, 2, 3))
    cam = torch.relu((alpha[:, None, None, None]*A).sum(0))[None, None]
    cam = Fn.interpolate(cam, size=(T, H, W), mode="trilinear", align_corners=False)[0, 0]
    return (cam / (cam.max()+1e-8)).detach().cpu().numpy()

cams = np.stack([cam_of(i) for i in range(len(y))])
h.remove()

print("mean CAM energy per downswing frame (top->impact):")
print("  OTT    :", np.round(cams[y == 1].mean((0, 2, 3)), 2))
print("  non-OTT:", np.round(cams[y == 0].mean((0, 2, 3)), 2))

# filmstrip: 8 frames, ghost skeleton + averaged CAM overlay, per class
fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for r, (lab, name) in enumerate([(1, "OTT"), (0, "non-OTT")]):
    bg = X[y == lab, 0].mean(0)          # ghost body outline
    cl = X[y == lab, 1].mean(0)          # ghost club
    cm = cams[y == lab].mean(0)
    for c in range(8):
        f = c*(T//8)
        ax = axes[r, c]
        ax.imshow(np.clip(bg[f]*0.7 + cl[f]*1.0, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.imshow(cm[f], cmap="jet", alpha=0.55, vmin=0, vmax=cams.mean(0).max())
        ax.axis("off")
        if r == 0: ax.set_title(["top", "", "", "mid", "", "", "", "impact"][c], fontsize=9)
    axes[r, 0].set_ylabel(name, fontsize=12, rotation=90)
fig.suptitle("Grad-CAM on pretrained r3d_18 — where/when it looks "
             "(gray=skeleton+club, red=attention)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("cnn_pretrained_gradcam.png", dpi=150, bbox_inches="tight")
print("saved cnn_pretrained_gradcam.png")
