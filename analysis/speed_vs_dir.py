"""Is the de-shortcut CNN using SPEED (how fast = skill/pro-vs-amateur) or
DIRECTION (which way = technique = the actual OTT fault)? Split each velocity
vector into the two and train on each alone."""
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import cnn_pose as c, cnn_deshortcut as d

X, y, _ = d.build_dataset()                      # (N,34,L)
T = X.shape[2]
vel = X[:, :30, :].reshape(len(y), 15, 2, T)     # 15 points x (vx,vy)
mag = np.linalg.norm(vel, axis=2, keepdims=True) # (N,15,1,T)

full = vel.reshape(len(y), 30, T).astype(np.float32)
speed = mag[:, :, 0, :].astype(np.float32)                       # (N,15,T)
direction = (vel / (mag + 1e-6)).reshape(len(y), 30, T).astype(np.float32)

def train_min(Xtr, ytr, Xva, yva, pw, epochs=60, patience=12):
    m = c.PoseCNN(Xtr.shape[1]).to(c.DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=c.DEVICE))
    Xva_t = torch.tensor(Xva, device=c.DEVICE)
    best, state, bad = -1, None, 0
    n = len(Xtr)
    for _ in range(epochs):
        m.train(); perm = np.random.permutation(n)
        for i in range(0, n, 16):
            idx = perm[i:i+16]
            xb = Xtr[idx] + np.random.randn(*Xtr[idx].shape).astype(np.float32)*0.02
            yb = torch.tensor(ytr[idx], dtype=torch.float32, device=c.DEVICE)
            opt.zero_grad(); lossf(m(torch.tensor(xb, device=c.DEVICE)), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = torch.sigmoid(m(Xva_t)).cpu().numpy()
        auc = roc_auc_score(yva, p) if len(np.unique(yva))>1 else .5
        if auc > best: best, bad, state = auc, 0, {k:v.cpu().clone() for k,v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); return m

def cv(Xv):
    means=[]
    for seed in range(3):
        np.random.seed(seed); torch.manual_seed(seed)
        skf=StratifiedKFold(5,shuffle=True,random_state=seed); a=[]
        for tr,te in skf.split(Xv,y):
            itr,iva=train_test_split(tr,test_size=0.15,stratify=y[tr],random_state=seed)
            pw=(y[itr]==0).sum()/max((y[itr]==1).sum(),1)
            m=train_min(Xv[itr],y[itr],Xv[iva],y[iva],pw)
            with torch.no_grad():
                p=torch.sigmoid(m(torch.tensor(Xv[te],device=c.DEVICE))).cpu().numpy()
            a.append(roc_auc_score(y[te],p))
        means.append(np.mean(a))
    return np.mean(means), np.std(means)

for name, Xv in [("FULL velocity (speed+dir)", full),
                 ("DIRECTION only (technique)", direction),
                 ("SPEED only (athleticism) ", speed)]:
    mu, sd = cv(Xv)
    print(f"{name}: CV-AUC = {mu:.3f} +/- {sd:.3f}")
