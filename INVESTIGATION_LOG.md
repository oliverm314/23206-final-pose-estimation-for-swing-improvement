# Over-the-Top (OTT) Swing Detection — Investigation Log

A record of the modelling + interpretability investigation. Over-the-top is a
downswing fault where the club is thrown *outside* the ideal plane. We classify
it from down-the-line swing videos using MediaPipe pose + a YOLO11 club detector.

---

## TL;DR — model scoreboard (5-fold CV ROC-AUC)

Ranked by **trustworthiness**, not raw AUC. The key tool is the **static-frame
probe**: freeze one frame so there is *no motion* — if the model still scores
high, it is reading body posture (the confound), not the swing.

| model | AUC | static probe | verdict |
|---|---|---|---|
| pretrained r3d_18 · **real video** (cropped) | 0.971 | **0.975** | ❌ pure appearance (worst confound) |
| pretrained r3d_18 · **skeleton frames** | 0.950 | **0.924** | ❌ reads posture |
| pretrained r3d_18 · **skeleton motion** (frame-diff) | **0.920** | **0.473** | ✅ posture eliminated — best honest model |
| XGBoost features | 0.913 | n/a | ✅ swing mechanics |
| direction 1D-CNN | 0.906 | ~chance (velocities) | ✅ motion-direction |
| skeleton 3D-CNN (from scratch) | 0.859 | — | weakest; no pretraining |

**The monotonic shortcut curve** (`analysis/run_realvideo.py`): the more *appearance*
the model can see, the higher the AUC and the *less* honest it is —
real-video 0.971 (static 0.975) → skeleton-frames 0.950 (0.924) → skeleton-motion
0.920 (0.473) → direction-CNN 0.906. AUC goes **up** as trustworthiness goes **down**.
Full RGB video (even cropped to a consistent box, which fixes zoom) reopens the
confound completely — one frozen frame beats every model because background /
clothing / lighting / resolution / physique are a near-perfect source fingerprint.

**Central thesis:** the *representation you allow* decides whether the model
finds the swing or the physique. Give a powerful model raw positions and it reads
the body; constrain it to motion and it reads the fault.

---

## 1. The core problem — a source/label confound

`dataset/OTT` and `dataset/NON_OTT` are **perfectly confounded with data source**:

| source | non-OTT | OTT |
|---|---|---|
| numbered files (`743.mp4`, a clean/pro DB) | 158 | 0 |
| `rapidsave.com_*` (amateur Reddit posts) | 0 | 72 |

So the task as-built is *"pro-database vs amateur-Reddit clip"*, not
*"over-the-top vs not"*.

**Proof (`analysis/source_probe.py`):** features with *zero* swing content —
`body_scale` (on-screen size = camera zoom), club-detection rate, framing,
clip length — get **CV-AUC 0.994**, *higher* than the swing-feature model (0.913).
Top leak: `body_scale` |corr| **0.82** (pro median 0.538 vs amateur 0.315,
10–90 percentile ranges don't overlap). Camera zoom is not a swing property.

**Consequence:** no metric on this dataset can separate "OTT detector" from
"pro-vs-amateur detector", because OTT and skill are the *same variable* here
(pros rarely come over the top; amateurs who post are usually faulty).

**The only fix is data:** disagreement cases used as a *test set* — a few
pro/good swings that ARE over the top + amateur swings that are NOT. Harness
ready: `eval_disagreement.py` scores clips dropped in `dataset/EVAL_OTT` and
`dataset/EVAL_CLEAN`. Coaching-demo clips are the best source of pro-looking OTT.

---

## 2. Data hygiene (improved the feature model honestly)

- **Native NaN instead of median-imputation** (`train_xgboost.build_dataset`):
  club features are NaN on ~57% of OTT clips; median-imputing faked a non-OTT
  value into them. Letting XGBoost handle NaN lifted CV **0.895 → 0.905**.
  (Caveat: some of that gain is club-missingness = a source cue.)
- **De-duplication:** removed same-clip `- Trim`/`(1)` copies + one
  conflicting-label clip → CV **0.905 → 0.912 ± 0.025** (std nearly halved; the
  dupes were adding instability + fold leakage).
- **Club detection `CLUB_IMGSZ` 320 → 640:** amateur clubhead detection
  **0.26 → 0.69** (the 320 downscale was dropping the small, blurred club on
  exactly the amateur clips). `analysis/club_detect.py`.

---

## 3. Coordinate 1D-CNN — catching and removing the shortcut

- **From scratch on raw pose coords: 0.960**, but the **static-frame probe = 0.96
  too** → it wasn't using the swing at all. Saliency showed it keyed on
  **knee height** (`L_kn_y` alone = 0.76 AUC, ~static) — a stance fingerprint.
  (`analysis/run_probe.py`, `cnn_saliency_heatmap.png`.)
- **`analysis/run_canon.py`:** normalising *harder* (centre+scale+rotation)
  didn't help — static pose still 0.97 → the leak is build/posture, not framing.
- **De-shortcut fixes** (`cnn_deshortcut.py`): (1) crop to downswing, (2)
  velocities not positions (a static knee → 0), (3) a `head_above_plane` channel.
  → CV **0.902**, and saliency moved onto **arm + club velocity in the downswing**
  (`cnn_deshortcut_saliency.png`).
- **Speed vs direction** (`analysis/speed_vs_dir.py`): direction-only **0.910** >
  speed-only **0.812** > full-velocity 0.890. Technique (which-way) beats
  athleticism (how-fast), and dropping speed *helps*. Made `DIRECTION_ONLY` the
  default → **0.906**; saliency then flips onto `head_above_plane` = the literal
  OTT geometry (`cnn_deshortcut_direction_saliency.png`).
- **Plane-channel ablation** (`analysis/noplane.py`): removing `head_above_plane`
  barely drops CV (0.906 → 0.897); signal is redundant, model falls back on
  **lead-wrist lateral direction early in the downswing** (= hands going *out* at
  the transition). The CNN independently rediscovered the two hand-features
  (`early_downswing_out_vs_down`, `clubhead_above_plane`).

---

## 4. Pretrained (transfer learning) — powerful ≠ honest

Kinetics-pretrained torchvision `r3d_18` as a **frozen feature extractor** (255
clips too few to fine-tune) + logistic head, on rendered skeleton videos.
Colab notebook `golf_pretrained_colab.ipynb` (+ `prep_colab_bundle.py`,
`build_notebook.py`); runs locally too (`analysis/run_pretrained.py`).

- **Fed frames (positions): 0.950** — highest of all — but **static probe 0.924**
  → ~97% of the signal is static posture. **Grad-CAM** (`analysis/run_gradcam.py`,
  `cnn_pretrained_gradcam.png`) confirms: attention is a diffuse blob on the
  torso, brightest at the top of the downswing, *not* tracking the clubhead.
- **Fed motion (frame-differences): 0.920** with **static probe 0.473 (chance)**
  → posture provably eliminated, and **0.920 honest beats XGBoost 0.913 and the
  direction CNN 0.906**. `analysis/run_pretrained_motion.py`. Best trustworthy
  model. (Caveat: frame-diff still contains *speed*, which carries ~0.81 of the
  pro-vs-amateur signal, so some of 0.920 may be athleticism, not OTT path.)

---

## Figures (in repo root)

| file | what it shows |
|---|---|
| `cnn_saliency_heatmap.png` | coord-CNN keys on knees/stance (the shortcut) |
| `cnn_deshortcut_saliency.png` | after downswing-crop + velocities → arm/club |
| `cnn_deshortcut_direction_saliency.png` | direction-only → clubhead-above-plane |
| `cnn_deshortcut_noplane_saliency.png` | plane removed → lead-wrist-out early |
| `cnn_pretrained_gradcam.png` | pretrained-on-frames looks at torso, not club |
| `skeleton_example.png` | example rendered skeleton downswing (16 frames) |
| `ott_confusion_matrix.png`, `ott_roc.png`, `ott_importance.png` | XGBoost eval |

## Key files

| file | role |
|---|---|
| `train_xgboost.py` | feature pipeline + XGBoost (native-NaN, CLUB_IMGSZ=640) |
| `cnn_pose.py` | coordinate 1D-CNN (pose + club channels) |
| `cnn_deshortcut.py` | de-shortcut CNN (downswing crop, velocities, DIRECTION_ONLY) |
| `skeleton_cnn.py` | from-scratch skeleton 3D-CNN |
| `eval_disagreement.py` | **the validator** — scores EVAL_OTT / EVAL_CLEAN clips |
| `prep_colab_bundle.py` → `golf_pose_bundle.npz` | compact data for pretrained |
| `build_notebook.py` → `golf_pretrained_colab.ipynb` | Colab pretrained notebook |
| `analysis/` | all diagnostic/probe scripts (confound proof, ablations, etc.) |

---

## What's left (highest value first)

1. **Disagreement clips** (`EVAL_OTT` / `EVAL_CLEAN`, ~10–15 each) — the only
   thing that converts any AUC above into *verified* over-the-top detection.
2. Optional: Grad-CAM on the motion pretrained model (expect attention on the
   moving club, completing the frames→posture / motion→club contrast).
3. Optional: direction-only version of the pretrained motion input (removes the
   residual speed/athleticism component).
