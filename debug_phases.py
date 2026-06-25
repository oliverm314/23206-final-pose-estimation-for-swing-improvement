"""Quick visual check that swing-phase detection is sensible.

Plots the lead-wrist height (what the detector actually uses) vs frame for a
sample of videos, with vertical markers at the detected backswing start, top
(downswing start) and impact (downswing end). If detection is working, the red
'top' line sits on the peak and the purple 'impact' line on the following trough.

Run after the pose cache is populated:  python debug_phases.py
Output: phase_debug.png  +  a table of phase indices.
"""
import glob
import numpy as np
import matplotlib.pyplot as plt

import train_xgboost as T


def sample(n_each: int = 4):
    ott = sorted(glob.glob("dataset/OTT/*.mp4"))[:n_each]
    non = sorted(glob.glob("dataset/NON_OTT/*.mp4"))[:n_each]
    return [(v, 1) for v in ott] + [(v, 0) for v in non]


def main():
    vids = sample(4)
    cols = 4
    rows = (len(vids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    print(f"{'video':32s} {'n':>4} {'bs':>5} {'top':>5} {'imp':>5}")
    for ax, (v, label) in zip(axes, vids):
        seq, _ = T.load_or_extract_pose(v)
        lm = seq.landmarks
        name = v.replace("\\", "/").split("/")[-1]
        if lm.shape[0] == 0:
            ax.set_title(f"{name[:24]} (no pose)", fontsize=9)
            continue

        y = lm[:, seq.lead_wrist_idx, 1]
        yi = T._interp_nan(y)
        h = T._moving_average(-yi, 5) if yi is not None else -y
        ax.plot(h, color="tab:blue", lw=1.3, label="wrist height (up=+)")

        ph = T.detect_swing_phases(seq)
        for idx, c, lab in [(ph.backswing_start, "tab:green", "backswing"),
                            (ph.downswing_start, "tab:red", "top"),
                            (ph.downswing_end, "tab:purple", "impact")]:
            if idx is not None:
                ax.axvline(idx, color=c, ls="--", lw=1.4, label=lab)

        ax.set_title(f"[{'OTT' if label else 'NON'}] {name[:22]}", fontsize=9)
        ax.set_xlabel("frame")
        ax.legend(fontsize=7)
        print(f"{name[:32]:32s} {lm.shape[0]:>4} "
              f"{str(ph.backswing_start):>5} {str(ph.downswing_start):>5} "
              f"{str(ph.downswing_end):>5}")

    for ax in axes[len(vids):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("phase_debug.png", dpi=110)
    print("\nsaved phase_debug.png")


if __name__ == "__main__":
    main()
