import os, re, numpy as np
import train_xgboost as t

def is_pro(n): return re.match(r'^\d+\.(mp4|mov|avi|mkv|webm)$', n, re.I) is not None

def head_rate(club):
    if club.shape[0] == 0: return 0.0
    cc = club[:, t.HEAD, 2]
    return float(np.mean(np.isfinite(cc) & (cc > 0)))

rows = []  # (name, path, label, grp, rate_or_None, nframes)
for path, label in t.find_videos(t.DATASET_DIR):
    n = os.path.basename(path)
    cpath = t._club_cache_path(path)
    if os.path.exists(cpath):
        club = np.load(cpath)
        rows.append((n, path, label, "pro" if is_pro(n) else "noob",
                     head_rate(club), club.shape[0]))
    else:
        rows.append((n, path, label, "pro" if is_pro(n) else "noob", None, 0))

cached = [r for r in rows if r[4] is not None]
uncached = [r for r in rows if r[4] is None]
print(f"cached={len(cached)}  uncached(new, not yet processed)={len(uncached)}")

cached.sort(key=lambda r: r[4])
print("\n== 15 LOWEST clubhead-detection clips (cached) ==")
for n, _, lab, grp, rate, nf in cached[:15]:
    print(f"  {rate:.2f}  [{grp} lbl{lab} {nf:4d}f]  {n[:52]}")

# Re-detect the worst NOOB clips at imgsz=640 (non-destructive: no cache write)
worst_noob = [r for r in cached if r[3] == "noob"][:8]
print("\n== imgsz 320 -> 640 on the 8 worst amateur clips ==")
t.CLUB_IMGSZ = 640
for n, path, lab, grp, old, nf in worst_noob:
    track = t.extract_club_track(path)     # runs YOLO at 640, does not cache
    new = head_rate(track)
    print(f"  {old:.2f} -> {new:.2f}  ({'+' if new>=old else ''}{new-old:+.2f})  {n[:50]}")
