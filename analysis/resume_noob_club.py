"""Resume the 640 amateur club rebuild: does NOT delete cache, so already-done
clips are reused (fast) and only the missing ones get extracted at 640."""
import os, re, time
import numpy as np
import train_xgboost as t

def is_pro(n): return re.match(r'^\d+\.(mp4|mov|avi|mkv|webm)$', n, re.I) is not None
def head_rate(c):
    return 0.0 if c.shape[0]==0 else float(np.mean(np.isfinite(c[:,t.HEAD,2]) & (c[:,t.HEAD,2]>0)))

noob = [(p, l) for p, l in t.find_videos(t.DATASET_DIR)
        if not is_pro(os.path.basename(p))]
todo = [(p, l) for p, l in noob if not os.path.exists(t._club_cache_path(p))]
print(f"imgsz={t.CLUB_IMGSZ}  noob total={len(noob)}  still missing={len(todo)}", flush=True)

t0, rates = time.time(), []
for i, (p, _) in enumerate(noob, 1):
    club, hit = t.load_or_extract_club(p)   # reuse if cached, else extract@640
    r = head_rate(club); rates.append(r)
    if not hit:
        print(f"[{i}/{len(noob)}] {r:.2f}  {os.path.basename(p)[:55]}", flush=True)

print(f"\ndone in {time.time()-t0:.0f}s  mean head-detect(all noob)={np.mean(rates):.2f} "
      f"(was ~0.26 at imgsz 320)", flush=True)
