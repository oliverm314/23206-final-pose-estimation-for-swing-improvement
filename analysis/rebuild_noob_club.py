"""Rebuild the club cache for the AMATEUR (non-numbered) clips at CLUB_IMGSZ=640.
Non-numbered = rapidsave/tiktok/other. Pros (numbered) keep their cache. Deletes
each noob club-cache entry then re-extracts at 640 (writes the new cache)."""
import os, re, time
import numpy as np
import train_xgboost as t

def is_pro(n): return re.match(r'^\d+\.(mp4|mov|avi|mkv|webm)$', n, re.I) is not None
def head_rate(c):
    return 0.0 if c.shape[0]==0 else float(np.mean(np.isfinite(c[:,t.HEAD,2]) & (c[:,t.HEAD,2]>0)))

noob = [(p, l) for p, l in t.find_videos(t.DATASET_DIR)
        if not is_pro(os.path.basename(p))]
print(f"imgsz={t.CLUB_IMGSZ}  rebuilding {len(noob)} amateur clips", flush=True)

for p, _ in noob:                         # force re-extraction
    cp = t._club_cache_path(p)
    if os.path.exists(cp):
        os.remove(cp)

t0 = time.time()
rates = []
for i, (p, _) in enumerate(noob, 1):
    club, _ = t.load_or_extract_club(p)   # cache miss -> extract@640 -> cache
    r = head_rate(club)
    rates.append(r)
    print(f"[{i}/{len(noob)}] {r:.2f}  {os.path.basename(p)[:55]}", flush=True)

print(f"\ndone in {time.time()-t0:.0f}s  mean head-detect={np.mean(rates):.2f} "
      f"(was ~0.26 at imgsz 320)", flush=True)
