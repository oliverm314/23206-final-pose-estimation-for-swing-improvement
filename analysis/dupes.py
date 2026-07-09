import os, re
import train_xgboost as t

def stem(n):
    n = re.sub(r'\.(mp4|mov|avi|mkv)$', '', n, flags=re.I)
    n = re.sub(r'\s*\(\d+\)$|\s*-\s*Trim.*$|\s*-\s*T$|_segment\d+$', '', n)
    return n.strip()

groups = {}
for path, label in t.find_videos(t.DATASET_DIR):
    groups.setdefault(stem(os.path.basename(path)), []).append((path, label))

print("=== Duplicate / conflicting clips ===")
for s, items in sorted(groups.items()):
    if len(items) > 1:
        labels = {l for _, l in items}
        tag = "  <<< CONFLICTING LABELS" if len(labels) > 1 else ""
        print(f"\n[{len(items)} copies]{tag}")
        for path, label in items:
            print(f"   label={label}  {path}")
