"""Build a memory-mapped uint8 image stack + per-image percentile stats.

Avoids the ~2.8 GB resident image cache that triggered Windows
\"The paging file is too small\" (os error 1455) during training startup.
Training reads crops through np.memmap — pages load on demand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import TRAIN_IMG_DIR, train_stems  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STACK = ROOT / "runs" / "images_u8.npy"
STATS = ROOT / "runs" / "image_stats.json"


def main():
    STACK.parent.mkdir(parents=True, exist_ok=True)
    if STACK.exists() and STATS.exists():
        print("mmap stack already exists")
        return
    stems = train_stems()
    arr = np.lib.format.open_memmap(STACK, mode="w+", dtype=np.uint8,
                                    shape=(len(stems), 2048, 2048))
    stats = {}
    for k, stem in enumerate(stems):
        img = cv2.imread(str(TRAIN_IMG_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
        arr[k] = img
        stats[stem] = [float(p1), float(max(p99, p1 + 1))]
        if (k + 1) % 100 == 0:
            print(f"{k+1}/{len(stems)}", flush=True)
    arr.flush()
    json.dump(stats, open(STATS, "w"))
    print(f"wrote {STACK} ({arr.nbytes/1e9:.2f} GB) + stats for {len(stems)} stems")


def load_stack():
    """Return (memmap array, {stem: (row, p1, p99)})."""
    stats = json.load(open(STATS))
    stems = train_stems()
    row_of = {s: i for i, s in enumerate(stems)}
    arr = np.load(STACK, mmap_mode="r")
    table = {s: (row_of[s], v[0], v[1]) for s, v in stats.items()}
    return arr, table


if __name__ == "__main__":
    main()
