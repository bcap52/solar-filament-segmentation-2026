"""Build a memory-mapped float16 NORMALIZED image stack (fast dataloading).

Same layout as images_u8.npy but percentile-normalized to [0,1] per image and
stored as f16 — training crops read a small memmap slice with zero CPU prep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import TRAIN_IMG_DIR, train_stems  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STACK = ROOT / "runs" / "images_f16.npy"


def main():
    STACK.parent.mkdir(parents=True, exist_ok=True)
    if STACK.exists():
        print("f16 stack already exists")
        return
    stems = train_stems()
    arr = np.lib.format.open_memmap(STACK, mode="w+", dtype=np.float16,
                                    shape=(len(stems), 2048, 2048))
    for k, stem in enumerate(stems):
        img = cv2.imread(str(TRAIN_IMG_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
        norm = np.clip((img.astype(np.float32) - p1) / max(p99 - p1, 1.0), 0, 1)
        arr[k] = norm.astype(np.float16)
        if (k + 1) % 100 == 0:
            print(f"{k+1}/{len(stems)}", flush=True)
    arr.flush()
    print(f"wrote {STACK} ({arr.nbytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
