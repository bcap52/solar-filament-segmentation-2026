"""Classical baseline: local-contrast dark-feature segmentation (no ML).

Pipeline: solar disk estimation -> local background (large Gaussian) ->
threshold dark residual -> morphological cleanup -> CC -> disjoint instances.
Local OOF PQ ~0.13; Kaggle LB 0.12 (2026-08-16) — validates the submission loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rle import mask_to_rle  # noqa: E402


def solar_disk_mask(img: np.ndarray) -> np.ndarray:
    thr = np.percentile(img, 55)
    _, bw = cv2.threshold(img, max(thr, 30), 255, cv2.THRESH_BINARY)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    if n <= 1:
        return np.ones_like(img, np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    disk = (lab == k).astype(np.uint8)
    disk = cv2.morphologyEx(disk, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    return (cv2.dilate(disk, np.ones((15, 15), np.uint8)) > 0).astype(np.uint8)


def segment_filaments(img: np.ndarray, thr: float = 12.0, min_area: int = 200,
                      bg_sigma: int = 75, open_iter: int = 1) -> list[np.ndarray]:
    img_f = img.astype(np.float32)
    disk = solar_disk_mask(img)
    masked = np.where(disk > 0, img_f, cv2.borderInterpolate(0, 0, cv2.BORDER_REPLICATE))
    k = 2 * bg_sigma + 1
    background = cv2.GaussianBlur(masked, (k, k), bg_sigma)
    residual = img_f - background
    cand = ((residual < -thr) & (disk > 0)).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, lab = cv2.connectedComponents(cand, connectivity=8)
    return [(lab == i).astype(np.uint8) for i in range(1, n) if (lab == i).sum() >= min_area]


def build_submission(test_dir: Path, out_csv: Path, **seg_kwargs) -> pd.DataFrame:
    rows = []
    for p in sorted(test_dir.glob("*.jpeg")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        for j, m in enumerate(segment_filaments(img, **seg_kwargs), 1):
            rows.append({"filament_id": f"{p.stem}_{j}", "segmentation_rle": mask_to_rle(m)})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(df)} instances")
    return df


if __name__ == "__main__":
    from src.data import TEST_IMG_DIR
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    min_area = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    build_submission(TEST_IMG_DIR, Path(sys.argv[3]) if len(sys.argv) > 3
                     else Path("subs/baseline_classical.csv"), thr=thr, min_area=min_area)
