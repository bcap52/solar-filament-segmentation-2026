"""RLE utilities matching the competition submission format.

Submission CSV: filament_id | segmentation_rle
segmentation_rle = compressed COCO RLE counts STRING (pycocotools),
size fixed 2048x2048, column-major (Fortran) order.
filament_id must be UNIQUE per row: "<image_stem>_<n>" (verified on the LB:
duplicate ids -> SubmissionStatus.ERROR).
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

H, W = 2048, 2048


def mask_to_rle(mask: np.ndarray) -> str:
    """Binary uint8 mask (H, W) -> compressed COCO RLE counts string."""
    assert mask.dtype == np.uint8 and mask.ndim == 2
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")


def rle_to_mask(counts: str, height: int = H, width: int = W) -> np.ndarray:
    """Compressed COCO RLE counts string -> binary uint8 mask (H, W)."""
    rle = {"size": [height, width], "counts": counts.encode("utf-8")}
    return mask_util.decode(rle)


def polygons_to_mask(polygons: list[list[float]], height: int = H, width: int = W) -> np.ndarray:
    """COCO polygon list -> binary uint8 mask (union of polygons)."""
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(np.uint8)
