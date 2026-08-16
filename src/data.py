"""Data loading for the MAGFiLO Kaggle-2026 train annotations.

Key structure of the COCO-ish JSON:
- images: 1154 entries, id = "<annotator_set_id>-<jpeg_stem>" -> 707 physical
  JPEGs; 296 JPEGs have MULTIPLE annotator sets.
- annotations: 8199, each with polygon `segmentation`, `spine`, `area`, `bbox`,
  `image_id` (annotator-image key), `category_id` (1..4 chirality).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .rle import polygons_to_mask

DATA_ROOT = Path(__file__).resolve().parents[1] / "data_extract" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_JSON = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMG_DIR = DATA_ROOT / "train" / "train_images"
TEST_IMG_DIR = DATA_ROOT / "test" / "test_images"


@dataclass
class Instance:
    ann_id: str
    image_key: str
    stem: str
    category_id: int
    area: float
    bbox: tuple[float, float, float, float]
    polygons: list[list[float]]
    spine: list[float]
    _mask: np.ndarray | None = field(default=None, repr=False)

    def mask(self, cache: bool = False) -> np.ndarray:
        if self._mask is None:
            m = polygons_to_mask(self.polygons)
            if cache:
                self._mask = m
            return m
        return self._mask


@dataclass
class AnnImage:
    """One annotator's annotation set for one physical image."""
    image_key: str
    stem: str
    instances: list[Instance] = field(default_factory=list)


def load_annotations(json_path: Path = TRAIN_JSON) -> dict[str, AnnImage]:
    """Return {image_key: AnnImage} for all 1154 annotator-image sets."""
    with open(json_path) as f:
        d = json.load(f)
    by_key: dict[str, AnnImage] = {}
    stem_of = {}
    for im in d["images"]:
        key = im["id"]
        stem = key.split("-", 1)[1]
        stem_of[key] = stem
        by_key[key] = AnnImage(image_key=key, stem=stem)
    for a in d["annotations"]:
        seg = a["segmentation"]
        if isinstance(seg, str):
            seg = json.loads(seg)
        bbox = json.loads(a["bbox"]) if isinstance(a["bbox"], str) else a["bbox"]
        area = float(json.loads(a["area"]) if isinstance(a["area"], str) else a["area"])
        spine = json.loads(a["spine"]) if isinstance(a["spine"], str) else a["spine"]
        inst = Instance(
            ann_id=a["id"], image_key=a["image_id"], stem=stem_of[a["image_id"]],
            category_id=int(a["category_id"]), area=area, bbox=tuple(bbox),
            polygons=seg, spine=spine,
        )
        by_key[a["image_id"]].instances.append(inst)
    return by_key


def group_by_stem(by_key: dict[str, AnnImage]) -> dict[str, list[AnnImage]]:
    """{stem: [AnnImage, ...]} — annotator sets per physical JPEG."""
    g: dict[str, list[AnnImage]] = defaultdict(list)
    for ai in by_key.values():
        g[ai.stem].append(ai)
    return dict(g)


def train_stems() -> list[str]:
    return sorted(p.stem for p in TRAIN_IMG_DIR.glob("*.jpeg"))


def test_stems() -> list[str]:
    return sorted(p.stem for p in TEST_IMG_DIR.glob("*.jpeg"))


def image_path(stem: str, train: bool = True) -> Path:
    d = TRAIN_IMG_DIR if train else TEST_IMG_DIR
    return d / f"{stem}.jpeg"
