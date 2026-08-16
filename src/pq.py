"""Official Panoptic Quality evaluation — exact port of the organizer's
Self_Evaluation_Notebook (azimahmadzadeh/self-evaluation-notebook, 2026-08-09).

Semantics (verified against the notebook source):
- Per annotator-image (GT filament_id "<annotator>-<image>"): IoU & Dice matrices
  between that annotator's GT instances and ALL predictions for the image.
- IoU/Dice = 0 when union == 0.
- hit := IoU > 0.5 (strictly greater).
- TP pairs = ALL (gt, pred) pairs with IoU > 0.5 (multi-match allowed).
- FP = pred with no hit; FN = GT with no hit; per annotator-image.
- n_gt == 0 row: all preds are FPs. n_pred == 0 row: all GTs are FNs.
- PQ = sum(TP IoU) / (|TP| + 0.5*FP + 0.5*FN); 0.0 if denominator 0.

Implementation: IoU matrix via pycocotools C implementation; Dice derived by the
exact identity dice = 2*iou/(1+iou) for iou>0 else 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_util

IOU_THRESHOLD = 0.5


def _rle(mask: np.ndarray):
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))


def overlap_matrices(
    gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """(iou_matrix, dice_matrix) of shapes (n_gt, n_pred); official semantics."""
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred)), np.zeros((n_gt, n_pred))
    gt_rles = [_rle(m) for m in gt_masks]
    pred_rles = [_rle(m) for m in pred_masks]
    iou = mask_util.iou(gt_rles, pred_rles, [0] * len(pred_rles))
    iou = np.asarray(iou, dtype=np.float64)
    dice = np.where(iou > 0, 2.0 * iou / (1.0 + iou), 0.0)
    return iou, dice


@dataclass
class PQResult:
    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    n_gt_images: int
    iou_scores: list = field(default_factory=list)
    pair_ious: list = field(default_factory=list)
    pair_dices: list = field(default_factory=list)
    per_image: list = field(default_factory=list)


@dataclass
class PQAccumulator:
    """Streaming accumulator — evaluate image-by-image without holding all masks."""
    tp_ious: list = field(default_factory=list)
    fp: int = 0
    fn: int = 0
    pair_ious: list = field(default_factory=list)
    pair_dices: list = field(default_factory=list)
    per_image: list = field(default_factory=list)

    def add(self, stem: str, gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]):
        n_gt, n_pred = len(gt_masks), len(pred_masks)
        iou, dice = overlap_matrices(gt_masks, pred_masks)
        if n_gt == 0:
            self.fp += n_pred
            self.per_image.append(dict(stem=stem, n_gt=0, n_pred=n_pred, tp=0, fp=n_pred, fn=0))
            return
        if n_pred == 0:
            self.fn += n_gt
            self.per_image.append(dict(stem=stem, n_gt=n_gt, n_pred=0, tp=0, fp=0, fn=n_gt))
            return
        hit = iou > IOU_THRESHOLD
        self.tp_ious.extend(iou[hit].tolist())
        col, row = hit.sum(axis=0), hit.sum(axis=1)
        img_fp = int((col == 0).sum())
        img_fn = int((row == 0).sum())
        self.fp += img_fp
        self.fn += img_fn
        self.per_image.append(dict(stem=stem, n_gt=n_gt, n_pred=n_pred,
                                   tp=int(hit.sum()), fp=img_fp, fn=img_fn))
        ov = iou > 0
        self.pair_ious.extend(iou[ov].tolist())
        self.pair_dices.extend(dice[ov].tolist())

    def result(self) -> PQResult:
        tp = len(self.tp_ious)
        denom = tp + 0.5 * self.fp + 0.5 * self.fn
        pq = float(np.sum(self.tp_ious) / denom) if denom > 0 else 0.0
        sq = float(np.mean(self.tp_ious)) if tp > 0 else 0.0
        rq = float(tp / denom) if denom > 0 else 0.0
        return PQResult(pq=pq, sq=sq, rq=rq, tp=tp, fp=self.fp, fn=self.fn,
                        n_gt_images=len(self.per_image), iou_scores=list(self.tp_ious),
                        pair_ious=list(self.pair_ious), pair_dices=list(self.pair_dices),
                        per_image=list(self.per_image))


def evaluate_pq(
    gt_rows: list[tuple[str, list[np.ndarray]]],
    pred_rows: dict[str, list[np.ndarray]],
) -> PQResult:
    """Evaluate PQ the official way (batch; use PQAccumulator for streaming)."""
    acc = PQAccumulator()
    for stem, gt_masks in gt_rows:
        acc.add(stem, gt_masks, pred_rows.get(stem, []))
    return acc.result()
