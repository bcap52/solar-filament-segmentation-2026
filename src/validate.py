"""Validate a segmenter on train images against the multi-annotator GT (streaming PQ)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_annotations, group_by_stem, image_path, train_stems  # noqa: E402
from src.pq import PQAccumulator  # noqa: E402


def validate(segmenter, stems=None, verbose=True) -> dict:
    by_key = load_annotations()
    by_stem = group_by_stem(by_key)
    stems = stems or train_stems()
    acc = PQAccumulator()
    for k, stem in enumerate(stems):
        img = cv2.imread(str(image_path(stem)), cv2.IMREAD_GRAYSCALE)
        pred_masks = segmenter(img)
        for ann in by_stem[stem]:
            acc.add(stem, [i.mask() for i in ann.instances], pred_masks)
        if verbose and (k + 1) % 50 == 0:
            r = acc.result()
            print(f"  [{k+1}/{len(stems)}] PQ={r.pq:.4f} SQ={r.sq:.4f} RQ={r.rq:.4f}")
    res = acc.result()
    if verbose:
        print(f"FINAL: PQ={res.pq:.4f} SQ={res.sq:.4f} RQ={res.rq:.4f} "
              f"TP={res.tp} FP={res.fp} FN={res.fn}")
    return dict(pq=res.pq, sq=res.sq, rq=res.rq, tp=res.tp, fp=res.fp, fn=res.fn)


if __name__ == "__main__":
    from src.baseline_classical import segment_filaments
    stems = train_stems()[:100]
    for thr in (8.0, 12.0, 16.0):
        print(f"--- thr={thr}")
        validate(lambda im, t=thr: segment_filaments(im, thr=t), stems)
