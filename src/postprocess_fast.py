"""Fast post-processing experiments with pre-cached GT masks + prob maps.

GT masks are rasterized once per (stem, annotator-set) into gt_cache.npz (RLE
strings); probability maps are cached as float16 .npy. Variant evaluation then
only does pycocotools IoU — seconds per variant instead of minutes.
Resume-safe: variants already present in postprocess_results.txt are skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import group_by_stem, load_annotations  # noqa: E402
from src.pq import PQAccumulator  # noqa: E402
from src.rle import mask_to_rle, rle_to_mask  # noqa: E402


def cache_gt(val_stems, by_stem, run_dir: Path):
    gt_file = run_dir / "gt_cache.npz"
    if gt_file.exists():
        z = np.load(gt_file, allow_pickle=True)
        return {k: list(z[k]) for k in z.files}
    table: dict[str, list[str]] = {}
    for stem in val_stems:
        for k, ann in enumerate(by_stem[stem]):
            table[f"{stem}#{k}"] = [mask_to_rle(i.mask()) for i in ann.instances]
    np.savez_compressed(gt_file, **table)
    return table


def eval_variant(val_stems, gt_cache, prob_dir, instancer, tag, quiet=False):
    acc = PQAccumulator()
    for stem in val_stems:
        prob = np.load(prob_dir / f"{stem}.npy").astype(np.float32)
        preds = instancer(prob)
        keys = [k for k in gt_cache if k.split("#")[0] == stem]
        for k in keys:
            gts = [rle_to_mask(r) for r in gt_cache[k]]
            acc.add(stem, gts, preds)
    r = acc.result()
    if not quiet:
        print(f"{tag:44s} PQ={r.pq:.4f} SQ={r.sq:.4f} RQ={r.rq:.4f} "
              f"TP={r.tp} FP={r.fp} FN={r.fn}", flush=True)
    return r


def hysteresis(prob: np.ndarray, hi: float, lo: float) -> np.ndarray:
    seeds = (prob > hi).astype(np.uint8)
    low = (prob > lo).astype(np.uint8)
    n, lab = cv2.connectedComponents(low, connectivity=8)
    seed_labels = set(np.unique(lab[seeds > 0])) - {0}
    return np.isin(lab, list(seed_labels)).astype(np.uint8) if seed_labels else np.zeros_like(low)


def instances_from_mask(mask, min_area, close=0, watershed=False,
                        dist_thr_frac=0.45, min_split_area=1500):
    if close:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    n, lab = cv2.connectedComponents(mask, 8)
    out = []
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        area = int(comp.sum())
        if area < min_area:
            continue
        if watershed and area >= min_split_area:
            dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
            _, thr = cv2.threshold(dist, dist.max() * dist_thr_frac, 255, cv2.THRESH_BINARY)
            nn, markers = cv2.connectedComponents(thr.astype(np.uint8), 8)
            if nn > 2:
                markers = cv2.watershed(cv2.cvtColor(comp * 255, cv2.COLOR_GRAY2BGR), markers)
                for j in range(1, nn):
                    sub = ((markers == j).astype(np.uint8)) & comp
                    if sub.sum() >= min_area:
                        out.append(sub)
                continue
        out.append(comp)
    return out


def main(run_dir: Path):
    from src.eval_and_submit import load_image
    from src.train_unet import build_model, get_device, predict_full

    prob_dir = run_dir / "probs"
    prob_dir.mkdir(exist_ok=True)
    folds = json.load(open(run_dir / "folds.json"))
    val_stems = sorted(s for s, f in folds.items() if f == 0)
    missing = [s for s in val_stems if not (prob_dir / f"{s}.npy").exists()]
    if missing:
        import torch
        device = get_device()
        model = build_model().to(device)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
        model.eval()
        for k, stem in enumerate(missing):
            np.save(prob_dir / f"{stem}.npy",
                    predict_full(model, load_image(stem), device).astype(np.float16))
            if (k + 1) % 25 == 0:
                print(f"  prob-cached {k+1}/{len(missing)}", flush=True)

    by_stem = group_by_stem(load_annotations())
    gt_cache = cache_gt(val_stems, by_stem, run_dir)
    print(f"GT cache: {len(gt_cache)} annotator-sets", flush=True)

    done = set()
    res_file = run_dir / "postprocess_results.txt"
    if res_file.exists():
        for line in open(res_file):
            if line[:4].strip() and not line.startswith(("-", "G")):
                done.add(" ".join(line.split()[:2]))

    def already(tag):
        return " ".join(tag.split()[:2]) in done

    print("--- plain CC: thr x min_area ---", flush=True)
    for thr in (0.4, 0.5):
        for ma in (700, 1000, 1500):
            tag = f"CC thr={thr} ma={ma}"
            if already(tag):
                continue
            eval_variant(val_stems, gt_cache, prob_dir,
                         lambda p, t=thr, m=ma: instances_from_mask((p > t).astype(np.uint8), m),
                         tag)
    print("--- hysteresis ---", flush=True)
    for hi, lo in ((0.6, 0.3), (0.7, 0.35), (0.5, 0.25), (0.65, 0.4)):
        for ma in (300, 500, 700):
            tag = f"hyst hi={hi} lo={lo} ma={ma}"
            if already(tag):
                continue
            eval_variant(val_stems, gt_cache, prob_dir,
                         lambda p, h=hi, l=lo, m=ma: instances_from_mask(
                             hysteresis(p, h, l), m),
                         tag)
    print("--- hysteresis + watershed ---", flush=True)
    for hi, lo in ((0.6, 0.3), (0.7, 0.35)):
        tag = f"hyst+ws hi={hi} lo={lo} ma=300"
        if already(tag):
            continue
        eval_variant(val_stems, gt_cache, prob_dir,
                     lambda p, h=hi, l=lo: instances_from_mask(
                         hysteresis(p, h, l), 300, watershed=True),
                     tag)
    tag = "CC0.4+ws ma=300"
    if not already(tag):
        eval_variant(val_stems, gt_cache, prob_dir,
                     lambda p: instances_from_mask((p > 0.4).astype(np.uint8), 300,
                                                   watershed=True),
                     tag)
    tag = "CC0.4+close3+ws ma=300"
    if not already(tag):
        eval_variant(val_stems, gt_cache, prob_dir,
                     lambda p: instances_from_mask((p > 0.4).astype(np.uint8), 300,
                                                   close=3, watershed=True),
                     tag)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/unet_r34_v1_fold0"))
