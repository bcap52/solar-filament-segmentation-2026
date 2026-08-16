"""Evaluate a trained fold checkpoint: OOF PQ sweep + optional test submission.

Usage: python src/eval_and_submit.py runs/unet_r34_v1_fold0 [--submit] [--thr 0.5]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import TEST_IMG_DIR, TRAIN_IMG_DIR, group_by_stem  # noqa: E402
from src.data import load_annotations, test_stems  # noqa: E402
from src.pq import PQAccumulator  # noqa: E402
from src.rle import mask_to_rle  # noqa: E402
from src.train_unet import build_model, get_device, instances_from_prob  # noqa: E402
from src.train_unet import predict_full  # noqa: E402


def load_image(stem: str, train: bool = True) -> np.ndarray:
    d = TRAIN_IMG_DIR if train else TEST_IMG_DIR
    img = cv2.imread(str(d / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img.astype(np.float32) - p1) / max(p99 - p1, 1.0), 0, 1).astype(np.float32)


def main(run_dir: Path, submit: bool = False, thr: float | None = None,
         min_area: int = 150):
    import torch
    import pandas as pd
    device = get_device()
    model = build_model().to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()

    by_key = load_annotations()
    by_stem = group_by_stem(by_key)
    folds = json.load(open(run_dir / "folds.json"))
    val_stems = sorted(s for s, f in folds.items() if f == 0)

    print(f"OOF PQ on {len(val_stems)} val stems ({run_dir.name}):")
    probs = {}
    for k, stem in enumerate(val_stems):
        probs[stem] = predict_full(model, load_image(stem), device)
        if (k + 1) % 25 == 0:
            print(f"  predicted {k+1}/{len(val_stems)}")

    sweep = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        acc = PQAccumulator()
        for stem in val_stems:
            preds = instances_from_prob(probs[stem], t, min_area)
            for ann in by_stem[stem]:
                acc.add(stem, [i.mask() for i in ann.instances], preds)
        r = acc.result()
        sweep[t] = r
        print(f"  thr={t:.1f}: PQ={r.pq:.4f} SQ={r.sq:.4f} RQ={r.rq:.4f} "
              f"TP={r.tp} FP={r.fp} FN={r.fn}")
    best_thr = max(sweep, key=lambda t: sweep[t].pq)
    print(f"best thr = {best_thr} (PQ {sweep[best_thr].pq:.4f})")
    json.dump({str(t): dict(pq=r.pq, sq=r.sq, rq=r.rq, tp=r.tp, fp=r.fp, fn=r.fn)
               for t, r in sweep.items()},
              open(run_dir / "oof_pq.json", "w"), indent=1)

    if submit:
        t = thr if thr is not None else best_thr
        rows = []
        for stem in test_stems():
            prob = predict_full(model, load_image(stem, train=False), device)
            for j, m in enumerate(instances_from_prob(prob, t, min_area), 1):
                rows.append({"filament_id": f"{stem}_{j}", "segmentation_rle": mask_to_rle(m)})
        out = Path("subs") / f"{run_dir.name}_thr{t:.2f}.csv"
        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(out, index=False)
        print(f"wrote {out}: {len(rows)} rows")


if __name__ == "__main__":
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/unet_r34_v1_fold0")
    do_submit = "--submit" in sys.argv
    thr = None
    if "--thr" in sys.argv:
        thr = float(sys.argv[sys.argv.index("--thr") + 1])
    main(run, do_submit, thr)
