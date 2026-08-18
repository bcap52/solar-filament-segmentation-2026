"""Mask R-CNN family: OOF system PQ + ensemble test submission (mirrors ensemble.py).

Each stem scored by its out-of-fold detector; test = union of all folds'
detections, disjoint-merged in arrival order.

Usage:
  python -u src/ensemble_maskrcnn.py eval [thr]     # system PQ over present folds
  python -u src/ensemble_maskrcnn.py submit [thr]   # test CSV
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import group_by_stem, load_annotations, test_stems, train_stems  # noqa: E402
from src.pq import PQAccumulator  # noqa: E402
from src.rle import mask_to_rle  # noqa: E402
from src.train_maskrcnn import predict_full_det  # noqa: E402
from src.train_unet import get_device, make_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def det_oof_pq(thr: float, device, folds_present: set[int]) -> float:
    by_stem = group_by_stem(load_annotations())
    folds = make_folds(train_stems())
    acc = PQAccumulator()
    for fold in sorted(folds_present):
        import torch
        from src.train_maskrcnn import build
        run_dir = ROOT / f"runs/maskrcnn_v1_fold{fold}"
        model = build().to(device)
        model.load_state_dict(torch.load(run_dir / "last.pt", map_location=device))
        model.eval()
        for stem in [s for s in train_stems() if folds[s] == fold]:
            preds = predict_full_det(model, stem, device, thr)
            for ann in by_stem[stem]:
                acc.add(stem, [i.mask() for i in ann.instances], preds)
    r = acc.result()
    print(f"DET-SYSTEM thr={thr} ({len(folds_present)} folds): PQ={r.pq:.4f} "
          f"SQ={r.sq:.4f} RQ={r.rq:.4f} TP={r.tp} FP={r.fp} FN={r.fn}", flush=True)
    return r.pq


def det_submit(thr: float, device, folds_present: set[int]):
    import pandas as pd
    import torch
    from src.train_maskrcnn import build
    models = []
    for fold in sorted(folds_present):
        run_dir = ROOT / f"runs/maskrcnn_v1_fold{fold}"
        m = build().to(device)
        m.load_state_dict(torch.load(run_dir / "last.pt", map_location=device))
        m.eval()
        models.append(m)
    rows = []
    for stem in test_stems():
        seen = []
        for m in models:
            seen.extend(predict_full_det(m, stem, device, thr))
        for i in range(len(seen) - 1, 0, -1):
            for j in range(i):
                seen[i] &= (1 - seen[j])
        seen = [m for m in seen if m.sum() >= 500]
        for j, m in enumerate(seen, 1):
            rows.append({"filament_id": f"{stem}_{j}", "segmentation_rle": mask_to_rle(m)})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out = ROOT / "subs" / f"maskrcnn_ens_thr{thr}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows")


if __name__ == "__main__":
    device = get_device()
    present = {f for f in range(5)
               if (ROOT / f"runs/maskrcnn_v1_fold{f}/last.pt").exists()}
    print("folds present:", sorted(present))
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    if mode == "eval":
        for t in (0.2, 0.3, 0.4, 0.5):
            det_oof_pq(t, device, present)
    else:
        det_submit(thr, device, present)
