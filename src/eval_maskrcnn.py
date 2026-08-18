"""Evaluate a trained Mask R-CNN checkpoint: full OOF PQ sweep + optional test CSV.

Usage:
  python -u src/eval_maskrcnn.py 0                # full val sweep on fold 0
  python -u src/eval_maskrcnn.py 0 --submit 0.3   # test submission
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import group_by_stem, load_annotations, test_stems, train_stems  # noqa: E402
from src.pq import PQAccumulator  # noqa: E402
from src.rle import mask_to_rle  # noqa: E402
from src.train_maskrcnn import predict_full_det  # noqa: E402
from src.train_unet import get_device, make_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(fold: int, submit: bool = False, submit_thr: float = 0.3):
    import torch
    device = get_device()
    run_dir = ROOT / f"runs/maskrcnn_v1_fold{fold}"
    from src.train_maskrcnn import build
    model = build().to(device)
    model.load_state_dict(torch.load(run_dir / "last.pt", map_location=device))
    model.eval()

    folds = make_folds(train_stems())
    val_stems = sorted(s for s, f in folds.items() if f == fold)
    by_stem = group_by_stem(load_annotations())

    sweep = {}
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6):
        acc = PQAccumulator()
        for stem in val_stems:
            preds = predict_full_det(model, stem, device, thr)
            for ann in by_stem[stem]:
                acc.add(stem, [i.mask() for i in ann.instances], preds)
        r = acc.result()
        sweep[thr] = r
        print(f"DET fold{fold} thr={thr}: PQ={r.pq:.4f} SQ={r.sq:.4f} RQ={r.rq:.4f} "
              f"TP={r.tp} FP={r.fp} FN={r.fn}", flush=True)
    json.dump({str(t): dict(pq=r.pq, sq=r.sq, rq=r.rq, tp=r.tp, fp=r.fp, fn=r.fn)
               for t, r in sweep.items()}, open(run_dir / "oof_pq.json", "w"), indent=1)

    if submit:
        import pandas as pd
        rows = []
        for stem in test_stems():
            for j, m in enumerate(predict_full_det(model, stem, device, submit_thr), 1):
                rows.append({"filament_id": f"{stem}_{j}", "segmentation_rle": mask_to_rle(m)})
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        out = ROOT / "subs" / f"maskrcnn_v1_fold{fold}_thr{submit_thr}.csv"
        df.to_csv(out, index=False)
        print(f"wrote {out}: {len(df)} rows")


if __name__ == "__main__":
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sub = "--submit" in sys.argv
    thr = 0.3
    if sub and sys.argv.index("--submit") + 1 < len(sys.argv):
        thr = float(sys.argv[sys.argv.index("--submit") + 1])
    main(fold, sub, thr)
