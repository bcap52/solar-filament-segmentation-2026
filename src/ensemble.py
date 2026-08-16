"""5-fold ensemble: OOF system PQ + test submission.

OOF: each stem scored by the fold model that did NOT train on it (proper
out-of-fold estimate of the full system). Test: mean of 5 fold models' sigmoid
probability maps (optionally D4 TTA), instanced by the best known recipe
(CC + min_area). All probability maps cached to disk — power-cut resilient.

Usage:
  python -u src/ensemble.py eval           # OOF PQ sweep over thresholds
  python -u src/ensemble.py submit 0.45 500 [--tta]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import group_by_stem, load_annotations, test_stems, train_stems  # noqa: E402
from src.eval_and_submit import load_image  # noqa: E402
from src.pq import PQAccumulator  # noqa: E402
from src.postprocess_fast import instances_from_mask  # noqa: E402
from src.train_unet import build_model, get_device, predict_full  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_model(run_dir: Path, device):
    import torch
    m = build_model().to(device)
    m.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    m.eval()
    return m


def oof_probs(fold: int, device) -> Path:
    """Cache OOF probability maps for fold's val stems; returns cache dir."""
    run_dir = ROOT / f"runs/unet_r34_v1_fold{fold}"
    pdir = run_dir / "probs"
    pdir.mkdir(exist_ok=True)
    folds = json.load(open(run_dir / "folds.json"))
    val_stems = sorted(s for s, f in folds.items() if f == fold)
    missing = [s for s in val_stems if not (pdir / f"{s}.npy").exists()]
    if missing:
        model = load_model(run_dir, device)
        for k, stem in enumerate(missing):
            np.save(pdir / f"{stem}.npy",
                    predict_full(model, load_image(stem), device).astype(np.float16))
            if (k + 1) % 25 == 0:
                print(f"  fold{fold} OOF probs {k+1}/{len(missing)}", flush=True)
    return pdir


def oof_system_pq(thr: float, min_area: int, device) -> float:
    """PQ over all 707 stems, each scored by its out-of-fold model."""
    by_stem = group_by_stem(load_annotations())
    all_stems = train_stems()
    fold_of = json.load(open(ROOT / "runs/unet_r34_v1_fold0/folds.json"))
    acc = PQAccumulator()
    for fold in range(5):
        pdir = oof_probs(fold, device)
        for stem in [s for s in all_stems if fold_of[s] == fold]:
            prob = np.load(pdir / f"{stem}.npy").astype(np.float32)
            preds = instances_from_mask((prob > thr).astype(np.uint8), min_area)
            for ann in by_stem[stem]:
                acc.add(stem, [i.mask() for i in ann.instances], preds)
    r = acc.result()
    print(f"OOF SYSTEM PQ thr={thr} ma={min_area}: PQ={r.pq:.4f} SQ={r.sq:.4f} "
          f"RQ={r.rq:.4f} TP={r.tp} FP={r.fp} FN={r.fn}", flush=True)
    return r.pq


def build_ensemble_submission(thr: float, min_area: int, device, tta: bool = False):
    import pandas as pd
    from src.rle import mask_to_rle
    models = []
    for fold in range(5):
        run_dir = ROOT / f"runs/unet_r34_v1_fold{fold}"
        if not (run_dir / "best.pt").exists():
            print(f"WARNING: fold {fold} checkpoint missing — ensemble of available folds")
            continue
        models.append(load_model(run_dir, device))
    assert models, "no fold checkpoints found"
    print(f"ensembling {len(models)} fold models (tta={tta})")

    rows = []
    edir = ROOT / "runs/ensemble_test_probs"
    edir.mkdir(exist_ok=True)
    for k, stem in enumerate(test_stems()):
        fp = edir / f"{stem}.npy"
        if fp.exists():
            prob = np.load(fp).astype(np.float32)
        else:
            img = load_image(stem, train=False)
            probs = []
            for m in models:
                probs.append(predict_full(m, img, device))
                if tta:
                    probs.append(predict_full(m, img[:, ::-1].copy(), device)[:, ::-1])
                    probs.append(predict_full(m, img[::-1, :].copy(), device)[::-1, :])
                    probs.append(predict_full(m, img.T.copy(), device).T)
            prob = np.mean(probs, axis=0)
            np.save(fp, prob.astype(np.float16))
        masks = instances_from_mask((prob > thr).astype(np.uint8), min_area)
        for j, m in enumerate(masks, 1):
            rows.append({"filament_id": f"{stem}_{j}", "segmentation_rle": mask_to_rle(m)})
        if (k + 1) % 20 == 0:
            print(f"  test {k+1}/180, rows so far {len(rows)}", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out = ROOT / "subs" / f"ensemble5_thr{thr}_ma{min_area}{'_tta' if tta else ''}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows")
    return out


if __name__ == "__main__":
    device = get_device()
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
    ma = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    if mode == "eval":
        for t in (0.35, 0.4, 0.45, 0.5):
            oof_system_pq(t, ma, device)
    elif mode == "submit":
        tta = "--tta" in sys.argv
        build_ensemble_submission(thr, ma, device, tta=tta)
