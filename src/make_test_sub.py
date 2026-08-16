"""Generate test submission from a fold checkpoint with a fixed instancing recipe.

Usage: python src/make_test_sub.py runs/unet_r34_v1_fold0 0.5 500 [--ws]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import test_stems  # noqa: E402
from src.eval_and_submit import load_image  # noqa: E402
from src.postprocess_fast import instances_from_mask  # noqa: E402
from src.rle import mask_to_rle  # noqa: E402
from src.train_unet import build_model, get_device, predict_full  # noqa: E402


def main(run_dir: Path, thr: float, min_area: int, close: int = 0, watershed: bool = False):
    import torch
    device = get_device()
    model = build_model().to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()

    rows = []
    for k, stem in enumerate(test_stems()):
        prob = predict_full(model, load_image(stem, train=False), device)
        masks = instances_from_mask((prob > thr).astype(np.uint8), min_area,
                                    close=close, watershed=watershed)
        for j, m in enumerate(masks, 1):
            rows.append({"filament_id": f"{stem}_{j}", "segmentation_rle": mask_to_rle(m)})
        if (k + 1) % 30 == 0:
            print(f"  {k+1}/180 images, {len(rows)} rows", flush=True)
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out = Path("subs") / f"{run_dir.name}_thr{thr}_ma{min_area}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows over {df['filament_id'].str.split('_').str[0].nunique()} images")


if __name__ == "__main__":
    run = Path(sys.argv[1])
    thr = float(sys.argv[2])
    ma = int(sys.argv[3])
    ws = "--ws" in sys.argv
    main(run, thr, ma, watershed=ws)
