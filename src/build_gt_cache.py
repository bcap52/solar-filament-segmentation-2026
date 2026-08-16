"""Pre-rasterize all annotator-set semantic masks to RLE (training speedup).

Writes runs/gt_semantic_rle.npz: keys sem::<image_key> (semantic union of that
annotator set) and inst::<ann_id> (per-instance masks), compressed-RLE strings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_annotations  # noqa: E402
from src.rle import mask_to_rle  # noqa: E402


def main(out: Path = Path("runs/gt_semantic_rle.npz")):
    by_key = load_annotations()
    sem: dict[str, str] = {}
    inst: dict[str, str] = {}
    for k, (key, ai) in enumerate(by_key.items()):
        m = np.zeros((2048, 2048), np.uint8)
        for i in ai.instances:
            im = i.mask()
            inst[i.ann_id] = mask_to_rle(im)
            m |= im
        sem[key] = mask_to_rle(m)
        if (k + 1) % 150 == 0:
            print(f"{k+1}/{len(by_key)}", flush=True)
    np.savez_compressed(out, semantic=np.array(list(sem.keys())),
                        **{f"sem::{k}": v for k, v in sem.items()},
                        **{f"inst::{k}": v for k, v in inst.items()})
    print(f"wrote {out}: {len(sem)} semantic + {len(inst)} instance RLEs")


if __name__ == "__main__":
    main()
