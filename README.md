# Solar Filament Segmentation Challenge 2026 — Solution

Instance segmentation of solar filaments in GONG H-alpha 2048×2048 filtergrams
(NSO / IEEE BigData Cup 2026). Metric: **Panoptic Quality** (per-annotator,
IoU>0.5 matching, disjoint predictions).

## Results (out-of-fold system PQ on grouped folds; LB = public leaderboard)

| Configuration | OOF PQ | LB |
|---|---|---|
| Classical CV baseline | 0.130 | 0.12 |
| U-Net ResNet34, single fold | 0.354 | 0.30 |
| 5-fold ensemble | 0.357 | 0.32 |
| + boundary-weighted BCE (bw=4) | 0.368 | **0.33** |
| + boundary weight 8 (best) | **0.373** | **0.33** |

Full ablation ledger (17 experiments incl. negative results) in `docs/PIPELINE.md`.

## Approach

1. **Multi-annotator-aware validation**: GroupKFold by physical image stem;
   OOF system PQ scored against *every* annotator set with the official metric.
2. **Physics-informed inputs** (optional channels): limb-darkening-corrected
   contrast (azimuthal-median radial profile), multi-scale Frangi vesselness,
   and **GONG LOS magnetograms** registered to the H-alpha disk geometry
   (nearest-in-time same-site, 10-min cadence, cross-site fallback).
3. **Boundary-weighted BCE**: W = 1 + 8·1[boundary band] — thin-structure
   upweighting (filaments are ~1–12 px wide); measured RQ 0.530→0.557.
4. **Topological losses** (optional): stabilized soft-clDice (anti-fragmentation)
   + truncated signed-distance boundary loss.
5. **Tiled native-res inference** (3×3 tiles stride 512, CPU float64 accumulation
   — XPU in-place slice-add is unreliable), CC instancing + min-area 500.

## Repository layout

- `src/` — full pipeline (data, metric port, trainers, eval, submission, builders)
- `docs/PIPELINE.md` — micro-level pipeline description (qualitative report source)
- `run_folds_local.sh`, `run_union_folds.sh`, `night_pipeline.sh` — orchestration

## Reproduce

```bash
pip install --user torch torchvision --index-url https://download.pytorch.org/whl/xpu
pip install --user pycocotools opencv-python-headless pandas matplotlib \
    segmentation-models-pytorch scikit-image astropy kaggle
python -u src/build_gt_cache.py
python -u src/build_f16_stack.py            # normalized f16 image stack
python -u src/build_features_stack.py       # contrast + Frangi channels
python -u src/build_mag_stack.py            # GONG magnetograms (FTP)
python -u src/train_unet.py 0 24 per_annotator resnet34 unet_r34_bnd8 1024 4 8.0
RUN_PREFIX=unet_r34_bnd8 python -u src/ensemble.py eval
RUN_PREFIX=unet_r34_bnd8 python -u src/ensemble.py submit 0.4 500 --tta
```

Hardware: Intel Arc B570 (10 GB), torch 2.13.0+xpu, fp16 autocast.

## License

MIT
