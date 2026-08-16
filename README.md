# Solar Filament Segmentation 2026 — bcap52

Solution repository for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(NSO / IEEE BigData Cup). Instance segmentation of solar filaments in GONG H-alpha
full-disk images (2048×2048), evaluated with Panoptic Quality (PQ) plus IoU/Dice
distributions and relation counts (rubric: 70% quantitative / 30% qualitative).

## Results so far (see PROJECT_LOG.md for the full history)

| Model | Local OOF PQ | Kaggle LB |
|---|---|---|
| Classical local-contrast + CC | 0.130 | 0.12 |
| U-Net ResNet34, native-res crops, fold 0, CC thr0.5 ma500 | 0.353 | submitted 2026-08-16 |
| Expert vs expert (annotator ceiling) | 0.332 | — |
| Union-of-annotators consensus + CC (upper probe) | 0.619 | — |

## Approach

1. **Official-metric local evaluation** — exact port of the organizer's
   Self_Evaluation_Notebook semantics (`src/pq.py`): per-annotator-set GT rows,
   all-pairs IoU>0.5 matching, PQ = ΣIoU(TP)/(|TP|+0.5|FP|+0.5|FN|).
2. **Multi-annotator-aware training** — every annotator-image set (1154 rows over
   707 JPEGs) is a training sample; the model learns the annotator consensus.
   Folds are grouped by physical JPEG (no leakage across annotators of one frame).
3. **Native-resolution U-Net** (ResNet34 encoder) trained on random 1024² crops at
   native 2048² scale to preserve thin barbs; tiled 3×3 inference, CC instancing,
   threshold + min-area chosen by out-of-fold PQ.
4. **PQ-oriented post-processing** — disjoint instances (competition rejects
   overlapping predictions), min-area cleanup, size calibration.

## Repository layout

```
src/
  data.py                 # COCO-ish multi-annotator loading, grouping by JPEG stem
  rle.py                  # compressed COCO RLE I/O (submission format)
  pq.py                   # official PQ scorer port (streaming accumulator)
  eda.py                  # annotation statistics, inter-annotator PQ ceiling
  validate.py             # OOF validation harness (segmenter -> streaming PQ)
  baseline_classical.py   # local-contrast dark-feature baseline (no ML)
  train_unet.py           # U-Net training (Intel XPU / CUDA / CPU)
  eval_and_submit.py      # OOF PQ sweep + test submission
  postprocess_fast.py     # instancing experiment grid (cached GT/probs)
  build_gt_cache.py       # pre-rasterized RLE targets (fast dataloading)
  make_test_sub.py        # fixed-recipe test submission
PROJECT_LOG.md            # chronological development log
```

## Data

Competition data (MAGFiLO 1.0 Kaggle 2026 release) is NOT redistributed here.
Download from the competition page. Only the provided train annotations are used
(organizer clarification 2026-07-27: public MAGFiLO annotations beyond the
competition JSON are forbidden).

## Environment

Python ≥3.12, torch (XPU build for Intel Arc, or CUDA), opencv, pycocotools,
pandas, scipy, matplotlib, segmentation-models-pytorch.

## License

MIT (code). Competition data remains under the competition's terms.
