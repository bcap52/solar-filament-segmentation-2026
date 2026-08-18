"""Mask R-CNN instance detection track (C3).

torchvision maskrcnn_resnet50_fpn (v1, ImageNet/COCO pretrain) trained on
1024-res full disks, targets = each annotator-image set's instances (boxes +
masks). Inference: score threshold, per-mask single-CC, min-area 500, disjoint
enforcement by arrival order. Evaluated with the official PQ (src/pq.py).

Memory notes (10 GB Intel Arc B570): v2 backbone OOMs; v1 fits at batch 1 fp16
(~500-580 s/epoch); batch 2 enabled after fold-0 readout justified the track
(revert BATCH to 1 if OOM recurs). torchvision roi_align + maskrcnn forward
verified working on XPU (tv 0.28.0+xpu).
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_annotations, group_by_stem, train_stems  # noqa: E402
from src.eval_and_submit import load_image  # noqa: E402
from src.pq import PQAccumulator  # noqa: E402
from src.rle import rle_to_mask  # noqa: E402
from src.train_unet import get_device, make_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260817
IMSIZE = 1024
EPOCHS = 18
BATCH = 2  # v1 fits batch2 on 10GB (verify at runtime; revert to 1 if OOM)
LR = 2.5e-4


class DetDataset(Dataset):
    def __init__(self, items, train=True):
        self.items = items  # AnnImage list
        self.train = train
        z = np.load(ROOT / "runs/gt_semantic_rle.npz", allow_pickle=True)
        zi = np.load(ROOT / "runs/gt_semantic_rle.npz", allow_pickle=True)
        self.inst = {k[6:]: str(zi[k]) for k in zi.files if k.startswith("inst::")}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ai = self.items[idx]
        img = load_image(ai.stem)
        img = cv2.resize(img, (IMSIZE, IMSIZE))
        masks = []
        for i in ai.instances:
            m = rle_to_mask(self.inst[i.ann_id])
            masks.append(cv2.resize(m, (IMSIZE, IMSIZE), interpolation=cv2.INTER_NEAREST))
        if self.train:
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k); masks = [np.rot90(m, k) for m in masks]
            if random.random() < 0.5:
                img = img[:, ::-1]; masks = [m[:, ::-1] for m in masks]
            if random.random() < 0.5:
                a = random.uniform(0.85, 1.15); img = np.clip(img * a, 0, 1)
        img = np.ascontiguousarray(img)
        boxes, keep = [], []
        for m in masks:
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
            keep.append(m)
        target = dict(
            boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            labels=torch.ones(len(keep), dtype=torch.int64),
            masks=torch.tensor(np.stack(keep) if keep else np.zeros((0, IMSIZE, IMSIZE), np.uint8)),
        )
        return torch.from_numpy(img).repeat(3, 1, 1), target


def collate(batch):
    return tuple(zip(*batch))


def build():
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    return maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)


def predict_full_det(model, stem: str, device, score_thr=0.3):
    img = load_image(stem)
    small = cv2.resize(img, (IMSIZE, IMSIZE))
    x = torch.from_numpy(np.ascontiguousarray(small)).repeat(3, 1, 1)[None].to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)[0]
    keep = out["scores"] > score_thr
    masks = (out["masks"][keep, 0] > 0.5).cpu().numpy().astype(np.uint8)
    insts = []
    for m in masks:
        m2 = cv2.resize(m, (2048, 2048), interpolation=cv2.INTER_NEAREST)
        n, lab = cv2.connectedComponents(m2, 8)
        if n < 2:
            continue
        areas = [(lab == i).sum() for i in range(1, n)]
        mm = (lab == 1 + int(np.argmax(areas))).astype(np.uint8)
        if mm.sum() >= 500:
            insts.append(mm)
    for i in range(len(insts) - 1, 0, -1):
        for j in range(i):
            insts[i] &= (1 - insts[j])
        if insts[i].sum() < 500:
            insts[i] = np.zeros_like(insts[i])
    return [m for m in insts if m.sum() >= 500]


def main(fold=0, epochs=EPOCHS):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = get_device()
    run_dir = ROOT / f"runs/maskrcnn_v1_fold{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    folds = make_folds(train_stems())
    json.dump(folds, open(run_dir / "folds.json", "w"))
    by_key = load_annotations()
    train_items = [ai for ai in by_key.values() if folds[ai.stem] != fold]
    val_stems = sorted(s for s, f in folds.items() if f == fold)
    print(f"train {len(train_items)} val {len(val_stems)} device {device}", flush=True)

    model = build().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    dl = DataLoader(DetDataset(train_items), batch_size=BATCH, shuffle=True,
                    num_workers=2, persistent_workers=True, prefetch_factor=4,
                    collate_fn=collate, drop_last=True)
    scaler = torch.amp.GradScaler(device.type, enabled=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl))

    for ep in range(epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for imgs, targets in dl:
            imgs = [i.to(device) for i in imgs]
            tg = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if any(t["boxes"].numel() == 0 for t in tg):
                continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16):
                losses = model(imgs, tg)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss.detach()); nb += 1
        print(f"ep{ep} loss {tot/max(nb,1):.3f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), run_dir / "last.pt")

    by_stem = group_by_stem(by_key)
    for thr in (0.3, 0.5):
        acc = PQAccumulator()
        for stem in val_stems[:50]:
            preds = predict_full_det(model, stem, device, thr)
            for ann in by_stem[stem]:
                acc.add(stem, [i.mask() for i in ann.instances], preds)
        r = acc.result()
        print(f"DET fold{fold} thr={thr} (50 stems): PQ={r.pq:.4f} SQ={r.sq:.4f} "
              f"RQ={r.rq:.4f} TP={r.tp} FP={r.fp} FN={r.fn}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0,
         int(sys.argv[2]) if len(sys.argv) > 2 else EPOCHS)
