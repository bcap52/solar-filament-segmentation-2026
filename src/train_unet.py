"""U-Net semantic segmentation training for filament PQ.

Samples = annotator-image sets (1154). Each epoch draws ONE random native-res
1024x1024 crop per sample (70% foreground-biased). Target = semantic union of
that annotator set's instances (pre-rasterized RLE cache). GroupKFold(5) by
physical JPEG stem. SMP UNet + ResNet34 (ImageNet). BCE + soft Dice, AMP,
cosine LR. Device: xpu > cuda > cpu.

NOTE (XPU): tiled inference accumulates in numpy on CPU — torch 2.13.0+xpu's
in-place slice-add path was observed returning corrupt values.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import TRAIN_IMG_DIR, load_annotations, group_by_stem, train_stems  # noqa: E402
from src.pq import PQAccumulator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "runs"
SEED = 20260816
CROP = 1024
EPOCHS = 24
BATCH = 4
LR = 3e-4
VAL_FOLD = 0
MIN_AREA = 150


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def get_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_folds(stems: list[str], n_folds: int = 5, seed: int = SEED) -> dict[str, int]:
    rng = random.Random(seed)
    stems = sorted(stems)
    rng.shuffle(stems)
    return {s: i % n_folds for i, s in enumerate(stems)}


class FilamentCropDataset(Dataset):
    """One sample = (annotator-image, random native-res crop)."""

    def __init__(self, ann_images: list, img_cache: dict, crop: int = CROP,
                 train: bool = True, gt_sem_rle: dict | None = None):
        self.items = ann_images
        self.img_cache = img_cache
        self.crop = crop
        self.train = train
        self.gt_sem_rle = gt_sem_rle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from src.rle import rle_to_mask
        ai = self.items[idx]
        raw, p1, p99 = self.img_cache[ai.stem]
        img = np.clip((raw.astype(np.float32) - p1) / (p99 - p1), 0, 1)
        if self.gt_sem_rle is not None:
            mask = rle_to_mask(self.gt_sem_rle[ai.image_key]).astype(np.float32)
        else:
            mask = np.zeros((2048, 2048), np.float32)
            for inst in ai.instances:
                mask[inst.mask() > 0] = 1.0

        c = self.crop
        if self.train:
            fg = np.nonzero(mask)
            if len(fg[0]) > 0 and random.random() < 0.7:
                cy = int(np.clip(random.choice(fg[0]) + random.randint(-c // 2, c // 2), 0, 2048 - c))
                cx = int(np.clip(random.choice(fg[1]) + random.randint(-c // 2, c // 2), 0, 2048 - c))
            else:
                cy = random.randint(0, 2048 - c)
                cx = random.randint(0, 2048 - c)
        else:
            cy = cx = (2048 - c) // 2
        x = img[cy:cy + c, cx:cx + c].copy()
        y = mask[cy:cy + c, cx:cx + c].copy()

        if self.train:
            if random.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            if random.random() < 0.5:
                x, y = x[::-1], y[::-1]
            k = random.randint(0, 3)
            if k:
                x, y = np.rot90(x, k), np.rot90(y, k)
            if random.random() < 0.5:
                a = random.uniform(0.85, 1.15); b = random.uniform(-0.08, 0.08)
                x = np.clip(x * a + b, 0, 1)
            if random.random() < 0.3:
                x = np.clip(x + np.random.normal(0, 0.02, x.shape).astype(np.float32), 0, 1)
        return np.ascontiguousarray(x)[None], np.ascontiguousarray(y)[None]


def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    sum_ = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + eps) / (sum_ + eps)).mean()


def build_model():
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                    in_channels=1, classes=1, activation=None)


def predict_full(model, img: np.ndarray, device, crop: int = CROP) -> np.ndarray:
    """Tiled inference at native resolution: 3x3 tiles of 1024, stride 512, avg
    logits -> sigmoid probability map. Accumulation in numpy on CPU (XPU bug)."""
    h = w = 2048
    logits_sum = np.zeros((h, w), np.float64)
    count = np.zeros((h, w), np.float64)
    model.eval()
    with torch.no_grad():
        for oy in range(0, h - crop + 1, crop // 2):
            for ox in range(0, w - crop + 1, crop // 2):
                tile = torch.from_numpy(np.ascontiguousarray(
                    img[oy:oy + crop, ox:ox + crop], dtype=np.float32)[None, None]).to(device)
                out = model(tile)[0, 0].float().cpu().numpy()
                logits_sum[oy:oy + crop, ox:ox + crop] += out
                count[oy:oy + crop, ox:ox + crop] += 1
    avg = (logits_sum / count).astype(np.float32)
    return 1.0 / (1.0 + np.exp(-avg))


def instances_from_prob(prob: np.ndarray, thr: float, min_area: int = MIN_AREA) -> list[np.ndarray]:
    cand = (prob > thr).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(cand, connectivity=8)
    return [(lab == i).astype(np.uint8) for i in range(1, n)
            if (lab == i).sum() >= min_area]


def main(fold: int = VAL_FOLD, epochs: int = EPOCHS, tag: str = "unet_r34_v1"):
    set_seed(SEED)
    device = get_device()
    print(f"device: {device}")
    run_dir = CKPT_DIR / f"{tag}_fold{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    by_key = load_annotations()
    folds = make_folds(train_stems())
    json.dump(folds, open(run_dir / "folds.json", "w"), indent=1)

    train_items = [ai for ai in by_key.values() if folds[ai.stem] != fold]
    val_items = [ai for ai in by_key.values() if folds[ai.stem] == fold]
    print(f"train ann-images: {len(train_items)}")

    print("caching images (train+val) as uint8 + percentile stats...")
    img_cache = {}
    for stem in sorted({ai.stem for ai in by_key.values()}):
        img = cv2.imread(str(TRAIN_IMG_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
        img_cache[stem] = (img, float(p1), float(max(p99, p1 + 1)))

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train_items) / BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type != "cpu"))
    ctx = (torch.autocast(device.type, dtype=torch.float16)
           if device.type != "cpu" else torch.autocast("cpu", enabled=False))

    gt_sem = {}
    gt_npz = ROOT / "runs" / "gt_semantic_rle.npz"
    if gt_npz.exists():
        z = np.load(gt_npz, allow_pickle=True)
        for k in z.files:
            if k.startswith("sem::"):
                gt_sem[k[5:]] = str(z[k])
        print(f"GT semantic RLE cache: {len(gt_sem)} sets")
    train_ds = FilamentCropDataset(train_items, img_cache, gt_sem_rle=gt_sem)
    val_ds = FilamentCropDataset(val_items, img_cache, train=False, gt_sem_rle=gt_sem)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    best_val = 1e9
    history = []
    for ep in range(epochs):
        model.train()
        t0, tl, nb = time.time(), 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with ctx:
                logits = model(xb)
                loss = 0.5 * F.binary_cross_entropy_with_logits(logits, yb) + \
                       0.5 * soft_dice_loss(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item(); nb += 1
        model.eval()
        vl, vnb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                with ctx:
                    logits = model(xb)
                    loss = 0.5 * F.binary_cross_entropy_with_logits(logits, yb) + \
                           0.5 * soft_dice_loss(logits, yb)
                vl += loss.item(); vnb += 1
        history.append(dict(epoch=ep, train_loss=tl / max(nb, 1), val_loss=vl / max(vnb, 1)))
        print(f"ep {ep:02d} train {tl/max(nb,1):.4f} val {vl/max(vnb,1):.4f} "
              f"lr {sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)
        if vl / max(vnb, 1) < best_val:
            best_val = vl / max(vnb, 1)
            torch.save(model.state_dict(), run_dir / "best.pt")
    json.dump(history, open(run_dir / "history.json", "w"), indent=1)
    print(f"done. artifacts in {run_dir}")


if __name__ == "__main__":
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else VAL_FOLD
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else EPOCHS
    main(fold, epochs)
