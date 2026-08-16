#!/usr/bin/env python
"""Run the full local pipeline: build caches, validate OOF PQ, make a submission.

Usage: python run_all.py [--fold 0] [--thr 0.5] [--ma 500]
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.check_call(cmd, shell=True, cwd=ROOT)


if __name__ == "__main__":
    fold = "0"
    thr = "0.5"
    ma = "500"
    args = sys.argv[1:]
    if "--fold" in args:
        fold = args[args.index("--fold") + 1]
    if "--thr" in args:
        thr = args[args.index("--thr") + 1]
    if "--ma" in args:
        ma = args[args.index("--ma") + 1]

    if not (ROOT / "runs" / "gt_semantic_rle.npz").exists():
        sh(f"{sys.executable} -u src/build_gt_cache.py")
    run = f"runs/unet_r34_v1_fold{fold}"
    if not (ROOT / run / "best.pt").exists():
        sh(f"{sys.executable} -u src/train_unet.py {fold} 24")
    sh(f"{sys.executable} -u src/eval_and_submit.py {run}")
    sh(f"{sys.executable} -u src/make_test_sub.py {run} {thr} {ma}")
    print("done — submission in subs/")
