#!/bin/bash
# Power-cut-resilient local fold training: skips folds with complete history.json
cd "$(dirname "$0")"
for f in 1 2 3 4; do
  d=runs/unet_r34_v1_fold$f
  if [ -f "$d/history.json" ] && [ "$(python -c "import json;print(len(json.load(open('$d/history.json'))))" 2>/dev/null)" = "24" ]; then
    echo "fold $f already complete, skipping"
    continue
  fi
  echo "=== training fold $f ==="
  python -u src/train_unet.py $f 24 > runs/train_fold$f.log 2>&1 || { echo "fold $f FAILED"; exit 1; }
done
echo DONE > runs/folds_done.txt
