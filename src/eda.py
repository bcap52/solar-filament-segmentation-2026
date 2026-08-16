"""EDA: annotation statistics, GT overlap check, inter-annotator PQ ceiling.

Findings (2026-08-16): 411 JPEGs have 1 annotator set, 145 have 2, 151 have 3.
Instances: 8199, mean 7.10/annotator-image. Areas: median 1228px, p97.5 9818px.
Expert-vs-expert PQ = 0.332 (SQ 0.633, RQ 0.525). Union-of-annotators + CC
instancing reaches PQ 0.619 against each individual annotator — the consensus
view is the effective target.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_annotations, group_by_stem  # noqa: E402
from src.pq import PQAccumulator  # noqa: E402


def main():
    by_key = load_annotations()
    by_stem = group_by_stem(by_key)
    print(f"annotator-image sets: {len(by_key)} | physical JPEGs: {len(by_stem)}")
    mult = Counter(len(v) for v in by_stem.values())
    print("annotator sets per JPEG:", dict(sorted(mult.items())))

    areas, fills, elongs, n_inst = [], [], [], []
    overlap_pairs = total_pairs = 0
    for ai in by_key.values():
        n_inst.append(len(ai.instances))
        for inst in ai.instances:
            m = inst.mask()
            areas.append(int(m.sum()))
            x, y, w, h = inst.bbox
            fills.append(m.sum() / max(1.0, w * h))
            elongs.append(max(w, h) / max(1.0, min(w, h)))
        ms = [i.mask() for i in ai.instances]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                total_pairs += 1
                if (ms[i] & ms[j]).sum() > 0:
                    overlap_pairs += 1
    areas = np.array(areas)
    print(f"instances: {len(areas)} | per set mean {np.mean(n_inst):.2f}")
    print(f"area px: median {np.median(areas):.0f} p97.5 {np.percentile(areas, 97.5):.0f}")
    print(f"intra-set overlapping GT pairs: {overlap_pairs}/{total_pairs}")

    multi = {s: v for s, v in by_stem.items() if len(v) >= 2}
    acc12, acc21 = PQAccumulator(), PQAccumulator()
    for stem, sets in multi.items():
        sets = sorted(sets, key=lambda a: a.image_key)
        m1 = [i.mask() for i in sets[0].instances]
        m2 = [i.mask() for i in sets[1].instances]
        acc12.add(stem, m2, m1)
        acc21.add(stem, m1, m2)
    for name, res in [("set1->set2", acc12.result()), ("set2->set1", acc21.result())]:
        print(f"  inter-annotator {name}: PQ={res.pq:.4f} SQ={res.sq:.4f} RQ={res.rq:.4f}")

    acc_c = PQAccumulator()
    for stem, sets in multi.items():
        union_mask = np.zeros((2048, 2048), np.uint8)
        for s in sets:
            for i in s.instances:
                union_mask |= i.mask()
        n_cc, labeled = cv2.connectedComponents(union_mask, connectivity=8)
        cc_masks = [(labeled == k).astype(np.uint8) for k in range(1, n_cc)]
        for s in sets:
            acc_c.add(stem, [i.mask() for i in s.instances], cc_masks)
    res_c = acc_c.result()
    print(f"  union-CC vs each annotator: PQ={res_c.pq:.4f} SQ={res_c.sq:.4f} RQ={res_c.rq:.4f}")


if __name__ == "__main__":
    main()
