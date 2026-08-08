#!/usr/bin/env python3
"""Build leakage-resistant folds with ISOLATED rows (mimics testA).

testA holds out entire raster rows (row % 10 == 8); every such row's immediate
neighbors are labeled train rows. A fold where val = ``position % k`` over the
sorted labeled rows reproduces that geometry: each val row's labeled neighbors
(above/below) fall in other folds, so context/prior methods are evaluated the
same way they run at inference. Also keeps ``{"folds": {fold: [{split, name}]}}``
schema that ``cropseg.data.build_records`` reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cropseg.data import tile_id
from cropseg.tasks import get_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/public")
    parser.add_argument("--task", choices=["wheat_rape", "wheat", "rape", "rice"], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--grid-width", type=int, default=83)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", choices=["isolated-rows", "contiguous"],
                        default="isolated-rows")
    args = parser.parse_args()
    task = get_task(args.task)
    root = Path(args.data_root)
    records = []
    for split in ("train", "val"):
        for path in sorted((root / split / "image" / task.domain).glob("*.png")):
            row = (tile_id(path.name) - 1) // args.grid_width
            records.append({"split": split, "name": path.name, "row": row})
    rows = sorted({item["row"] for item in records})
    if args.folds < 2 or args.folds > len(rows):
        raise ValueError("fold count must be between 2 and the number of labeled rows")
    if args.strategy == "isolated-rows":
        # Consecutive labeled rows land in different folds, so a held-out row's
        # immediate labeled neighbors are always in the training set.
        row_to_fold = {row: position % args.folds for position, row in enumerate(rows)}
    else:  # contiguous: keep the old band behavior
        row_to_fold = {row: min(args.folds - 1, position * args.folds // len(rows))
                       for position, row in enumerate(rows)}
    folds = {str(index): [] for index in range(args.folds)}
    for item in records:
        folds[str(row_to_fold[item["row"]])].append(
            {"split": item["split"], "name": item["name"]}
        )
    manifest = {
        "version": 3,
        "strategy": args.strategy,
        "task": task.name,
        "grid_width": args.grid_width,
        "rows": rows,
        "folds": folds,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[folds] strategy={args.strategy} rows={len(rows)} per-fold={[len(v) for v in folds.values()]}")
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
