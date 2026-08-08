#!/usr/bin/env python3
"""Create leakage-resistant folds from contiguous raster row bands."""
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
    parser.add_argument("--data-root", default="/root/competition_data/public")
    parser.add_argument("--task", choices=["wheat_rape", "wheat", "rape", "rice"], required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--grid-width", type=int, default=83)
    parser.add_argument("--output", required=True)
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
        raise ValueError("fold count must be between 2 and the number of raster rows")
    row_to_fold = {}
    for position, row in enumerate(rows):
        row_to_fold[row] = min(args.folds - 1, position * args.folds // len(rows))
    folds = {str(index): [] for index in range(args.folds)}
    for item in records:
        fold = row_to_fold[item.pop("row")]
        folds[str(fold)].append(item)
    manifest = {
        "version": 2,
        "strategy": "contiguous_raster_rows",
        "task": task.name,
        "grid_width": args.grid_width,
        "rows": rows,
        "folds": folds,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({key: len(value) for key, value in folds.items()})
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
