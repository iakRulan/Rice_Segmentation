#!/usr/bin/env python3
"""Unified native-context inference for v2 checkpoints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cropseg.data import MosaicImageDataset, MosaicStore
from cropseg.models import build_model
from cropseg.tasks import get_task


def transform(x, rotation, flip):
    x = torch.rot90(x, rotation, (-2, -1))
    return torch.flip(x, (-1,)) if flip else x


def inverse(x, rotation, flip):
    if flip:
        x = torch.flip(x, (-1,))
    return torch.rot90(x, -rotation, (-2, -1))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="/root/competition_data/public")
    parser.add_argument("--split", choices=["val", "testA"], default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    task = get_task(checkpoint["task"])
    model = build_model(checkpoint["model_config"], task.classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda().eval()
    store = MosaicStore(args.data_root, task.domain, cache=args.cache)
    dataset = MosaicImageDataset(
        args.data_root, task, args.split, store,
        context_size=int(checkpoint.get("context_size", 512)),
        normalization=checkpoint.get("normalization", "zero_one"),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    transforms = [(0, False)] if not args.tta else [(r, f) for r in range(4) for f in (False, True)]
    crop = (int(checkpoint.get("context_size", 512)) - 256) // 2
    predictions = {}
    for images, names in loader:
        images = images.cuda(non_blocking=True)
        total = None
        for rotation, flip in transforms:
            with torch.amp.autocast("cuda"):
                probs = torch.sigmoid(model(transform(images, rotation, flip)))
            probs = inverse(probs, rotation, flip)
            total = probs if total is None else total + probs
        total = (total / len(transforms))[..., crop:crop + 256, crop:crop + 256]
        total = total.float().cpu().numpy()
        for index, name in enumerate(names):
            value = total[index].astype(np.float16)
            predictions[name] = value[0] if task.classes == 1 else value
        print(f"[{len(predictions)}/{len(dataset)}]", flush=True)
    np.savez(args.output, **predictions)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
