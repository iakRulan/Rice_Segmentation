#!/usr/bin/env python3
"""Config-driven fine-tuning entry point for SMP and Satlas models."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cropseg.config import load_experiment
from cropseg.data import MosaicSegDataset, MosaicStore, build_records, make_area_sampler
from cropseg.engine import EMA, cosine_with_warmup, train_epoch, validate
from cropseg.losses import SegmentationLoss
from cropseg.models import build_model, set_backbone_trainable
from cropseg.tasks import get_task


def resolve(path: str, base: Path = ROOT) -> str:
    value = Path(path).expanduser()
    return str(value if value.is_absolute() else (base / value).resolve())


def save_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def make_optimizer(model, lr: float, backbone_scale: float, weight_decay: float):
    backbone = [p for p in model.backbone_parameters() if p.requires_grad]
    head = [p for p in model.head_parameters() if p.requires_grad]
    groups = [{"params": head, "lr": lr}]
    if backbone:
        groups.append({"params": backbone, "lr": lr * backbone_scale})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    task = get_task(cfg.data["task"])
    data_root = Path(resolve(cfg.data["root"]))
    fold_file = cfg.data.get("fold_file")
    if fold_file:
        fold_file = resolve(fold_file)
    train_records = build_records(data_root, task, True, fold_file, cfg.data.get("fold"))
    val_records = build_records(data_root, task, False, fold_file, cfg.data.get("fold"))
    print(f"[data] task={task.name} train={len(train_records)} val={len(val_records)}")
    if args.dry_run:
        print(json.dumps(cfg.as_dict(), indent=2))
        return

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True

    context_size = int(cfg.data.get("context_size", 512))
    normalization = cfg.data.get(
        "normalization", "zero_one" if cfg.model["backend"] == "satlas" else "imagenet"
    )
    store = MosaicStore(
        data_root, task.domain, int(cfg.data.get("grid_width", 83)),
        bool(cfg.data.get("cache", False)),
    )
    common = dict(
        root=data_root, task=task, store=store, context_size=context_size,
        target_size=int(cfg.data.get("target_size", 256)), normalization=normalization,
    )
    train_set = MosaicSegDataset(records=train_records, augment=True, **common)
    val_set = MosaicSegDataset(records=val_records, augment=False, **common)
    sampler = None
    if cfg.train.get("bucket_probs"):
        sampler = make_area_sampler(train_set, cfg.train["bucket_probs"])
    workers = int(cfg.train.get("workers", 2))
    train_loader = DataLoader(
        train_set, batch_size=int(cfg.train.get("batch_size", 2)),
        shuffle=sampler is None, sampler=sampler, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=int(cfg.train.get("val_batch_size", cfg.train.get("batch_size", 2))),
        shuffle=False, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0, drop_last=False,
    )

    resume = Path(args.resume).resolve() if args.resume else None
    model = build_model(cfg.model, task.classes, pretrained=resume is None).cuda()
    ema = EMA(model, float(cfg.train.get("ema_decay", 0.999)))
    history, global_epoch, best = [], 0, -1.0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("train_state_dict", checkpoint["model_state_dict"]))
        ema.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
        history = checkpoint.get("history", [])
        global_epoch = int(checkpoint.get("epoch", 0))
        best = float(checkpoint.get("best_fixed_iou", -1.0))
        print(f"[resume] {resume} epoch={global_epoch} best={best:.6f}")

    criterion = SegmentationLoss(cfg.loss).cuda()
    output = Path(resolve(cfg.output_dir)) / cfg.name
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "resolved_config.json", cfg.as_dict())
    accumulation = int(cfg.train.get("accumulation", 1))
    grad_clip = float(cfg.train.get("grad_clip", 1.0))
    weight_decay = float(cfg.train.get("weight_decay", 1e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    stop_all = False
    for stage_index, stage in enumerate(cfg.stages):
        set_backbone_trainable(model, not stage.freeze_backbone)
        optimizer = make_optimizer(model, stage.lr, stage.backbone_lr_scale, weight_decay)
        updates = math.ceil(len(train_loader) / accumulation) * stage.epochs
        scheduler = cosine_with_warmup(
            optimizer, updates, float(cfg.train.get("warmup_fraction", 0.05))
        )
        stale = 0
        print(
            f"[stage {stage_index + 1}/{len(cfg.stages)}] {stage.name} "
            f"epochs={stage.epochs} freeze_backbone={stage.freeze_backbone} lr={stage.lr:g}"
        )
        for stage_epoch in range(1, stage.epochs + 1):
            global_epoch += 1
            started = time.time()
            loss = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, scaler, ema,
                accumulation, grad_clip,
            )
            result = validate(model, val_loader, ema.state_dict())
            row = {
                "epoch": global_epoch,
                "stage": stage.name,
                "stage_epoch": stage_epoch,
                "loss": loss,
                "fixed_iou": result.fixed_iou,
                "class_iou": result.class_iou,
                "tuned_iou": result.tuned_iou,
                "threshold": result.threshold,
                "lr": [group["lr"] for group in optimizer.param_groups],
                "seconds": round(time.time() - started, 1),
            }
            history.append(row)
            improved = result.fixed_iou > best
            if improved:
                best, stale = result.fixed_iou, 0
            else:
                stale += 1
            payload = {
                "schema_version": 2,
                "epoch": global_epoch,
                "best_fixed_iou": best,
                "model_config": cfg.model,
                "task": task.name,
                "normalization": normalization,
                "context_size": context_size,
                "threshold": result.threshold,
                "model_state_dict": ema.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "train_state_dict": model.state_dict(),
                "history": history,
                "experiment": cfg.as_dict(),
            }
            torch.save(payload, output / "last.pth")
            if improved:
                torch.save(payload, output / "best.pth")
            save_json(output / "history.json", history)
            print(
                f"[ep {global_epoch:03d}] loss={loss:.4f} fixed={result.fixed_iou:.6f} "
                f"tuned={result.tuned_iou:.6f}@{result.threshold:.2f} "
                f"best={best:.6f} time={row['seconds']:.0f}s",
                flush=True,
            )
            if stage.patience and stale >= stage.patience:
                print(f"[early-stop] stage={stage.name} stale={stale}")
                break
        if stop_all:
            break
    print(f"[done] best_fixed_iou={best:.6f} output={output}")


if __name__ == "__main__":
    main()
