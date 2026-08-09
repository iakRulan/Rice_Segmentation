"""Honest 5-fold OOF evaluation for the joint (6ch dual-temporal) line.

For each fold, load ``weights/<exp_dir>/<name>_f{fold}/best.pth`` (embedded
``experiment`` config rebuilds model + val dataset), infer the fold's held-out
tiles, then concatenate all folds' probabilities and search thresholds on the
FULL OOF set (P1 protocol). Also reports per-fold IoU and saves the OOF
probability/name arrays for later blending.

Usage:
    python scripts/oof_eval.py --config configs/joint_256_6ch.json --folds 5 \
        --out outputs/oof_joint_256_6ch
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from cropseg.config import load_experiment
from cropseg.data import build_joint_records, MosaicMultiTemporalDataset, MosaicStore
from cropseg.engine import center_crop
from cropseg.metrics import mean_iou, search_threshold
from cropseg.models import build_model, init_first_conv
from cropseg.tasks import get_task


def infer_fold(fold: int, base_config: dict, root: Path, fold_file: Path,
               out_dir: Path):
    chk = torch.load(out_dir / f"joint_256_6ch_f{fold}" / "best.pth",
                     map_location="cpu", weights_only=False)
    exp = chk["experiment"]
    task = get_task(chk["task"])
    mcfg = exp["model"]
    data = exp["data"]
    context_size = int(data.get("context_size", 256))
    normalization = data.get("normalization", "imagenet")
    grid_width = int(data.get("grid_width", 83))
    temporal = str(data.get("temporal", "dual"))
    prior = bool(data.get("prior", False))

    _, val_records, train_ids = build_joint_records(root, fold_file, fold)
    rice_store = MosaicStore(root, "rice", grid_width, False)
    wr_store = MosaicStore(root, "wheat_rape", grid_width, False)
    val_set = MosaicMultiTemporalDataset(
        root=root, records=val_records, rice_store=rice_store, wr_store=wr_store,
        train_ids=train_ids, context_size=context_size,
        target_size=int(data.get("target_size", 256)), normalization=normalization,
        temporal=temporal, prior=prior, grid_width=grid_width, augment=False,
    )
    loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4,
                        pin_memory=True, drop_last=False)

    model = build_model(mcfg, task.classes, pretrained=False).cuda()
    if mcfg.get("init_first_conv"):
        init_first_conv(model, int(mcfg["in_channels"]), int(mcfg.get("n_img", 3)))
    model.load_state_dict(chk["ema_state_dict"])
    model.eval()

    probs, targets, names = [], [], []
    with torch.no_grad():
        for images, truth, name in loader:
            images = images.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                out = model(images)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            logits = center_crop(out, truth)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            targets.append(truth.numpy())
            names.extend(name)
    probs = np.concatenate(probs, 0)
    targets = np.concatenate(targets, 0)
    fixed, per_class = mean_iou(probs, targets, 0.5)
    tuned, thr = search_threshold(probs, targets)
    print(f"fold {fold}: n={len(probs)} fixed={fixed:.4f} "
          f"per_class={[round(c, 3) for c in per_class]} tuned={tuned:.4f}@{thr:.2f}")
    return dict(names=names, probs=probs, targets=targets,
                fixed=float(fixed), per_class=[float(c) for c in per_class])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="outputs/oof")
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    root = Path(cfg.data["root"])
    fold_file = Path(cfg.data["fold_file"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_probs, all_targets, all_names, per_fold = [], [], [], []
    for fold in range(args.folds):
        r = infer_fold(fold, cfg.as_dict(), root, fold_file, out_dir)
        all_probs.append(r["probs"])
        all_targets.append(r["targets"])
        all_names.extend(r["names"])
        per_fold.append(r)

    probs = np.concatenate(all_probs, 0)
    targets = np.concatenate(all_targets, 0)
    assert probs.shape[0] == 5478, f"OOF must cover 5478 tiles, got {probs.shape[0]}"

    fixed, per_class = mean_iou(probs, targets, 0.5)
    tuned, thr = search_threshold(probs, targets)
    print("\n=== HONEST 5-FOLD OOF ===")
    print(f"tiles={probs.shape[0]} fixed(0.5)={fixed:.4f} "
          f"per_class={[round(c, 3) for c in per_class]}")
    print(f"tuned={tuned:.4f}@{thr:.2f}")
    summary = dict(
        n=int(probs.shape[0]), fixed=float(fixed), tuned=float(tuned),
        threshold=float(thr), per_class=[float(c) for c in per_class],
        per_fold=per_fold,
    )
    (out_dir / "oof_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    np.save(out_dir / "oof_probs.npy", probs)
    np.save(out_dir / "oof_targets.npy", targets)
    with open(out_dir / "oof_names.txt", "w") as f:
        f.write("\n".join(all_names))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
