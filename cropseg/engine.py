from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

from .metrics import mean_iou, search_threshold
from .models import center_crop


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if self.shadow[key].is_floating_point():
                self.shadow[key].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = state


@dataclass
class ValidationResult:
    fixed_iou: float
    class_iou: list[float]
    tuned_iou: float
    threshold: float


def cosine_with_warmup(optimizer, updates: int, warmup_fraction: float = 0.05,
                       min_scale: float = 0.02):
    warmup = max(1, round(updates * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup:
            return max((step + 1) / warmup, min_scale)
        progress = min(1.0, (step - warmup) / max(1, updates - warmup))
        return min_scale + (1.0 - min_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, ema,
                accumulation: int, grad_clip: float) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    skipped = 0
    for step, (images, targets, _) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = center_crop(model(images), targets)
        # Keep the model forward in AMP, but do loss reductions in FP32.
        # Lovasz/top-k reductions and per-image sums can overflow in fp16.
        with torch.amp.autocast("cuda", enabled=False):
            loss = criterion(logits.float(), targets.float())
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            skipped += 1
            print(f"[skip] nonfinite forward/loss at step={step}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue
        scaled_loss = loss / accumulation
        scaler.scale(scaled_loss).backward()
        total += float(loss.detach())
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            old_scale = scaler.get_scale()
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip, error_if_nonfinite=True
                )
            except RuntimeError:
                skipped += 1
                print(f"[skip] nonfinite gradient at step={step}", flush=True)
                # Let GradScaler record the overflow and reduce its scale;
                # do not advance the scheduler or EMA for a skipped update.
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= old_scale:
                scheduler.step()
                ema.update(model)
    if skipped:
        print(f"[stability] skipped_updates={skipped}", flush=True)
    return total / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, state_dict=None) -> ValidationResult:
    original = None
    if state_dict is not None:
        original = copy.deepcopy(model.state_dict())
        model.load_state_dict(state_dict)
    model.eval()
    probabilities, targets = [], []
    for images, truth, _ in loader:
        images = images.cuda(non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = center_crop(model(images), truth)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        targets.append(truth.numpy())
    probabilities = np.concatenate(probabilities)
    targets = np.concatenate(targets)
    fixed, per_class = mean_iou(probabilities, targets, 0.5)
    tuned, threshold = search_threshold(probabilities, targets)
    if original is not None:
        model.load_state_dict(original)
    return ValidationResult(fixed, per_class, tuned, threshold)
