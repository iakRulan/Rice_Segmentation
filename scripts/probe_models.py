"""Probe peak GPU memory for larger Satlas models at training settings.

For each candidate model, runs a realistic forward+backward under bf16 autocast
(mirroring finetune_v2) at batch 2 then batch 1, and picks the largest batch that
keeps the projected training footprint under a safety threshold.

Training peak ~= forward/backward peak + optimizer moments (fp32, params*8)
                     + EMA copy (fp32, params*4).
So proj = measured_peak + params*12.

Writes the chosen batch/accum into the per-model config on the fly.
Prints plan lines to stdout for `wait_and_launch.sh`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cropseg.models import build_model  # noqa: E402

# model_id -> (key, config path)
TARGETS = {
    "Sentinel2_SwinB_SI_RGB": ("s2base", "configs/finetune_satlas_rape_s2base.json"),
    "Sentinel2_Resnet152_SI_RGB": ("s2res152", "configs/finetune_satlas_rape_s2res152.json"),
    "Sentinel2_Resnet50_SI_RGB": ("s2res50", "configs/finetune_satlas_rape_s2res50.json"),
}

SAFE_PROJ_MB = 10000  # stay clear of the ~11.2 GiB usable on a 12 GiB card


def peak_mb(model: torch.nn.Module, batch: int, context: int = 512) -> float:
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, 3, context, context, device="cuda")
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = logits.float().pow(2).mean()
        loss.backward()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def main() -> None:
    if not torch.cuda.is_available():
        print("NO_CUDA", flush=True)
        return
    free = torch.cuda.mem_get_info()[0] / (1024 ** 2)
    print(f"free_mem_before={free:.0f}MB", flush=True)

    for model_id, (key, cfg_path) in TARGETS.items():
        cfg = json.loads((ROOT / cfg_path).read_text(encoding="utf-8"))
        try:
            model = build_model(cfg["model"], 1, pretrained=False).cuda().eval()
            params = sum(p.numel() for p in model.parameters())
        except Exception as exc:  # noqa: BLE001
            print(f"{key} BUILD_FAIL {exc}", flush=True)
            continue

        overhead_mb = params * 12 / (1024 ** 2)
        print(f"{key} params_m={params / 1e6:.1f} overhead_mb={overhead_mb:.0f}", flush=True)

        chosen = None
        for batch in (2, 1):
            try:
                peak = peak_mb(model, batch)
            except torch.cuda.OutOfMemoryError:
                print(f"{key} batch={batch} OOM", flush=True)
                peak = float("inf")
            proj = peak + overhead_mb
            print(f"{key} batch={batch} peak={peak:.0f}MB proj={proj:.0f}MB", flush=True)
            if proj < SAFE_PROJ_MB:
                chosen = (batch, 4 // batch)
                break
        del model
        torch.cuda.empty_cache()

        if chosen is None:
            print(f"{key} NO_FIT", flush=True)
            continue
        batch, accum = chosen
        cfg["train"]["batch_size"] = batch
        cfg["train"]["accumulation"] = accum
        cfg["train"]["val_batch_size"] = 2
        (ROOT / cfg_path).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(f"{key} CHOSEN batch={batch} accum={accum} -> {cfg_path}", flush=True)


if __name__ == "__main__":
    main()
