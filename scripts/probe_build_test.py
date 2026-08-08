"""Quick CPU pre-flight: build the 3 larger Satlas models (pretrained=False),
print param counts, to catch constructor errors before the GPU probe runs."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cropseg.models import build_model  # noqa: E402

MODELS = [
    ("Sentinel2_SwinB_SI_RGB", "configs/finetune_satlas_rape_s2base.json"),
    ("Sentinel2_Resnet152_SI_RGB", "configs/finetune_satlas_rape_s2res152.json"),
    ("Sentinel2_Resnet50_SI_RGB", "configs/finetune_satlas_rape_s2res50.json"),
]

def main() -> None:
    for model_id, cfg_path in MODELS:
        try:
            cfg = {"backend": "satlas", "model_id": model_id, "decoder_channels": 128}
            model = build_model(cfg, 1, pretrained=False).eval()
            params = sum(p.numel() for p in model.parameters())
            with torch.no_grad():
                out = model(torch.zeros(1, 3, 64, 64))
            print(f"OK {model_id} params_m={params/1e6:.1f} out={tuple(out.shape)} "
                  f"finite={torch.isfinite(out).all().item()}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {model_id}: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
