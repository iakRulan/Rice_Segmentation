from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageConfig:
    name: str
    epochs: int
    lr: float
    backbone_lr_scale: float = 0.1
    freeze_backbone: bool = False
    patience: int = 0


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    output_dir: str
    data: dict[str, Any]
    model: dict[str, Any]
    train: dict[str, Any]
    loss: dict[str, Any]
    stages: list[StageConfig] = field(default_factory=list)
    source_path: str = ""

    def validate(self) -> None:
        required_data = {"root", "task"}
        missing = required_data - self.data.keys()
        if missing:
            raise ValueError(f"data config missing: {sorted(missing)}")
        if self.model.get("backend") not in {"smp", "satlas"}:
            raise ValueError("model.backend must be 'smp' or 'satlas'")
        if not self.stages:
            raise ValueError("at least one training stage is required")
        if any(stage.epochs <= 0 or stage.lr <= 0 for stage in self.stages):
            raise ValueError("each stage needs positive epochs and lr")
        if int(self.data.get("context_size", 512)) < int(self.data.get("target_size", 256)):
            raise ValueError("context_size cannot be smaller than target_size")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "data": self.data,
            "model": self.model,
            "train": self.train,
            "loss": self.loss,
            "stages": [stage.__dict__ for stage in self.stages],
        }


def load_experiment(path: str | Path) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig(
        name=raw["name"],
        seed=int(raw.get("seed", 42)),
        output_dir=raw.get("output_dir", "weights/v2"),
        data=dict(raw["data"]),
        model=dict(raw["model"]),
        train=dict(raw.get("train", {})),
        loss=dict(raw.get("loss", {})),
        stages=[StageConfig(**stage) for stage in raw["stages"]],
        source_path=str(path),
    )
    cfg.validate()
    return cfg
