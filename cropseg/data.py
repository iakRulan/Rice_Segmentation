from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from .tasks import TaskSpec

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], np.float32)


def tile_id(name: str) -> int:
    return int(Path(name).stem.rsplit("_", 1)[1])


@dataclass(frozen=True)
class Record:
    split: str
    name: str


class MosaicStore:
    """Build native-scale context without resizing the 256px source tiles."""

    def __init__(self, root: str | Path, domain: str, grid_width: int = 83,
                 cache: bool = False):
        self.root = Path(root)
        self.grid_width = grid_width
        self.paths: dict[int, Path] = {}
        for split in ("train", "val", "testA"):
            folder = self.root / split / "image" / domain
            if folder.exists():
                for path in folder.glob("*.png"):
                    self.paths[tile_id(path.name)] = path
        if not self.paths:
            raise FileNotFoundError(f"no images found for domain={domain} under {self.root}")
        self._cache = None
        if cache:
            self._cache = {idx: self._read(path) for idx, path in self.paths.items()}

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    def get(self, idx: int, fallback: np.ndarray) -> np.ndarray:
        path = self.paths.get(idx)
        if path is None:
            return fallback
        return self._cache[idx] if self._cache is not None else self._read(path)

    def window(self, idx: int, size: int = 512) -> np.ndarray:
        if size < 256 or size > 768:
            raise ValueError("context size must be in [256, 768]")
        center = self.get(idx, np.zeros((256, 256, 3), np.uint8))
        row, col = divmod(idx - 1, self.grid_width)
        rows = []
        for dy in (-1, 0, 1):
            cells = []
            for dx in (-1, 0, 1):
                rr, cc = row + dy, col + dx
                neighbor = rr * self.grid_width + cc + 1
                # A single membership test covers every out-of-grid case the old
                # code missed (rr >= n_rows at the bottom edge, plus any tile id
                # that simply does not exist in the store). Missing neighbors
                # fall back to the center tile instead of fabricating context.
                invalid = neighbor not in self.paths
                cells.append(center if invalid else self.get(neighbor, center))
            rows.append(np.concatenate(cells, axis=1))
        mosaic = np.concatenate(rows, axis=0)
        start = (768 - size) // 2
        return mosaic[start:start + size, start:start + size]


def _records_in_split(root: Path, task: TaskSpec, split: str) -> list[Record]:
    folder = root / split / "image" / task.domain
    return [Record(split, path.name) for path in sorted(folder.glob("*.png"))]


def build_records(root: str | Path, task: TaskSpec, training: bool,
                  fold_file: str | None = None, fold: int | None = None) -> list[Record]:
    root = Path(root)
    if not fold_file:
        return _records_in_split(root, task, "train" if training else "val")
    if fold is None:
        raise ValueError("data.fold is required when data.fold_file is set")
    manifest = json.loads(Path(fold_file).read_text(encoding="utf-8"))
    held_out = {f"{item['split']}:{item['name']}" for item in manifest["folds"][str(fold)]}
    all_records = _records_in_split(root, task, "train") + _records_in_split(root, task, "val")
    return [r for r in all_records if (f"{r.split}:{r.name}" in held_out) != training]


def build_joint_records(root: str | Path, fold_file: str, fold: int):
    """Return (train_records, val_records, train_tile_ids) over ALL labeled tiles.

    The original train/val split is dissolved: every labeled tile (5478) is pooled
    and re-split by the isolated-row fold file. ``train_ids`` is the set of tile
    ids whose labels are public for this fold (used to build neighbor-row priors:
    a held-out row's labels must NOT leak into the prior, exactly like testA).
    """
    root = Path(root)
    manifest = json.loads(Path(fold_file).read_text(encoding="utf-8"))
    held_out = {f"{item['split']}:{item['name']}" for item in manifest["folds"][str(fold)]}
    all_records = []
    for split in ("train", "val"):
        folder = root / split / "image" / "rice"
        all_records += [Record(split, path.name) for path in sorted(folder.glob("*.png"))]
    train_records = [r for r in all_records if f"{r.split}:{r.name}" not in held_out]
    val_records = [r for r in all_records if f"{r.split}:{r.name}" in held_out]
    train_ids = {tile_id(r.name) for r in train_records}
    return train_records, val_records, train_ids


class MosaicSegDataset(Dataset):
    def __init__(self, root: str | Path, task: TaskSpec, records: Iterable[Record],
                 store: MosaicStore, context_size: int = 512,
                 target_size: int = 256, augment: bool = False,
                 normalization: str = "zero_one"):
        self.root = Path(root)
        self.task = task
        self.records = list(records)
        self.store = store
        self.context_size = context_size
        self.target_size = target_size
        self.augment = augment
        self.normalization = normalization
        if target_size != 256:
            raise ValueError("labels are native 256px; target_size currently must be 256")

    def __len__(self) -> int:
        return len(self.records)

    def load_mask(self, record: Record) -> np.ndarray:
        masks = []
        for label in self.task.labels:
            path = self.root / record.split / "label" / label / record.name
            masks.append((np.asarray(Image.open(path)) > 0).astype(np.float32))
        return np.stack(masks, axis=-1)

    def foreground_ratio(self, index: int) -> float:
        return float(self.load_mask(self.records[index]).mean())

    def __getitem__(self, index: int):
        record = self.records[index]
        image = self.store.window(tile_id(record.name), self.context_size)
        mask = self.load_mask(record)
        if self.augment:
            k = random.randrange(4)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
            if random.random() < 0.5:
                image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
            if random.random() < 0.5:
                image, mask = image[::-1].copy(), mask[::-1].copy()
            if random.random() < 0.5:
                gain, bias = random.uniform(0.9, 1.1), random.uniform(-10, 10)
                image = np.clip(image.astype(np.float32) * gain + bias, 0, 255)
        image = image.astype(np.float32) / 255.0
        if self.normalization == "imagenet":
            image = (image - IMAGENET_MEAN) / IMAGENET_STD
        elif self.normalization != "zero_one":
            raise ValueError(f"unknown normalization: {self.normalization}")
        x = torch.from_numpy(image.transpose(2, 0, 1)).float()
        y = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        return x, y, record.name


class MosaicImageDataset(Dataset):
    def __init__(self, root: str | Path, task: TaskSpec, split: str,
                 store: MosaicStore, context_size: int = 512,
                 normalization: str = "zero_one"):
        self.records = _records_in_split(Path(root), task, split)
        self.store = store
        self.context_size = context_size
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = self.store.window(tile_id(record.name), self.context_size).astype(np.float32) / 255.0
        if self.normalization == "imagenet":
            image = (image - IMAGENET_MEAN) / IMAGENET_STD
        elif self.normalization != "zero_one":
            raise ValueError(f"unknown normalization: {self.normalization}")
        return torch.from_numpy(image.transpose(2, 0, 1)).float(), record.name


class MosaicMultiTemporalDataset(Dataset):
    """Joint 3-class dataset over the two temporals (rice + wheat_rape).

    Input channels:
      * temporal="dual"  -> 6ch: [rice 3ch | wheat_rape 3ch] at context_size
      * temporal="wheat_rape" | "rice" -> 3ch from that temporal only
      * prior=True       -> +3ch neighbor-row label priors (wheat/rape/rice),
                            center tile's whole row zeroed (mimics testA).
    Target: center tile's 3 masks (wheat, rape, rice), 256x256.
    Only flips + rot180 are used for augmentation: rot90 would move the center
    tile out of the window center and desync image/prior/target.
    """

    def __init__(self, root: str | Path, records: Iterable[Record],
                 rice_store: MosaicStore, wr_store: MosaicStore,
                 train_ids: set[int], context_size: int = 768,
                 target_size: int = 256, augment: bool = False,
                 normalization: str = "imagenet",
                 temporal: str = "dual", prior: bool = True,
                 grid_width: int = 83):
        self.root = Path(root)
        self.records = list(records)
        self.rice_store = rice_store
        self.wr_store = wr_store
        self.train_ids = train_ids
        self.context_size = context_size
        self.target_size = target_size
        self.augment = augment
        self.normalization = normalization
        self.temporal = temporal
        self.prior = prior
        self.grid_width = grid_width
        if temporal not in ("rice", "wheat_rape", "dual"):
            raise ValueError(f"unknown temporal: {temporal!r}")
        if target_size != 256:
            raise ValueError("labels are native 256px; target_size currently must be 256")

    def __len__(self) -> int:
        return len(self.records)

    def _label_paths(self, name: str):
        for split in ("train", "val"):
            base = self.root / split / "label"
            if (base / "wheat" / name).exists():
                return base, split
        return self.root / "train" / "label", "train"

    def load_labels(self, tile: int) -> np.ndarray:
        """(256, 256, 3) float masks [wheat, rape, rice] for one tile."""
        name = f"clip_{tile:05d}.png"
        base, _ = self._label_paths(name)
        masks = []
        for label in ("wheat", "rape", "rice"):
            path = base / label / name
            masks.append((np.asarray(Image.open(path)) > 0).astype(np.float32))
        return np.stack(masks, axis=-1)

    def build_prior(self, center: int) -> np.ndarray:
        """768x768x3 neighbor-row labels; the center tile's entire row is zeroed.

        Only rows dy = -1/+1 (immediately above/below) are filled; a neighbor tile
        whose labels are not public for this fold (held-out/testA/testB/out-of-grid)
        contributes zero, matching what inference sees.
        """
        prior = np.zeros((768, 768, 3), np.float32)
        row, col = divmod(center - 1, self.grid_width)
        for dy in (-1, 1):
            for dx in (-1, 0, 1):
                rr, cc = row + dy, col + dx
                neighbor = rr * self.grid_width + cc + 1
                if neighbor in self.train_ids:
                    masks = self.load_labels(neighbor)
                    y0, x0 = (1 + dy) * 256, (1 + dx) * 256
                    prior[y0:y0 + 256, x0:x0 + 256] = masks
        start = (768 - self.context_size) // 2
        return prior[start:start + self.context_size, start:start + self.context_size]

    def _image(self, store: MosaicStore, tile: int) -> np.ndarray:
        return store.window(tile, self.context_size).astype(np.float32) / 255.0

    def __getitem__(self, index: int):
        record = self.records[index]
        tile = tile_id(record.name)
        if self.temporal == "dual":
            image = np.concatenate(
                [self._image(self.rice_store, tile), self._image(self.wr_store, tile)], axis=-1
            )
        else:
            store = self.rice_store if self.temporal == "rice" else self.wr_store
            image = self._image(store, tile)
        prior = self.build_prior(tile) if self.prior else None
        mask = self.load_labels(tile)  # (256, 256, 3)

        if self.augment:
            k = random.choice((0, 2))  # rot180 keeps the center tile in the center
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
            if prior is not None:
                prior = np.rot90(prior, k).copy()
            if random.random() < 0.5:
                image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
                if prior is not None:
                    prior = prior[:, ::-1].copy()
            if random.random() < 0.5:
                image, mask = image[::-1].copy(), mask[::-1].copy()
                if prior is not None:
                    prior = prior[::-1].copy()
            if random.random() < 0.5:
                gain, bias = random.uniform(0.9, 1.1), random.uniform(-10, 10)
                image = np.clip(image.astype(np.float32) * gain + bias, 0, 255)

        if self.normalization == "imagenet":
            n_img = image.shape[-1]
            mean = np.tile(IMAGENET_MEAN, n_img // 3)
            std = np.tile(IMAGENET_STD, n_img // 3)
            image = (image - mean) / std
        elif self.normalization != "zero_one":
            raise ValueError(f"unknown normalization: {self.normalization}")

        if prior is not None:
            image = np.concatenate([image, prior], axis=-1)
        x = torch.from_numpy(image.transpose(2, 0, 1)).float()
        y = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        return x, y, record.name


def make_area_sampler(dataset: MosaicSegDataset,
                      bucket_probs: list[float] | tuple[float, ...],
                      boundaries=(0.0, 0.02, 0.10)) -> WeightedRandomSampler:
    """Sample empty/tiny/small/large masks at explicit probabilities."""
    if len(bucket_probs) != len(boundaries) + 1:
        raise ValueError("bucket_probs must have one more item than boundaries")
    probs = np.asarray(bucket_probs, dtype=np.float64)
    if np.any(probs < 0) or probs.sum() <= 0:
        raise ValueError("bucket probabilities must be non-negative")
    probs /= probs.sum()
    ratios = np.asarray([dataset.foreground_ratio(i) for i in range(len(dataset))])
    buckets = np.digitize(ratios, boundaries, right=True)
    counts = np.bincount(buckets, minlength=len(probs))
    if np.any((counts == 0) & (probs > 0)):
        absent = np.flatnonzero((counts == 0) & (probs > 0)).tolist()
        raise ValueError(f"requested probability for empty sampler buckets: {absent}")
    weights = np.asarray([probs[b] / max(counts[b], 1) for b in buckets])
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(dataset), True)
