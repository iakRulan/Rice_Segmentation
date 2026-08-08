import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cropseg.config import load_experiment
from cropseg.metrics import mean_iou, search_threshold
from cropseg.models import center_crop
from cropseg.tasks import get_task


class CoreTests(unittest.TestCase):
    def test_task_mapping(self):
        self.assertEqual(get_task("rape").domain, "wheat_rape")
        self.assertEqual(get_task("rice").classes, 1)

    def test_empty_masks_score_one(self):
        probs = np.zeros((2, 1, 8, 8), dtype=np.float32)
        masks = np.zeros_like(probs)
        score, per_class = mean_iou(probs, masks)
        self.assertEqual(score, 1.0)
        self.assertEqual(per_class, [1.0])

    def test_threshold_search(self):
        probs = np.asarray([[[[0.4, 0.6]]]], dtype=np.float32)
        masks = np.asarray([[[[0.0, 1.0]]]], dtype=np.float32)
        score, threshold = search_threshold(probs, masks, 0.3, 0.7, 0.1)
        self.assertEqual(score, 1.0)
        self.assertGreaterEqual(threshold, 0.4)

    def test_center_crop(self):
        logits = torch.arange(64).reshape(1, 1, 8, 8)
        target = torch.zeros(1, 1, 4, 4)
        cropped = center_crop(logits, target)
        self.assertEqual(tuple(cropped.shape), (1, 1, 4, 4))
        self.assertEqual(int(cropped[0, 0, 0, 0]), 18)

    def test_config_validation(self):
        raw = {
            "name": "test", "data": {"root": "/tmp", "task": "rape"},
            "model": {"backend": "satlas", "model_id": "x"},
            "stages": [{"name": "warmup", "epochs": 1, "lr": 0.001}],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(load_experiment(path).name, "test")


if __name__ == "__main__":
    unittest.main()
